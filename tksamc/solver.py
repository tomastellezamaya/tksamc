import numpy as np
import math
import random
import time
import sys
from numba import jit
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
# Constants
PI = 3.14159265359
EP = 4.0
ES = 78.5
K_BOLTZ = 1.3806488e-23
R = 8.314
EO = 8.8541878176e-12
E_CHARGE = 1.602e-19
X_CONST = 0.1

def solve_exact(Eij, charges, pkas, pH, T, return_microstate_energies=False):
    """
    Exact solution for TKSA.

    Args:
        Eij: Interaction matrix (NxN).
        charges: Initial charges (N).
        pkas: pKa values (N).
        pH: pH value.
        T: Temperature.
        return_microstate_energies: If True, also return exact microstate energies and Boltzmann weights.

    Returns:
        Gqq: Free energy per residue (N).
        If return_microstate_energies is True, also returns:
            energies: List of microstate energies .
            weights: List of corresponding Boltzmann weights.
            accumulated_energy_per_res: corresponding per-residue energy contributions for each microstate (shape: (num_states, N)).
    """
    n = len(charges)
    states = (1 << n) - 1  # 2^n - 1

    Zn = 0.0
    Zu = 0.0

    RT = R * T
    ln10RT = np.log(10) * RT
    ln10pH = np.log(10) * pH

    batch_size = 50000

    total_states = 1 << n

    sum_Zn = 0.0
    sum_Zu = 0.0

    if return_microstate_energies:
        energies = np.zeros(total_states, dtype=np.float64)
        weights = np.zeros(total_states, dtype=np.float64)
        accumulated_energy_per_res = np.zeros((total_states, n), dtype=np.float64)

    # We need to accumulate Gqq.
    # Since we can't store everything, we accumulate contribution to numerator.
    Gqq_num = np.zeros(n, dtype=np.float64)
    
    # Pre-compute charges q0
    q0 = charges

    num_batches = (total_states + batch_size - 1) // batch_size

    # print(f"Processing {total_states} states in {num_batches} batches...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_states)
        current_batch_size = end_idx - start_idx

        # indices: (current_batch_size,)
        indices = np.arange(start_idx, end_idx, dtype=np.int64)

        # X: (current_batch_size, n)
        # We need to unpack bits.
        # This creates the binary matrix.
        X = ((indices[:, None] & (1 << np.arange(n))) > 0).astype(np.float64)

        # Current charges Q = q0 + X
        Q = q0 + X

        # Term 1: Sum over a of (pKa_a * (q0_a + X_a))
        # Wait, C code: `termo1 = (Eij[a][m])*(Eij[a][0] + X[a-1]);`
        # Eij[a][m] is pKa. Eij[a][0] is q0.
        # So Term1 is correct.
        Term1 = np.sum(pkas * Q, axis=1)

        # Term 2: 0.5 * Sum over a, k of (E_ak * Q_a * Q_k)
        QE = Q @ Eij
        Term2 = 0.5 * np.sum(Q * QE, axis=1)

        # Gn = -Term1 * ln10RT + Term2
        Gn = -Term1 * ln10RT + Term2

        # Gu = -Term1 * ln10RT
        Gu = -Term1 * ln10RT

        # vi = sum(X)
        vi = np.sum(X, axis=1)

        # Term3 = vi * ln10pH
        Term3 = vi * ln10pH

        # Boltzmann factors
        # The C code uses: exp( -(Gn)/(R*T) - termo3 )
        # Gn is energy (J/mol). RT is energy (J/mol). Gn/RT is dimensionless.
        # termo3 (vi*ln10*pH) is dimensionless.
        # So correct expression is exp( -Gn/RT - Term3 ).

        exp_factor_n = np.exp(-Gn/RT - Term3)

        sum_Zn += np.sum(exp_factor_n)

        # Zu calculation
        # C code: `Zu = Zu + exp( -(Gu)/(R*T) - termo3);`

        exp_factor_u = np.exp(-Gu/RT - Term3)

        sum_Zu += np.sum(exp_factor_u)

        if return_microstate_energies:
            energies[start_idx:end_idx] = Gn
            weights[start_idx:end_idx] = exp_factor_n
            # Store per-residue energy contributions for each microstate (unsorted)
            accumulated_energy_per_res[start_idx:end_idx, :] = 0.5 * Q * QE

        # Accumulate Gqq
        # C code: `GC = (exp( -(Gn)/(R*T)  -  vi*(log(10))*PH))/Zn ;`
        # Wait, C code divides by Zn inside the loop?
        # But Zn is fully calculated in FIRST pass.
        # Ah, C code has TWO passes.
        # First pass: Calculate Zn, Zu.
        # Second pass: Calculate Gqq using Zn.

        # Here we do it in one pass (conceptually), but we need Zn to normalize.
        # So we accumulate the numerator: `Interaction * exp_factor`.
        # Then divide by sum_Zn at the end.

        Interaction_energy_per_res = 0.5 * Q * QE # (batch, n)

        Gqq_num += np.sum(Interaction_energy_per_res * exp_factor_n[:, None], axis=0)

    Gqq = 0.0005 * Gqq_num / sum_Zn

    if return_microstate_energies:
        indices = np.arange(2**n)
        #microstates = ((indices[:, None] & (1 << np.arange(n))) > 0).astype(int)
        return Gqq, energies, weights, accumulated_energy_per_res
    return Gqq

def sort_by_energy(energies, weights, per_residue_energy):
    """Sort microstates by energy and return sorted microstates, energies, weights, and per-residue energy."""
    sorted_indices = np.argsort(energies)
    sorted_microstates = sorted_indices
    sorted_energies = energies[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_per_residue_energy = per_residue_energy[sorted_indices]
    return sorted_microstates, sorted_energies, sorted_weights, sorted_per_residue_energy

def sort_by_weight(energies, weights, per_residue_energy):
    """Sort microstates by boltzmann weight and return sorted microstates, energies, weights, and per-residue energy."""
    sorted_indices = np.argsort(weights)[::-1] # Sort in descending order of weight
    sorted_microstates = sorted_indices
    sorted_energies = energies[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_per_residue_energy = per_residue_energy[sorted_indices]
    return sorted_microstates, sorted_energies, sorted_weights, sorted_per_residue_energy

@jit(nopython=True)
def _solve_mc_jit(Eij, charges, pkas, pH, T, steps, equil_steps):
    """
    JIT-compiled Monte Carlo loop.

    Args:
        Eij: Interaction matrix (NxN).
        charges: Initial charges (N).
        pkas: pKa values (N).
        pH: pH value.
        T: Temperature.
        steps: how many steps to run the sampling
        equil_steps: steps to equilibrate the system before starting to sample
    """
    n = len(charges)
    # Replicating C code behavior exactly, including constants.

    current_charges = np.zeros(n, dtype=np.float64)

    # Accumulators
    E_total = np.zeros(n, dtype=np.float64)
    E_total_sq = np.zeros(n, dtype=np.float64)

    # rng = np.random.default_rng() # Numba supports simple random functions
    LN10 = np.log(10.0)

    descorrela = 200
    sampling_dist = np.zeros(steps - equil_steps, dtype=np.float64)

    for step in range(steps):
        for _ in range(descorrela):
            res_idx = random.randint(0, n-1)

            old_q = current_charges[res_idx]
            new_q = 0.0

            base_charge = charges[res_idx]
            pka = pkas[res_idx]

            # Logic from C code to determine new charge and parte2
            parte2_diff = 0.0

            if base_charge == 0:
                if old_q == 0:
                    new_q = 1.0
                    parte2_diff = (pH - pka)
                else:
                    new_q = 0.0
                    parte2_diff = -(pH - pka)
            else: # Acidic
                if old_q == 0:
                    new_q = -1.0
                    parte2_diff = -(pH - pka)
                else:
                    new_q = 0.0
                    parte2_diff = (pH - pka)

            # Calculate DeltaE
            # DeltaE = (Enew - Eold) + parte2*log(10)
            # Enew - Eold = 0.0005 * (new_q - old_q) * Sum(E_ij * q_j)

            # Using loop for dot product to be safe with numba in older versions, but dot is supported
            interaction_sum = np.dot(Eij[res_idx, :], current_charges) - Eij[res_idx, res_idx]*current_charges[res_idx]
            delta_interaction = (new_q - old_q) * interaction_sum * 0.0005

            delta_E = delta_interaction + parte2_diff * LN10

            accept = False
            if delta_E <= 0:
                accept = True
            else:
                if np.exp(-delta_E) > random.random():
                    accept = True

            if accept:
                current_charges[res_idx] = new_q

        if step >= equil_steps:
            # ET1 loop in C code
            # for (a=1;a<=n;a++) ET1 = sum(0.0005 * E * q * q)
            # This is 0.0005 * q_a * sum(E_ak * q_k)

            # Calculate vector of interactions
            # (n,) vector
            # Dot product Eij * current_charges -> vector
            # Then elementwise multiply by current_charges * 0.0005

            # np.dot(Eij, current_charges) returns vector of size n
            Eij_q = np.dot(Eij, current_charges)
            E_interaction_all = 0.0005 * current_charges * Eij_q

            E_total += E_interaction_all
            E_total_sq += E_interaction_all**2
            
            # calculating the energy of the current microstate and recording it in the sampling distribution
            # Use the same sign convention as the exact solver: -pKa contribution + interaction energy.
            term1 = np.dot(pkas, current_charges) * np.log(10) * R * T
            term2 = 0.5 * np.sum(current_charges * Eij_q)
            E_microstate = -term1 + term2

            sampling_dist[step - equil_steps] = E_microstate
    count = steps - equil_steps
    avg_E = E_total / count

    return avg_E, sampling_dist

@jit(nopython=True)
def solve_mc_by_weights(Eij, charges, pkas, pH, T, steps = 100000, equil_steps = 1000):
  n = len(charges)
  current_charges = np.zeros(n, dtype=np.float64)
  E_total = np.zeros(n, dtype=np.float64)
  LN10 = np.log(10.0)
  ln10RT = LN10 * R * T
  descorrela = 200
  sampling_dist = np.zeros(steps - equil_steps, dtype=np.float64)
  index_sampling = np.zeros((steps - equil_steps, n), dtype=np.float64)

  Eij_q = np.zeros(n, dtype=np.float64)

  cached_E = 0.0
  cached_Vi = np.sum(current_charges - charges)

  for step in range(steps):
      for _ in range(descorrela):
        res_idx = random.randint(0, n - 1)
        old_q = current_charges[res_idx]
        new_q = 0.0

        base_charge = charges[res_idx]
        pka = pkas[res_idx]

        if base_charge == 0:
            if old_q == 0:
                new_q = 1.0
            else:
                new_q = 0.0
        else: # Acidic
            if old_q == 0:
                new_q = -1.0
            else:
                new_q = 0.0

      dq = new_q - old_q   # ±1 or ±2

      delta_interact = dq * (Eij_q[res_idx] + 0.5 * Eij[res_idx, res_idx] * dq)
      delta_pka      = -dq * pkas[res_idx] * ln10RT
      delta_E       = delta_pka + delta_interact

      new_E = cached_E + delta_E
      new_Vi = cached_Vi + dq

      # Boltzmann weights from cached values — no copy, no full recompute
      old_B = np.exp(-cached_E / (R * T) - cached_Vi * LN10 * pH)
      new_B = np.exp(-new_E / (R * T) - new_Vi * LN10 * pH)

      if new_B >= old_B:
        accept = True
      else:
        accept = new_B / old_B > random.random()

      if accept:
        current_charges[res_idx] = new_q
        for j in range(n): 
          Eij_q[j] += Eij[j, res_idx] * dq
        cached_E = new_E
        cached_Vi = new_Vi

      if step >= equil_steps:
        E_interaction_all = 0.0005 * current_charges * Eij_q 
        E_total += E_interaction_all
        term1 = np.dot(pkas, current_charges) * LN10 * R * T
        term2 = 0.5 * np.sum(current_charges * Eij_q)
        sampling_dist[step - equil_steps] = -term1 + term2
  count = steps - equil_steps
  avg_E = E_total / count
  G_res = avg_E * (0.0083145 * T) / 5.0
  return G_res, sampling_dist
            
def solve_mc(Eij, charges, pkas, pH, T, steps=100000, equil_steps=1000):
    """
    Monte Carlo solution wrapper.

    Returns:
        tuple: (G_res, sampling_dist)
            G_res: Estimated residue energies.
            sampling_dist: Array of sampled microstate energies after equilibration.
    """
    convert = 0.0083145 * T
    avg_E, sampling_dist = _solve_mc_jit(Eij, charges, pkas, pH, T, steps, equil_steps)
    G_res = avg_E * convert / 5.0
    return G_res, sampling_dist

def plot_mc_sampling_distribution(sampling_dist, bins=150, color='C0'):
    """Plot the MC microstate energy sampling distribution.

    Args:
        sampling_dist (array-like): Sampled microstate energies from the MC solver.
        bins (int): Number of histogram bins.
        color (str): Bar color.

    Returns:
        matplotlib.pyplot: The pyplot module after plotting.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(sampling_dist, bins=bins, alpha=0.8, color=color)
    plt.xlabel('Microstate index')
    plt.ylabel('Count of states')
    plt.title('MC microstate energy sampling distribution')
    plt.grid(True)
    return plt

def plot_ex_vs_mc(exact_energies, exact_weights, mc_sampling_dist, mc_type, figsize=(14, 5), bins=150,  color = 'tab:green'):
    """Compare exact microstate energies vs MC-sampled microstate energies.

    Visualization of how well MC explores the energy landscape.

    Args:
        exact_energies (array-like): Energies of allowed microstates (J/mol).
        exact_weights (array-like): Weights of allowed microstates.
        mc_sampling_dist (array-like): MC-sampled microstate energies (J/mol).
        mc_type (str): Type of MC sampling.
        figsize (tuple): Figure size.
        bins (int): Number of histogram bins.

    Returns:
        matplotlib.pyplot: The pyplot module after plotting.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Left: Overlay histograms
    color_hist = color
    ax1.set_xlabel('Energy of States')
    ax1.set_ylabel('Count of times sampled by MC', color='darkblue')
    ax1.hist(mc_sampling_dist, bins=bins, color=color_hist, edgecolor='black', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor= color)
    
    color_line = 'tab:red'
    ax2.set_ylabel('State weights', color=color_line)
    ax2.scatter(exact_energies, exact_weights, color=color_line, linewidth=2.5)
    plt.title(mc_type+" sampling vs. Exact microstates")
    plt.tight_layout()
    return plt

def error_limit_helper(microstate_weights, microstate_per_residue_energy):
    """
    Exact solution for TKSA with a limited state space (annealing helper).

    Args:
        microstate_per_residue_energy: Pre-computed per-residue energy contribution for each microstate (shape: (num_states, n)).
        microstate_weights: Pre-computed weights for each microstate (shape: (num_states,)).
        neither microstate_per_residue_energy nor microstate_weights need to be sorted, but they must correspond to the same microstates.
    Returns:
        Gqq: Free energy per residue (N).
    """
    G_qq = np.sum(microstate_per_residue_energy * microstate_weights[:, None], axis=0) / np.sum(microstate_weights)
    return 0.5 * G_qq

def error_limit_exact_microstates(Gqq_full, sorted_weights, sorted_per_residue_energy, rate=0.01):
    """Slowly lower the amount of states considered in the exact solver by applying an annealing-like approach.
    to find out how much of the state space is needed to get close to the exact solution.
    
    Args:
        rate: Fraction of states to keep at each iteration (0 < rate < 1).
        Gqq_full: pre-computed full Gqq from exact solver for error calculation.
        sorted_weights: pre-computed sorted weights from exact solver for state space restriction.
        sorted_per_residue_energy: pre-computed per-residue energy contributions for each microstate.
    
    """
    total_states = len(sorted_weights)
    error_tracker = {}
    
    # Calculate how many states to drop per iteration step
    step_size = int(total_states * rate)
    if step_size < 1:
        step_size = 1
    min_states = max(1, int(total_states * 0.01))

    # Loop using exact integer counts from total_states down to min_states
    for current_count in range(total_states, min_states - 1, -step_size):
        
        # Calculate true percentage for the dictionary key mapping
        current_percentage = current_count / total_states
        
        weight_space = sorted_weights[:current_count]
        energy_space = sorted_per_residue_energy[:current_count]
        
        # Execute the helper calculation on restricted space
        Gqq_reduced = error_limit_helper(weight_space, energy_space)
        
        # Track the Euclidean norm error
        error = np.linalg.norm(Gqq_full - Gqq_reduced)
        error_tracker[current_percentage] = error

    return error_tracker
    
def plot_error_limit_results(error_tracker):
    """Plot the error as a function of the percentage of states considered in the exact solver."""
    import matplotlib.pyplot as plt

    percentages = list(error_tracker.keys())
    errors = list(error_tracker.values())

    plt.figure(figsize=(8, 5))
    plt.plot(percentages, errors, marker='o')
    plt.xlabel('Percentage of states considered')
    plt.ylabel('Error (norm of Gqq difference)')
    plt.title('Error vs Percentage of States in Exact Solver')
    plt.grid(True)
    return plt

def plot_microstate_counts(energies, bins=200, color='C1'):
    """Plot the exact microstate energy distribution as a histogram.

    Args:
        energies (array-like): Exact microstate energies (J/mol).
        bins (int): Number of histogram bins.
        color (str): Bar color.

    Returns:
        matplotlib.pyplot: The pyplot module after plotting.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(energies, bins=bins, alpha=0.8, color=color)
    plt.xlabel('Microstate energy')
    plt.ylabel('Count of states')
    plt.title('Exact microstate energy distribution')
    plt.grid(True)
    return plt

def plot_energy_vs_weights(energies, weights):
    """Plot microstate energy vs Boltzmann weight to visualize the energy landscape and state contributions.
    
    Args:
    energies (array-like): Exact microstate energies (J/mol).
    weights (array-like): Boltzmann weights corresponding to the microstates.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(energies, weights, alpha=0.6)
    plt.xlabel('Microstate Energy (J/mol)')
    plt.ylabel('Boltzmann Weight')
    plt.title('Microstate Energy vs Boltzmann Weight')
    plt.grid(True)
    return plt

def weights_to_energies(microstates, weights, pH, T):
    """
    Convert Boltzmann weights to energies for sampled microstates.

    Args:
        microstates: Array of sampled microstates (n_samples, N).
        weights: Corresponding Boltzmann weights for each microstate.
        pH: pH value.
        T: Temperature.
        """
    RT = R * T
    ln10 = np.log(10)
    return -RT * (np.log(weights) + ln10 * pH * np.sum(microstates, axis=1))

def build_bqm(Eij, charges, pkas, pH, T):
    """
    Builds the bqm for sampling
    """
    n = len(charges)
    ln10 = np.log(10)
    RT = R * T
    Eq = Eij @ charges

    # Vectorized computation
    linear_coefficients = ln10*(pH - pkas) + (Eq) / RT
    diagonal = np.diag(Eij) / (2 * RT)
    
    constant = np.dot(charges, Eq) / (2 * RT) - ln10 * np.dot(charges, pkas)
     
    linear = dict(enumerate(linear_coefficients + diagonal))
    
    # Only include non-zero quadratic terms (sparse representation)
    iu, ju = np.triu_indices(n, k=1)
    vals = Eij[iu, ju]
    mask = vals !=0
    quadratic = dict(zip(zip(iu[mask].tolist(), ju[mask].tolist()),
                          (vals[mask] / RT).tolist()))
    
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, constant, 'BINARY')
    return bqm

def get_e_diff(bqm, n):
    sampler = sampler = dimod.SimulatedAnnealingSampler()
    initial_sample = sampler.sample(bqm, num_reads=5, num_sweeps=100)
    best = initial_sample.first    
    rand_samples = np.random.randint(0, 2, size=(1000, n))
    sample_like = (rand_samples, list(bqm.variables))
    rand_energy_avg = np.average(bqm.energies(sample_like))
    e_diff = rand_energy_avg - best.energy
    return e_diff, best

def single_bit_flip(X):
    """
    Given an (m, n) array of binary vectors, return a deduplicated
    array containing each original vector and its n single-bit-flip
    neighbors.
    """
    Xi = X.astype(np.int64)

    m, n = Xi.shape
    flip_mask = np.eye(n, dtype=np.int64)
    flipped = Xi[:, None, :] ^ flip_mask[None, :, :]
    flipped = flipped.reshape(m * n, n)

    combined = np.vstack([Xi, flipped])
    result = np.unique(combined, axis=0)

    return result.astype(np.float64) 

def sample_qa(Eij, charges, pkas, pH, T, mult):
    """
    return a set of unique microstates that maximize the boltzmann weight of the system using quantum annealing.

    Args:
        Eij: Interaction matrix (NxN).
        charges: Initial charges (N).
        pkas: pKa values (N).
        pH: pH value.
        T: Temperature.
        samples: Number of annealing samples.
    """
    n = len(charges)
    variables = range(n)
    microstates = []
    samples = n*20
    oversample = 2
    oversample_growth = 1.7
    max_batches = 10

    bqm = build_bqm(Eij, charges, pkas, pH, T)
    sampler = dimod.SimulatedAnnealingSampler()
    seen_states = set()

    # calculating possible lagrange multplier value 
    # taking the difference between the energy of a random state (most likely very high energy/low weight)
    # and the best possible state
    
    e_diff, best = get_e_diff(bqm, n)

    lagrange = e_diff*mult   
    # adding the best state to seen
    best_state = np.array([best.sample[i] for i in range(n)], dtype=np.float64)
    seen_states.add(tuple(best_state))
    microstates.append(best_state)

    # adding new constraint to avoid sampling the best state again
    
    terms = [(var, 1-2*best_state[var]) for var in variables]
    bqm.add_linear_inequality_constraint(
        terms,
        lb = 1-np.sum(best_state),
        lagrange_multiplier = lagrange,
        label = "penalty_1"            
    )

    sampleset = sampler.sample(bqm, num_reads=samples, num_sweeps=10).aggregate()
    recs = sampleset.record
    num_unique = len(sampleset)
    print("Found "+str(num_unique)+"/"+str(samples)+" unique samples")
    # filtering out the slack variables from adding constraint
    # column_idxs is a list of the indeces in the raw variables data that contains the original vaiables so we can access only the ones we care about
    raw_variables = sampleset.variables
    column_idxs = [raw_variables.index(var) for var in variables]
    
    for rec in recs:
        state = rec.sample[column_idxs]
        key = tuple(state.tolist())
        if key in seen_states:
            continue
        seen_states.add(key)
        microstates.append(state.astype(np.float64))
        if len(microstates) == samples:
            break
    expanded_states = single_bit_flip(np.array(microstates))
    print("Used "+str(len(expanded_states))+" total states after expansion")
              
    #return np.asarray(microstates), np.asarray(weights)
    return expanded_states
   
def solve_qa(Eij, charges, pkas, pH, T, mult=None):
    """
    Solve TKSA using quantum annealing and return the average per-residue energies from unique sampled states.

    Args:
        Eij: Interaction matrix (NxN).
        charges: Initial charges (N).
        pkas: pKa values (N).
        pH: pH value.
        T: Temperature.
        samples: Number of annealing samples.
        run_local: Whether to run the quantum annealing locally.
        """
    n = len(charges)
    if mult == None:
        mult = 1e-1
    microstates = sample_qa(Eij, charges, pkas, pH, T, mult)
    
    Q = charges + microstates
    QE = Q @ Eij
    QQE = Q*QE
    RT = R * T
    ln10RT = np.log(10) * RT
    ln10pH = np.log(10)*pH
    Term1 = np.sum(pkas * Q, axis=1)
    Term2 = 0.5 * np.sum(QQE, axis=1)
    Gn = -Term1 * ln10RT + Term2
    vi = np.sum(microstates, axis=1)
    Term3 = vi * ln10pH
    interaction_energy_per_res = 0.0005 * QQE
    weights = np.exp(-Gn/RT - Term3)
    approx_Zn = np.sum(weights)
    Gqq_avg = np.sum(interaction_energy_per_res * weights[:, None], axis=0) / approx_Zn
    energies = weights_to_energies(microstates, weights, pH, T)
    return Gqq_avg/2, energies, weights
    #return Gqq_avg/2

def plot_mc_vs_qa(mc_sampling, qa_energies, qa_weights):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # Left: MC sampling histogram
    ax1.set_xlabel('Energy of States')
    ax1.set_ylabel('Count of times sampled by MC', color='darkblue')
    ax1.hist(mc_sampling, bins=150, color='blue', edgecolor='black', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Right: QA unique samples as a scatter
    color_scatter = 'tab:red'
    ax2.set_ylabel('QA Boltzmann weight', color=color_scatter)
    ax2.scatter(qa_energies, qa_weights, color=color_scatter, linewidth=2.5, zorder=3)
    ax2.tick_params(axis='y', labelcolor=color_scatter)

    plt.title('MC sampling vs. QA Sampling')
    plt.tight_layout()
    return plt

def plot_ex_vs_qa(ex_energies, ex_weights, qa_energies,  qa_weights):
    import matplotlib.pyplot as plt
    plt.figure(figsize = (14,5))
    plt.scatter(ex_energies, ex_weights, color = 'blue', linewidth = 2.5, label = 'EX model')
    plt.scatter(qa_energies, qa_weights, color = 'red', linewidth = 2.5, label = 'QA model')
    plt.title("EX vs QA model sampling")
    plt.xlabel("Energy")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    return plt

def find_optimal_multiplier(Eij, charges, pkas, pH, T):
    import matplotlib.pyplot as plt
    mc, _ = solve_mc_by_weights(Eij, charges, pkas, pH, T, steps = 100000, equil_steps = 1000)
    errors = []
    mults = np.linspace(0, .05, 101)
    for mult in mults:
        qa = solve_qa(Eij, charges, pkas, pH, T, mult, run_local=True)
        errors.append(np.linalg.norm(mc-qa))
    plt.figure(figsize=(10, 6))
    plt.scatter(mults, errors, alpha=0.6)
    plt.xlabel('E_diff multiplier value')
    plt.ylabel('MC vs QA error')
    plt.title('Multiplier Value vs Error')
    plt.grid(True)
    return mults[np.argmin(errors)], plt
