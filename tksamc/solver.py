import numpy as np
import math
import random
import time
import sys
from numba import jit

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
            energies: microstate energies (J/mol)
            weights: normalized Boltzmann weights
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

    energies = None
    weights = None
    if return_microstate_energies:
        energies = np.zeros(total_states, dtype=np.float64)
        weights = np.zeros(total_states, dtype=np.float64)

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

    Gqq = 0.5 * Gqq_num / sum_Zn

    if return_microstate_energies:
        return Gqq, energies, weights / np.sum(weights)

    return Gqq

@jit(nopython=True)
def _solve_mc_jit(Eij, charges, pkas, pH, T, steps, equil_steps):
    """
    JIT-compiled Monte Carlo loop.
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
            E_interaction_all = 0.0005 * current_charges * (np.dot(Eij, current_charges))

            E_total += E_interaction_all
            E_total_sq += E_interaction_all**2
            
            # calculating the energy of the current microstate and recording it in the sampling distribution
            # Use the same sign convention as the exact solver: -pKa contribution + interaction energy.
            term1 = np.dot(pkas, current_charges) * np.log(10) * R * T
            term2 = 0.5 * np.sum(current_charges * np.dot(Eij, current_charges))
            E_microstate = -term1 + term2

            sampling_dist[step - equil_steps] = E_microstate
    count = steps - equil_steps
    avg_E = E_total / count

    return (avg_E, sampling_dist)

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


def plot_mc_sampling_distribution(sampling_dist, bins=50, density=True, color='C0'):
    """Plot the MC microstate energy sampling distribution.

    Args:
        sampling_dist (array-like): Sampled microstate energies from the MC solver.
        bins (int): Number of histogram bins.
        density (bool): If True, plot density instead of counts.
        color (str): Bar color.

    Returns:
        matplotlib.pyplot: The pyplot module after plotting.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(sampling_dist, bins=bins, density=density, alpha=0.8, color=color)
    plt.xlabel('Microstate energy')
    plt.ylabel('Probability density' if density else 'Count')
    plt.title('MC microstate energy sampling distribution')
    plt.grid(True)
    return plt


def plot_exact_microstate_distribution(energies, weights=None, bins=50, density=True, color='C1'):
    """Plot the exact microstate energy distribution.

    Args:
        energies (array-like): Exact microstate energies (J/mol).
        weights (array-like or None): Boltzmann weights for each state.
        bins (int): Number of histogram bins.
        density (bool): If True, normalize the histogram.
        color (str): Bar color.

    Returns:
        matplotlib.pyplot: The pyplot module after plotting.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(energies, bins=bins, weights=weights, density=density, alpha=0.8, color=color)
    plt.xlabel('Microstate energy')
    plt.ylabel('Probability density' if density else 'Count')
    plt.title('Exact microstate energy distribution')
    plt.grid(True)
    return plt

