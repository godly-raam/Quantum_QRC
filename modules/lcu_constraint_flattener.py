import numpy as np
from qiskit import QuantumCircuit

def sample_lcu_branch(num_qubits: int, target_k: int, gamma: float = 1.0) -> float:
    """
    Computes the discrete Fourier coefficients for a quadratic penalty
    f(x) = (1^T x - k)^2 and samples a single LCU branch angle (theta_j).
    """
    m = num_qubits
    c_coeffs = np.zeros(m + 1, dtype=complex)
    theta_vals = np.zeros(m + 1)
    
    # Calculate discrete Fourier coefficients
    for j in range(m + 1):
        theta_j = 2 * np.pi * j / (m + 1)
        theta_vals[j] = theta_j
        
        # Sum over possible Hamming weights k_val from 0 to m
        coeff_sum = 0
        for k_val in range(m + 1):
            penalty = (k_val - target_k)**2
            coeff_sum += np.exp(-1j * gamma * penalty) * np.exp(-1j * theta_j * k_val)
        
        c_coeffs[j] = coeff_sum / (m + 1)
        
    # Calculate sampling probabilities q_j = |c_j| / Gamma
    magnitudes = np.abs(c_coeffs)
    gamma_cost = np.sum(magnitudes)
    probabilities = magnitudes / gamma_cost
    
    # Classically sample a branch j based on the LCU probabilities
    rng = np.random.default_rng(42)
    sampled_j = rng.choice(range(m + 1), p=probabilities)
    
    return theta_vals[sampled_j]

def build_lcu_constraint_layer(num_qubits: int, sampled_theta: float) -> QuantumCircuit:
    """
    Builds the sampled LCU basis circuit.
    Notice this uses NO entangling gates and NO ancilla qubits.
    """
    qc = QuantumCircuit(num_qubits)
    
    # The LCU basis for permutation-invariant diagonal unitaries
    # reduces to a uniform layer of single-qubit Rz gates.
    for i in range(num_qubits):
        qc.rz(sampled_theta, i)
        
    return qc
