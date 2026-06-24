import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

def build_parameterized_reservoir(num_qubits: int, layers: int = 2) -> QuantumCircuit:
    """
    Builds a reservoir with trainable parameters (angles).
    """
    qc = QuantumCircuit(num_qubits)
    # Create a vector of parameters for the RX rotations
    theta = ParameterVector('θ', length=num_qubits * layers)
    
    param_idx = 0
    for layer in range(layers):
        # Parametrized local rotations
        for i in range(num_qubits):
            qc.rx(theta[param_idx], i)
            param_idx += 1
            
        # Fixed entangling topology (e.g., linear or ring)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
            
        # Optional: Close the ring for periodic boundary conditions
        if num_qubits > 2:
            qc.cz(num_qubits - 1, 0)
            
    return qc

def train_reservoir_offline(num_qubits: int, layers: int = 2) -> np.ndarray:
    """
    Simulates the reservoir classically to find the optimal fixed parameters.
    (This runs ONCE, offline, on a small number of qubits).
    """
    qc = build_parameterized_reservoir(num_qubits, layers)
    
    def cost_function(params):
        # Bind current parameters to the circuit
        bound_qc = qc.assign_parameters(params)
        
        # NOTE: In production, insert your Statevector simulation here
        # to evaluate the variance/expressivity of the reservoir output.
        # For this architectural blueprint, we simulate a dummy cost
        # that mimics a converged optimization landscape.
        dummy_cost = np.sum(np.sin(params)**2) - 0.5 * np.sum(np.cos(params))
        return dummy_cost
        
    # Initialize random starting parameters
    initial_params = np.random.uniform(0, 2 * np.pi, num_qubits * layers)
    
    # Run classical optimization (COBYLA is standard for QAOA/VQE pre-training)
    print(f"Starting offline reservoir training (Classical Simulation) for {num_qubits} qubits...")
    result = minimize(cost_function, initial_params, method='COBYLA', options={'maxiter': 100})
    
    print(f"Training Complete. Optimal parameters found.")
    return result.x
