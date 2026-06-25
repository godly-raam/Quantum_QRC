import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector, SparsePauliOp

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
    Optimizes the fixed parameters to maximize the reservoir's expressivity
    (output variance across different logistics input states).
    """
    qc = build_parameterized_reservoir(num_qubits, layers)
    rng = np.random.default_rng(42)
    
    # Generate 5 random input states simulating different logistics graphs
    input_circuits = []
    for _ in range(5):
        circ = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circ.rz(rng.uniform(0, 2*np.pi), i)
        input_circuits.append(circ)
        
    # Define observables (Z expectations for each qubit)
    observables = []
    for i in range(num_qubits):
        pauli_str = ['I'] * num_qubits
        pauli_str[num_qubits - 1 - i] = 'Z'
        observables.append(SparsePauliOp("".join(pauli_str)))
        
    def expressivity_cost(params):
        bound_qc = qc.assign_parameters(params)
        all_expectations = []
        
        # Evaluate how the reservoir scatters the different inputs
        for in_circ in input_circuits:
            full_circ = in_circ.compose(bound_qc)
            sv = Statevector(full_circ)
            exp_vals = [np.real(sv.expectation_value(op)) for op in observables]
            all_expectations.append(exp_vals)
            
        # True Expressivity: Maximize the variance of outputs
        # We return the negative sum because scipy minimizes
        variance = np.var(all_expectations, axis=0)
        return -np.sum(variance)
        
    initial_params = rng.uniform(0, 2 * np.pi, num_qubits * layers)
    print("Starting rigorous expressivity training (Maximizing variance)...")
    
    # Run optimization
    result = minimize(expressivity_cost, initial_params, method='COBYLA', options={'maxiter': 100})
    
    print(f"Training Complete. Expressivity score: {-result.fun:.4f}")
    return result.x
