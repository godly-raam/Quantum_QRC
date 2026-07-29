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
    for _ in range(layers):  # layer index unused; _ signals intentionally discarded
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

def train_reservoir_offline(num_qubits: int, layers: int = 2, maxiter: int = 100) -> np.ndarray:
    """
    Optimizes the fixed parameters to maximize the reservoir's expressivity
    (output variance across different logistics input states).

    Added maxiter param so training runs can be compared quickly in tests.
    """
    qc = build_parameterized_reservoir(num_qubits, layers)
    rng = np.random.default_rng(42)
    
    # Generate 5 random input states simulating different logistics graphs
    input_circuits = []
    for _ in range(5):
        circ = QuantumCircuit(num_qubits)
        # Use RY to rotate qubits off the Z-pole. RY keeps states real-valued and
        # therefore causes measurable population changes in the Z-basis (unlike RZ,
        # which only adds phases on the |0> state). This change fixes a previous
        # bug where all training inputs were physically identical under Z measurements.
        for i in range(num_qubits):
            circ.ry(rng.uniform(0, 2*np.pi), i)
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
            # Use Statevector.from_instruction to ensure proper circuit->state conversion
            sv = Statevector.from_instruction(full_circ)
            exp_vals = [np.real(sv.expectation_value(op)) for op in observables]
            all_expectations.append(exp_vals)
            
        # True Expressivity: Maximize the variance of outputs
        # We return the negative sum because scipy minimizes
        variance = np.var(all_expectations, axis=0)
        return -np.sum(variance)
        
    initial_params = rng.uniform(0, 2 * np.pi, num_qubits * layers)

    # Sanity check: evaluate expressivity on several random parameter vectors
    sample_params = [rng.uniform(0, 2 * np.pi, num_qubits * layers) for _ in range(3)]
    sample_scores = [-expressivity_cost(p) for p in sample_params]
    print(f"Sanity-check expressivity scores (pre-optimization): {sample_scores}")
    # Assert that at least one sample shows non-zero expressivity; helps detect regressions
    assert max(sample_scores) > 1e-8, (
        "Expressivity sanity check failed: all sample scores are ~0. "
        "Input encoding may be ineffective (e.g., using RZ on |0> only adds phases)."
    )

    print("Starting rigorous expressivity training (Maximizing variance)...")
    
    # Run optimization
    result = minimize(expressivity_cost, initial_params, method='COBYLA', options={'maxiter': maxiter})
    
    print(f"Training Complete. Expressivity score: {-result.fun:.4f}")
    return result.x
