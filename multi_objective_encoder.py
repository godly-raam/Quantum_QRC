import numpy as np
from qiskit import QuantumCircuit

def generate_scalarized_cost_matrix(Q1: np.ndarray, Q2: np.ndarray) -> tuple:
    """
    Takes two competing logistics cost matrices (e.g., Distance and Time)
    and combines them using a randomly sampled convex weight vector.
    """
    # Sample a weight for the first objective between 0 and 1
    c1 = np.random.uniform(0, 1)
    # The second weight ensures the sum is exactly 1 (convex combination)
    c2 = 1.0 - c1
    # Create the unified cost matrix
    Q_c = c1 * Q1 + c2 * Q2
    return Q_c, (c1, c2)


def build_qrc_feature_map(Q_c: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """
    Encodes the scalarized cost matrix into a quantum circuit.
    This acts as the input layer before your reservoir dynamics take over.
    """
    qc = QuantumCircuit(num_qubits)

    # Initialize in an equal superposition (standard for QUBO/QAOA encodings)
    qc.h(range(num_qubits))

    # Encode linear terms (the diagonal of the cost matrix)
    for i in range(num_qubits):
        # We use RZ gates to imprint the local cost biases
        if Q_c[i, i] != 0:
            qc.rz(Q_c[i, i], i)

    # Encode quadratic terms (the off-diagonal routing connections)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if Q_c[i, j] != 0:
                # We use RZZ gates to imprint the connectivity/interaction costs
                qc.rzz(Q_c[i, j], i, j)

    return qc


if __name__ == "__main__":
    num_nodes = 4
    # Mock data: Q1 (e.g., Fuel Cost) and Q2 (e.g., Delivery Time)
    # In production, replace these with your actual logistics graph adjacencies
    Q_fuel = np.random.rand(num_nodes, num_nodes)
    Q_time = np.random.rand(num_nodes, num_nodes)

    # Step A: Scalarize
    Q_combined, weights = generate_scalarized_cost_matrix(Q_fuel, Q_time)
    print(f"Sampled Weights - Fuel: {weights[0]:.3f}, Time: {weights[1]:.3f}")

    # Step B: Encode into Quantum Circuit
    encoding_circuit = build_qrc_feature_map(Q_combined, num_nodes)
    print("\nFeature Map Circuit Depth:", encoding_circuit.depth())
    print(encoding_circuit.draw(fold=-1))
