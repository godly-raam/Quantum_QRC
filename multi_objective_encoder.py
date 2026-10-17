import numpy as np
from qiskit import QuantumCircuit


def generate_scalarized_cost_matrix(
    Q1: np.ndarray,
    Q2: np.ndarray,
    weights: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple:
    """
    Takes two competing logistics cost matrices (e.g., Distance and Time)
    and combines them using a convex weight vector.

    Passing ``weights`` or ``rng`` makes the scalarization reproducible.
    """
    if weights is None:
        generator = rng if rng is not None else np.random.default_rng()
        c1 = float(generator.uniform(0.0, 1.0))
    else:
        c1 = float(weights[0])
    c2 = 1.0 - c1
    Q_c = c1 * Q1 + c2 * Q2
    return Q_c, (c1, c2)


def normalize_cost_matrix(Q_c: np.ndarray) -> np.ndarray:
    """Normalize a cost matrix by its largest absolute entry."""
    scale = float(np.max(np.abs(Q_c)))
    if scale == 0.0:
        return np.array(Q_c, dtype=float, copy=True)
    return np.asarray(Q_c, dtype=float) / scale


def build_qrc_feature_map(
    Q_c: np.ndarray,
    num_qubits: int,
    angle_scale: float = 1.0,
    use_depot_diagonal: bool = False,
) -> QuantumCircuit:
    """
    Encodes the scalarized cost matrix into a quantum circuit.

    The matrix is normalized by ``max(abs(Q))`` before angles are computed.
    ``angle_scale`` then scales the normalized angles; values below one reduce
    the exponential concentration of local expectation values.
    ``use_depot_diagonal`` writes the normalized depot distance into the
    diagonal so the RZ terms (and the resulting <Y_i>) carry distance
    information even when the raw distance matrix has a zero diagonal.
    """
    Q_c = np.asarray(Q_c, dtype=float)
    Q_normalized = normalize_cost_matrix(Q_c)

    if use_depot_diagonal:
        Q_normalized = Q_normalized.copy()
        for i in range(1, Q_normalized.shape[0]):
            Q_normalized[i, i] = Q_normalized[0, i]

    qc = QuantumCircuit(num_qubits)

    # Initialize in an equal superposition (standard for QUBO/QAOA encodings)
    qc.h(range(num_qubits))

    # Encode linear terms (the diagonal of the cost matrix)
    for i in range(num_qubits):
        if Q_normalized[i, i] != 0:
            qc.rz(angle_scale * Q_normalized[i, i], i)

    # Encode quadratic terms (the off-diagonal routing connections)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if Q_normalized[i, j] != 0:
                qc.rzz(angle_scale * Q_normalized[i, j], i, j)

    return qc


if __name__ == "__main__":
    num_nodes = 4
    # Mock data: Q1 (e.g., Fuel Cost) and Q2 (e.g., Delivery Time)
    # In production, replace these with your actual logistics graph adjacencies
    rng = np.random.default_rng(0)
    Q_fuel = rng.random((num_nodes, num_nodes))
    Q_time = rng.random((num_nodes, num_nodes))

    # Step A: Scalarize
    Q_combined, weights = generate_scalarized_cost_matrix(Q_fuel, Q_time, rng=rng)
    print(f"Sampled Weights - Fuel: {weights[0]:.3f}, Time: {weights[1]:.3f}")

    # Step B: Encode into Quantum Circuit
    encoding_circuit = build_qrc_feature_map(Q_combined, num_nodes)
    print("\nFeature Map Circuit Depth:", encoding_circuit.depth())
    print(encoding_circuit.draw(fold=-1))
