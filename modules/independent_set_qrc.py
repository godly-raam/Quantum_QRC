"""QRC-based maximum-independent-set benchmark helpers.

The cost layer uses QOBLIB's maximum-independent-set QUBO, while the variational
QAOA mixer is replaced by the fixed circuit family used by ``QuantumReservoir``.
The resulting bitstrings are repaired classically before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from modules.quantum_reservoir_vrp import QuantumReservoir


@dataclass(frozen=True)
class IndependentSetGraph:
    """Undirected graph with zero-based internal vertex identifiers."""

    num_vertices: int
    edges: tuple[tuple[int, int], ...]

    @property
    def degrees(self) -> np.ndarray:
        values = np.zeros(self.num_vertices, dtype=int)
        for source, target in self.edges:
            values[source] += 1
            values[target] += 1
        return values


def parse_dimacs_graph(path: str | Path) -> IndependentSetGraph:
    """Parse QOBLIB's DIMACS ``p edge`` graph format."""
    num_vertices: int | None = None
    edges: list[tuple[int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] == "c":
            continue
        if fields[0] == "p":
            if len(fields) != 4 or fields[1] != "edge":
                raise ValueError(f"Unsupported DIMACS problem declaration: {line}")
            num_vertices = int(fields[2])
        elif fields[0] == "e":
            if num_vertices is None or len(fields) != 3:
                raise ValueError(f"Edge found before a valid problem declaration: {line}")
            source, target = int(fields[1]) - 1, int(fields[2]) - 1
            if not 0 <= source < num_vertices or not 0 <= target < num_vertices:
                raise ValueError(f"Edge endpoint outside declared graph: {line}")
            if source != target:
                edges.append((min(source, target), max(source, target)))

    if num_vertices is None:
        raise ValueError(f"No DIMACS problem declaration in {path}")
    return IndependentSetGraph(num_vertices, tuple(sorted(set(edges))))


def parse_optimal_solution(path: str | Path) -> set[int]:
    """Read one-indexed vertex identifiers from QOBLIB's ``.opt.sol`` format."""
    return {
        int(line.strip()) - 1
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def qubo_and_ising_terms(
    graph: IndependentSetGraph, penalty: float = 2.0
) -> tuple[np.ndarray, np.ndarray, dict[int, float], dict[tuple[int, int], float]]:
    """Build QUBO and its constant-free minimization Ising Hamiltonian.

    QOBLIB's maximization objective is ``sum_i x_i - penalty * sum_(i,j) x_i x_j``.
    The returned QUBO is its minimization negative.  With ``x_i=(1-Z_i)/2``:

    ``H_obj = 1/2 sum_i Z_i`` and
    ``H_const = penalty/4 sum_(i,j) (Z_i Z_j - Z_i - Z_j)``.

    Constant offsets are omitted, as they do not affect samples or optimization.
    """
    if penalty <= 1.0:
        raise ValueError("penalty must exceed 1 to discourage violated edges")

    qubo = -np.eye(graph.num_vertices, dtype=float)
    for source, target in graph.edges:
        qubo[source, target] += penalty

    local_terms = {vertex: 0.5 for vertex in range(graph.num_vertices)}
    pair_terms: dict[tuple[int, int], float] = {}
    for source, target in graph.edges:
        local_terms[source] -= penalty / 4.0
        local_terms[target] -= penalty / 4.0
        pair_terms[(source, target)] = penalty / 4.0
    return qubo, np.diag(qubo).copy(), local_terms, pair_terms


def _reservoir_ansatz(num_qubits: int, seed: int, layers: int = 2) -> QuantumCircuit:
    """Build the existing QuantumReservoir circuit family without state allocation.

    ``QuantumReservoir.__init__`` allocates a full statevector, which is unsuitable
    for the 52-vertex instance.  This proxy invokes its existing circuit-building
    method only; it does not alter the VRP reservoir implementation.
    """
    proxy = QuantumReservoir.__new__(QuantumReservoir)
    proxy.n_qubits = num_qubits
    proxy.reservoir_layers = layers
    proxy.trained_params = np.random.default_rng(seed).uniform(
        0.0, 2.0 * np.pi, size=num_qubits * layers
    )
    return proxy.build_reservoir_dynamics()


def build_qrc_independent_set_circuit(
    graph: IndependentSetGraph,
    gamma: float,
    seed: int,
    penalty: float = 2.0,
) -> QuantumCircuit:
    """Apply the QOBLIB Ising cost phase, then the QuantumReservoir ansatz."""
    _, _, local_terms, pair_terms = qubo_and_ising_terms(graph, penalty)
    circuit = QuantumCircuit(graph.num_vertices)
    circuit.h(range(graph.num_vertices))
    for vertex, coefficient in local_terms.items():
        circuit.rz(2.0 * gamma * coefficient, vertex)
    for (source, target), coefficient in pair_terms.items():
        circuit.rzz(2.0 * gamma * coefficient, source, target)
    circuit.compose(_reservoir_ansatz(graph.num_vertices, seed), inplace=True)
    circuit.measure_all()
    return circuit


def repair_independent_set(
    selected: Iterable[int], graph: IndependentSetGraph, rng: np.random.Generator
) -> set[int]:
    """Greedily remove an endpoint from each violated edge until feasible."""
    repaired = set(selected)
    while True:
        violations = [edge for edge in graph.edges if edge[0] in repaired and edge[1] in repaired]
        if not violations:
            return repaired
        source, target = violations[0]
        source_degree, target_degree = graph.degrees[source], graph.degrees[target]
        if source_degree == target_degree:
            repaired.remove(source if rng.integers(2) == 0 else target)
        else:
            repaired.remove(source if source_degree > target_degree else target)


def sample_repaired_independent_sets(
    graph: IndependentSetGraph,
    shots: int,
    seed: int,
    gamma: float = 0.7,
    penalty: float = 2.0,
) -> list[set[int]]:
    """Sample the reservoir circuit and repair every measured bitstring."""
    if shots < 1:
        raise ValueError("shots must be positive")
    circuit = build_qrc_independent_set_circuit(graph, gamma, seed, penalty)
    simulator = AerSimulator(method="matrix_product_state", seed_simulator=seed)
    counts = simulator.run(circuit, shots=shots).result().get_counts()
    rng = np.random.default_rng(seed)
    repaired: list[set[int]] = []
    for bitstring, count in counts.items():
        # Qiskit count strings are most-significant-qubit first.
        selected = {index for index, bit in enumerate(reversed(bitstring.replace(" ", ""))) if bit == "1"}
        repaired.extend(repair_independent_set(selected, graph, rng) for _ in range(count))
    return repaired
