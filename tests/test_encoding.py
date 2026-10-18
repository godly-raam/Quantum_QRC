"""Tests for feature-map normalization, angle scaling, and scalarization."""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.quantum_reservoir_vrp import QuantumReservoir, ReservoirVRPSolver  # noqa: E402
from multi_objective_encoder import (  # noqa: E402
    build_qrc_feature_map,
    normalize_cost_matrix,
)


def _expectation(state: Statevector, operator: str) -> float:
    return float(np.real(state.expectation_value(SparsePauliOp(operator))))


def test_closed_form_input_state_at_six_qubits() -> None:
    n_qubits = 6
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(n_qubits, n_qubits))
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, rng.normal(size=n_qubits))
    angle_scale = 0.7

    circuit = build_qrc_feature_map(matrix, n_qubits, angle_scale=angle_scale)
    state = Statevector.from_instruction(circuit)
    normalized = normalize_cost_matrix(matrix)

    for i in range(n_qubits):
        expected_x = np.cos(angle_scale * normalized[i, i]) * np.prod(
            [
                np.cos(angle_scale * normalized[i, j])
                for j in range(n_qubits)
                if j != i
            ]
        )
        expected_y = np.sin(angle_scale * normalized[i, i]) * np.prod(
            [
                np.cos(angle_scale * normalized[i, j])
                for j in range(n_qubits)
                if j != i
            ]
        )
        x_string = "I" * (n_qubits - 1 - i) + "X" + "I" * i
        y_string = "I" * (n_qubits - 1 - i) + "Y" + "I" * i
        z_string = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        assert abs(_expectation(state, x_string) - expected_x) < 1e-9
        assert abs(_expectation(state, y_string) - expected_y) < 1e-9
        assert abs(_expectation(state, z_string)) < 1e-9


def test_encoder_normalizes_before_angles() -> None:
    matrix = np.array(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 1.0], [4.0, 1.0, 0.0]]
    )
    first = Statevector.from_instruction(build_qrc_feature_map(matrix, 3)).data
    second = Statevector.from_instruction(
        build_qrc_feature_map(matrix * 37.0, 3)
    ).data
    assert np.allclose(first, second, atol=1e-12)


def test_angle_scale_changes_local_features() -> None:
    matrix = np.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, 0.25], [0.5, 0.25, 0.0]]
    )
    full = Statevector.from_instruction(
        build_qrc_feature_map(matrix, 3, angle_scale=1.0)
    )
    scaled = Statevector.from_instruction(
        build_qrc_feature_map(matrix, 3, angle_scale=0.25)
    )
    assert abs(_expectation(full, "IIX") - _expectation(scaled, "IIX")) > 1e-3


def test_depot_diagonal_populates_y_features() -> None:
    distances = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]]
    )
    plain = Statevector.from_instruction(
        build_qrc_feature_map(distances, 4, use_depot_diagonal=False)
    )
    diagonal = Statevector.from_instruction(
        build_qrc_feature_map(distances, 4, use_depot_diagonal=True)
    )
    assert abs(_expectation(plain, "IIYI")) < 1e-12
    assert abs(_expectation(diagonal, "IIYI")) > 1e-3


def test_multi_objective_scalarizes_once(monkeypatch) -> None:
    import multi_objective_encoder as module

    calls = {"count": 0}
    original = module.generate_scalarized_cost_matrix

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "generate_scalarized_cost_matrix", counting)
    distances = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]]
    )
    solver = ReservoirVRPSolver(
        n_reservoir_qubits=4,
        trained_params=np.zeros(8),
        ordering="index",
    )
    solver.solve_multi_objective(distances, distances, iterations=2, seed=0)
    assert calls["count"] == 2


def test_reservoir_seeded_architecture_is_deterministic() -> None:
    distances = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]]
    )
    first = QuantumReservoir(4, random_seed=5, trained_params=np.zeros(8))
    second = QuantumReservoir(4, random_seed=5, trained_params=np.zeros(8))
    circuit_a = first.build_architecture_from_scalar(distances, 2, lcu_theta=0.3)
    circuit_b = second.build_architecture_from_scalar(distances, 2, lcu_theta=0.3)
    assert np.allclose(
        Statevector.from_instruction(circuit_a).data,
        Statevector.from_instruction(circuit_b).data,
    )


def test_fast_observable_measurement_matches_qiskit() -> None:
    from scripts.benchmark_fair import measure_observables_fast

    n_qubits = 5
    reservoir = QuantumReservoir(
        n_qubits, random_seed=1, trained_params=np.zeros(2 * n_qubits)
    )
    rng = np.random.default_rng(9)
    state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    state /= np.linalg.norm(state)
    fast = measure_observables_fast(state, n_qubits)
    qiskit = reservoir.measure_observables(state)
    assert np.allclose(fast[: len(qiskit)], qiskit, atol=1e-12)
    assert len(fast) == 3 * n_qubits + n_qubits * (n_qubits - 1) // 2


def test_quantum_features_are_seeded() -> None:
    from scripts.benchmark_fair import quantum_features

    distances = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]]
    )
    first = QuantumReservoir(4, random_seed=3, trained_params=np.zeros(8))
    second = QuantumReservoir(4, random_seed=3, trained_params=np.zeros(8))
    assert np.allclose(
        quantum_features(first, distances, 2, lcu_theta=0.2),
        quantum_features(second, distances, 2, lcu_theta=0.2),
    )
