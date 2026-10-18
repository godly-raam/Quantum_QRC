"""Tests for fast all-pairs observables and the resumable feature cache."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.fast_sim import (  # noqa: E402
    FeatureCache,
    measure_features,
    zz_pair_index,
    zz_pairs,
)
from modules.quantum_reservoir_vrp import QuantumReservoir  # noqa: E402


def test_zz_pairs_put_adjacent_first() -> None:
    pairs = zz_pairs(5)
    assert pairs[:3] == [(0, 1), (1, 2), (2, 3)]
    assert len(pairs) == 10
    assert len(set(pairs)) == 10


def test_fast_features_match_qiskit_for_legacy_and_pairs() -> None:
    from qiskit.quantum_info import SparsePauliOp, Statevector

    n_qubits = 5
    reservoir = QuantumReservoir(
        n_qubits, random_seed=1, trained_params=np.zeros(2 * n_qubits)
    )
    rng = np.random.default_rng(9)
    state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    state /= np.linalg.norm(state)

    fast = measure_features(state, n_qubits)
    legacy = reservoir.measure_observables(state)
    assert np.allclose(fast[: len(legacy)], legacy, atol=1e-12)

    statevector = Statevector(state)
    index = zz_pair_index(n_qubits)
    for left, right in [(0, 4), (2, 4), (3, 4)]:
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - left] = "Z"
        pauli[n_qubits - 1 - right] = "Z"
        expected = np.real(
            statevector.expectation_value(SparsePauliOp("".join(pauli)))
        )
        assert np.isclose(fast[index[(left, right)]], expected, atol=1e-12)


def test_feature_cache_round_trip_and_resume(tmp_path) -> None:
    cache = FeatureCache(str(tmp_path))
    values = np.array([1.0, 2.0, 3.0])
    cache.put(4, 123, 0.25, "theta", values)
    cache.flush()

    reopened = FeatureCache(str(tmp_path))
    loaded = reopened.get(4, 123, 0.25, "theta")
    assert loaded is not None
    assert np.array_equal(loaded, values)

    calls = {"count": 0}

    def compute() -> np.ndarray:
        calls["count"] += 1
        return np.array([9.0, 9.0, 9.0])

    cached = reopened.get_or_compute(4, 123, 0.25, "theta", compute)
    assert np.array_equal(cached, values)
    assert calls["count"] == 0

    missing = reopened.get_or_compute(4, 124, 0.25, "theta", compute)
    assert calls["count"] == 1
    assert np.array_equal(missing, np.array([9.0, 9.0, 9.0]))
