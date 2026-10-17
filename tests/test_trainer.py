"""Tests for seeded reservoir-parameter training and deployment samplers."""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.reservoir_trainer import (  # noqa: E402
    build_parameterized_reservoir,
    build_shallow_local_reservoir,
    train_reservoir_offline,
)


def test_trainer_is_seeded_and_seed_changes_result() -> None:
    first = train_reservoir_offline(
        3, layers=1, maxiter=2, seed=11, circuit_builder=build_parameterized_reservoir
    )
    second = train_reservoir_offline(
        3, layers=1, maxiter=2, seed=11, circuit_builder=build_parameterized_reservoir
    )
    other = train_reservoir_offline(
        3, layers=1, maxiter=2, seed=12, circuit_builder=build_parameterized_reservoir
    )
    assert np.array_equal(first, second)
    assert not np.allclose(first, other)


def test_input_sampler_is_called_with_seeded_rng() -> None:
    calls = {"count": 0, "seed": None}

    def sampler(rng: np.random.Generator):
        calls["count"] += 1
        calls["seed"] = int(rng.integers(0, 1_000_000))
        circuits = []
        for _ in range(2):
            circuit = QuantumCircuit(3)
            for qubit in range(3):
                circuit.ry(float(rng.uniform(0.0, 2.0 * np.pi)), qubit)
            circuits.append(circuit)
        return circuits

    params = train_reservoir_offline(
        3,
        layers=1,
        maxiter=1,
        seed=7,
        circuit_builder=build_shallow_local_reservoir,
        input_sampler=sampler,
    )
    assert calls["count"] == 1
    assert calls["seed"] is not None
    assert params.shape == (3,)


def test_trainer_rejects_empty_sampler() -> None:
    try:
        train_reservoir_offline(
            3,
            layers=1,
            maxiter=1,
            seed=0,
            circuit_builder=build_shallow_local_reservoir,
            input_sampler=lambda rng: [],
        )
    except ValueError as error:
        assert "no circuits" in str(error)
    else:
        raise AssertionError("empty sampler should raise ValueError")


def test_deployment_sampler_with_offdiagonal_observables_trains() -> None:
    from scripts.benchmark_fair import deployment_input_sampler

    params = train_reservoir_offline(
        3,
        layers=1,
        maxiter=1,
        seed=3,
        circuit_builder=build_parameterized_reservoir,
        input_sampler=deployment_input_sampler(3, 1.0, False, 3),
        observables=("X", "Y", "Z"),
    )
    assert params.shape == (3,)
