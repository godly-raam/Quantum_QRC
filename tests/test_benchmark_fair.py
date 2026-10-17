"""Tests for the fair benchmark's deterministic plumbing."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark_fair import (  # noqa: E402
    bootstrap_ci,
    generate_instance,
    instance_seed,
    make_split,
    route_cost,
    sweep_assignment,
)


def test_instances_are_seeded_and_euclidean() -> None:
    first = generate_instance(1234, 8)
    second = generate_instance(1234, 8)
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.allclose(first.distances, first.distances.T)
    assert np.allclose(np.diag(first.distances), 0.0)
    assert first.distances.shape == (8, 8)


def test_split_is_by_instance_seed() -> None:
    train, test = make_split(outer_seed=2, n_locations=8, total=250, train_count=200)
    assert len(train) == 200
    assert len(test) == 50
    assert train[0].seed == instance_seed(2, 8, 0)
    assert test[0].seed == instance_seed(2, 8, 200)
    assert not ({instance.seed for instance in train} & {instance.seed for instance in test})


def test_route_cost_uses_full_closed_routes() -> None:
    distances = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    )
    assert route_cost([[0, 1, 0], [0, 2, 0]], distances) == 1.0 + 1.0 + 2.0 + 2.0


def test_sweep_assignment_is_a_partition() -> None:
    instance = generate_instance(7, 12)
    labels = sweep_assignment(instance, 4)
    assert len(labels) == 11
    assert set(labels).issubset(set(range(4)))
    assert all(np.count_nonzero(labels == vehicle) > 0 for vehicle in range(4))


def test_bootstrap_ci_recovers_mean() -> None:
    values = np.full(100, 2.5)
    mean, low, high = bootstrap_ci(values, resamples=200, seed=0)
    assert mean == 2.5
    assert low == 2.5
    assert high == 2.5
