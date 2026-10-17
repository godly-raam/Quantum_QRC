"""Tests for canonicalisation and the trained ridge readout."""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.readout import (  # noqa: E402
    RidgeReadout,
    canonicalize_routes,
    node_features,
    readout_assignment,
    routes_to_assignment,
)


def test_canonicalization_is_permutation_invariant() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    routes = [[0, 1, 2, 0], [0, 3, 0], [0, 4, 0], [0, 0]]
    first = canonicalize_routes(routes, coordinates, 4)
    permutation = [2, 0, 3, 1]
    second = canonicalize_routes([routes[index] for index in permutation], coordinates, 4)
    assert first == second
    assert routes_to_assignment(first, 5, 4).tolist() == routes_to_assignment(
        second, 5, 4
    ).tolist()


def test_canonicalization_pads_empty_routes() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    routes = [[0, 1, 0], [0, 0], [0, 2, 0]]
    canonical = canonicalize_routes(routes, coordinates, 4)
    assert len(canonical) == 4
    assert canonical[0] == [0, 1, 0]
    assert canonical[1] == [0, 2, 0]
    assert canonical[2:] == [[0, 0], [0, 0]]


def test_node_features_have_shared_shape() -> None:
    features = np.arange(3 * 6 + 3, dtype=float)
    rows = node_features(features, 6)
    assert rows.shape == (5, 5)
    assert np.array_equal(rows[0, :3], features[3:6])


def test_readout_reproduces_training_fit() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(80, 7))
    true_weights = rng.normal(size=(7, 4))
    targets = features @ true_weights
    readout = RidgeReadout(alphas=(1e-12,), cv_folds=4, seed=0).fit(features, targets)
    predictions = readout.predict(features)
    assert np.mean((predictions - targets) ** 2) < 1e-8


def test_readout_is_seeded() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(60, 5))
    targets = rng.normal(size=(60, 3))
    first = RidgeReadout(alphas=(1e-6, 1.0), seed=11).fit(features, targets)
    second = RidgeReadout(alphas=(1e-6, 1.0), seed=11).fit(features, targets)
    assert np.array_equal(first.weights, second.weights)
    assert first.best_alpha == second.best_alpha


def test_readout_save_load_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(40, 6))
    targets = rng.normal(size=(40, 2))
    readout = RidgeReadout(alphas=(1e-6,), seed=0).fit(features, targets)
    path = os.path.join(tmp_path, "readout.npz")
    readout.save(path)
    loaded = RidgeReadout.load(path)
    assert np.allclose(loaded.predict(features), readout.predict(features))


@pytest.mark.parametrize("mode", ["flat", "node"])
def test_readout_assignment_shapes(mode: str) -> None:
    rng = np.random.default_rng(6)
    n_locations, n_vehicles = 6, 3
    if mode == "flat":
        features = rng.normal(size=(100, 21))
        targets = np.vstack(
            [rng.integers(0, n_vehicles, size=n_locations - 1) for _ in range(100)]
        )
        targets = np.vstack(
            [
                np.eye(n_vehicles)[row].reshape(-1)
                for row in targets
            ]
        )
        readout = RidgeReadout(mode="flat", alphas=(1e-6,), seed=1).fit(
            features, targets
        )
        assignment = readout_assignment(
            readout, features[0], n_locations, n_vehicles
        )
    else:
        feature_rows = rng.normal(size=(500, 5))
        targets = np.vstack(
            [np.eye(n_vehicles)[rng.integers(0, n_vehicles, size=500)]]
        )
        readout = RidgeReadout(mode="node", alphas=(1e-6,), seed=1).fit(
            feature_rows, targets
        )
        vector = rng.normal(size=3 * n_locations + 3)
        assignment = readout_assignment(readout, vector, n_locations, n_vehicles)
    assert assignment.shape == (n_locations - 1,)
    assert set(assignment).issubset(set(range(n_vehicles)))
