"""Tests for label-free pairwise targets, features, and clustering decode."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.readout import (  # noqa: E402
    cluster_similarity,
    customer_pairs,
    distance_node_features,
    pair_feature_matrix,
    pair_targets_from_routes,
    reservoir_pair_components,
)


def _rigid_transform(coordinates: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return coordinates @ rotation.T


def test_pair_targets_are_vehicle_permutation_invariant() -> None:
    routes = [[0, 1, 2, 0], [0, 3, 0], [0, 4, 0], [0, 0]]
    first = pair_targets_from_routes(routes, 5)
    permutation = [2, 0, 3, 1]
    second = pair_targets_from_routes(
        [routes[index] for index in permutation], 5
    )
    assert np.array_equal(first, second)
    assert first.shape == (len(customer_pairs(5)),)


def test_pair_targets_are_rigid_motion_invariant() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    routes = [[0, 1, 2, 0], [0, 3, 0], [0, 4, 0], [0, 0]]
    mirrored = coordinates * np.array([-1.0, 1.0])
    rotated = _rigid_transform(coordinates, 0.7)

    def distances(points: np.ndarray) -> np.ndarray:
        deltas = points[:, None, :] - points[None, :, :]
        matrix = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    original = pair_targets_from_routes(routes, 5)
    for transformed in (mirrored, rotated):
        assert np.allclose(distances(coordinates), distances(transformed))
        assert np.array_equal(
            original, pair_targets_from_routes(routes, 5)
        )
        assert np.allclose(
            distance_node_features(distances(coordinates)),
            distance_node_features(distances(transformed)),
        )


def test_cluster_similarity_assigns_each_customer_once() -> None:
    rng = np.random.default_rng(17)
    for n_customers, n_clusters in [(1, 1), (7, 4), (12, 3), (5, 4)]:
        matrix = rng.normal(size=(n_customers, n_customers))
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
        labels = cluster_similarity(matrix, n_clusters)
        assert labels.shape == (n_customers,)
        assert set(labels).issubset(set(range(min(n_clusters, n_customers))))
        assert np.all(np.bincount(labels, minlength=min(n_clusters, n_customers)) > 0)


def test_esn_layout_matches_reservoir_pair_components() -> None:
    from scripts.benchmark_fair import EchoStateNetwork, pair_feature_rows

    n = 7
    rng = np.random.default_rng(23)
    distances = rng.random((n, n))
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    esn = EchoStateNetwork(n, n * n, seed=3)
    features = esn.features(distances)
    assert len(features) == 3 * n + n * (n - 1) // 2
    observables, zz_matrix = reservoir_pair_components(features, n)
    assert observables.shape == (n, 3)
    assert np.allclose(observables[:, 0], esn.last_state)
    assert np.allclose(observables[:, 1], esn.last_state**2)
    assert np.allclose(observables[:, 2], np.tanh(2.0 * esn.last_state))
    expected_zz = np.outer(esn.last_state, esn.last_state)
    for left in range(n):
        for right in range(left + 1, n):
            assert np.isclose(zz_matrix[left, right], expected_zz[left, right])
    rows = pair_feature_rows("classical_esn", features, n)
    assert rows.shape == ((n - 1) * (n - 2) // 2, 7)
    assert np.allclose(rows[:, -1], [
        zz_matrix[left, right] for left, right in customer_pairs(n)
    ])
