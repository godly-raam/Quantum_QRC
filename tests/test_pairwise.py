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
    pair_targets_from_routes,
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
