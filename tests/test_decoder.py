"""Tests for assignment decoding, route ordering, and capacity repair."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.quantum_reservoir_vrp import ReservoirVRPSolver  # noqa: E402


def _solver() -> ReservoirVRPSolver:
    return ReservoirVRPSolver(n_reservoir_qubits=6, trained_params=np.zeros(12))


def _customers(routes):
    return [node for route in routes for node in route if node != 0]


def test_decode_assignment_visits_every_customer_once() -> None:
    solver = _solver()
    rng = np.random.default_rng(1)
    coordinates = rng.random((7, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    assignment = np.array([0, 1, 2, 0, 3, 1])
    routes = solver._decode_assignment(  # pylint: disable=protected-access
        assignment, 4, distances, ordering="nn2opt"
    )
    customers = _customers(routes)
    assert sorted(customers) == list(range(1, 7))
    assert all(route[0] == 0 and route[-1] == 0 for route in routes)


def test_decode_routes_respects_active_locations() -> None:
    solver = _solver()
    distances = np.ones((8, 8))
    np.fill_diagonal(distances, 0.0)
    encoding = np.zeros(4 * 8)
    encoding.reshape(4, 8)[:, 1:5] = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    routes = solver._decode_routes(  # pylint: disable=protected-access
        encoding,
        n_vehicles=4,
        n_locations=8,
        active_locations=5,
        distance_matrix=distances,
        ordering="nn2opt",
    )
    assert sorted(_customers(routes)) == [1, 2, 3, 4]
    assert max(_customers(routes)) < 5


def test_nn2opt_is_no_worse_than_index_on_a_crossing_route() -> None:
    solver = _solver()
    coordinates = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]]
    )
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    assignment = np.zeros(4, dtype=int)
    index_routes = solver._decode_assignment(  # pylint: disable=protected-access
        assignment, 4, distances, ordering="index"
    )
    ordered_routes = solver._decode_assignment(  # pylint: disable=protected-access
        assignment, 4, distances, ordering="nn2opt"
    )
    index_cost = sum(
        distances[a, b] for route in index_routes for a, b in zip(route, route[1:])
    )
    ordered_cost = sum(
        distances[a, b] for route in ordered_routes for a, b in zip(route, route[1:])
    )
    assert ordered_cost <= index_cost + 1e-12


def test_capacity_repair_keeps_all_customers_and_capacity() -> None:
    solver = _solver()
    rng = np.random.default_rng(2)
    coordinates = rng.random((9, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    demands = np.array([0, 5, 5, 5, 5, 5, 5, 5, 5])
    capacity = 12
    assignment = np.zeros(8, dtype=int)
    routes = solver._decode_assignment(  # pylint: disable=protected-access
        assignment,
        4,
        distances,
        ordering="nn2opt",
        demands=demands,
        capacity=capacity,
    )
    loads = [sum(demands[node] for node in route if node != 0) for route in routes]
    assert all(load <= capacity for load in loads)
    assert sorted(_customers(routes)) == list(range(1, 9))


def test_capacity_repair_is_deterministic() -> None:
    solver = _solver()
    rng = np.random.default_rng(4)
    coordinates = rng.random((8, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    demands = np.array([0, 4, 7, 2, 9, 1, 5, 3])
    assignment = np.array([0, 0, 1, 1, 2, 2, 3])
    first = solver._decode_assignment(  # pylint: disable=protected-access
        assignment, 4, distances, ordering="nn2opt", demands=demands, capacity=13
    )
    second = solver._decode_assignment(  # pylint: disable=protected-access
        assignment, 4, distances, ordering="nn2opt", demands=demands, capacity=13
    )
    assert first == second
