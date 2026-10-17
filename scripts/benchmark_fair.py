"""Fair quantum-versus-classical VRP benchmark.

The benchmark uses identical route decoders and, for all feature-based methods,
the same trained ridge readout class.  It reports route cost, optimality gap
against an OR-Tools reference, wall-clock time, and paired bootstrap confidence
intervals over test instances and outer seeds.

Usage
-----
python scripts/benchmark_fair.py --mode benchmark
python scripts/benchmark_fair.py --mode encoding-sweep

The quantum runs always use ``n_locations == n_qubits`` and at most 20 qubits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from qiskit.quantum_info import Statevector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.quantum_reservoir_vrp import (  # noqa: E402
    QuantumReservoir,
    ReservoirVRPSolver,
)
from modules.reservoir_trainer import (  # noqa: E402
    build_parameterized_reservoir,
    build_shallow_local_reservoir,
    train_reservoir_offline,
)

METHODS = (
    "random_assignment",
    "sweep_assignment",
    "classical_esn",
    "quantum_ring_trained",
    "quantum_shallow_trained",
    "quantum_ring_random",
    "ortools_0.1s",
)
FEATURE_METHODS = (
    "classical_esn",
    "quantum_ring_trained",
    "quantum_shallow_trained",
    "quantum_ring_random",
)


# ---------------------------------------------------------------------------
# Instances and reference solutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """A random Euclidean, capacitated VRP instance."""

    seed: int
    n_locations: int
    coordinates: np.ndarray
    distances: np.ndarray
    demands: np.ndarray
    capacity: int


def generate_instance(
    seed: int,
    n_locations: int,
    n_vehicles: int = 4,
    capacity_slack: float = 1.25,
) -> Instance:
    """Generate a reproducible capacitated Euclidean instance in the unit square.

    Without a capacity constraint an uncapacitated VRP is a TSP: the optimal
    solution puts every customer on one vehicle, so one-hot vehicle targets are
    degenerate.  Demands and a capacity that requires roughly all vehicles make
    the assignment learnable.
    """
    rng = np.random.default_rng(seed)
    coordinates = rng.random((n_locations, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, 0.0)

    demands = rng.integers(1, 10, size=n_locations).astype(int)
    demands[0] = 0
    total_demand = int(demands.sum())
    capacity = max(
        int(np.ceil(capacity_slack * total_demand / n_vehicles)),
        int(demands.max()),
    )
    return Instance(seed, n_locations, coordinates, distances, demands, capacity)


def instance_seed(outer_seed: int, n_locations: int, index: int) -> int:
    """Deterministic instance seed for an outer seed, size, and index."""
    return int(
        10_000_000
        + outer_seed * 1_000_000
        + n_locations * 10_000
        + index
    )


def make_split(
    outer_seed: int,
    n_locations: int,
    total: int,
    train_count: int,
) -> tuple[list[Instance], list[Instance]]:
    """Build train/test instances; the split is by instance seed/index."""
    instances = [
        generate_instance(
            instance_seed(outer_seed, n_locations, index),
            n_locations,
            n_vehicles=4,
        )
        for index in range(total)
    ]
    return instances[:train_count], instances[train_count:]


def route_cost(routes: Sequence[Sequence[int]], distances: np.ndarray) -> float:
    """Total route cost for a list of routes."""
    return float(
        sum(
            distances[start, end]
            for route in routes
            for start, end in pairwise(route)
        )
    )


def solve_ortools(
    distances: np.ndarray,
    n_vehicles: int,
    time_limit_seconds: float,
    seed: int = 0,
    demands: Optional[np.ndarray] = None,
    capacity: Optional[int] = None,
) -> tuple[list[list[int]], float, float]:
    """Solve a (possibly capacitated) VRP with OR-Tools."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n_locations = distances.shape[0]
    manager = pywrapcp.RoutingIndexManager(n_locations, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    scale = 1_000_000

    def distance_callback(from_index: int, to_index: int) -> int:
        start = manager.IndexToNode(from_index)
        end = manager.IndexToNode(to_index)
        return int(round(float(distances[start, end]) * scale))

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    if demands is not None and capacity is not None:
        demands_array = np.asarray(demands, dtype=int)

        def demand_callback(from_index: int) -> int:
            node = manager.IndexToNode(from_index)
            return int(demands_array[node])

        demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_index,
            0,
            [int(capacity)] * n_vehicles,
            True,
            "Capacity",
        )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(
        max(1, int(round(time_limit_seconds * 1000)))
    )

    started = time.perf_counter()
    solution = routing.SolveWithParameters(parameters)
    elapsed = time.perf_counter() - started

    if solution is None:
        routes = fallback_capacity_routes(distances, n_vehicles, demands, capacity)
        return routes, route_cost(routes, distances), elapsed

    routes = []
    for vehicle in range(n_vehicles):
        index = routing.Start(vehicle)
        route = [0]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes, route_cost(routes, distances), elapsed


def fallback_capacity_routes(
    distances: np.ndarray,
    n_vehicles: int,
    demands: Optional[np.ndarray],
    capacity: Optional[int],
) -> list[list[int]]:
    """Greedy feasible fallback used only if OR-Tools finds no solution."""
    n_locations = distances.shape[0]
    customers = list(range(1, n_locations))
    if demands is None or capacity is None:
        return [[0] + customers + [0]]
    loads = [0] * n_vehicles
    routes = [[0] for _ in range(n_vehicles)]
    for customer in sorted(customers, key=lambda node: -int(demands[node])):
        feasible = [
            vehicle
            for vehicle in range(n_vehicles)
            if loads[vehicle] + int(demands[customer]) <= capacity
        ]
        vehicle = min(
            feasible or range(n_vehicles),
            key=lambda candidate: loads[candidate],
        )
        routes[vehicle].append(customer)
        loads[vehicle] += int(demands[customer])
    return [route + [0] for route in routes]


def canonical_assignments(
    routes_by_instance: Sequence[Sequence[Sequence[int]]],
    instances: Sequence[Instance],
    n_vehicles: int,
) -> list[np.ndarray]:
    """Canonicalise vehicle labels then build one-hot customer assignments."""
    from modules.readout import canonicalize_routes, routes_to_assignment

    return [
        routes_to_assignment(
            canonicalize_routes(routes, instance.coordinates, n_vehicles),
            instance.n_locations,
            n_vehicles,
        )
        for routes, instance in zip(routes_by_instance, instances)
    ]


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def random_assignment(
    instance: Instance, n_vehicles: int, rng: np.random.Generator
) -> np.ndarray:
    """Random vehicle label per customer."""
    return rng.integers(0, n_vehicles, size=instance.n_locations - 1)


def sweep_assignment(instance: Instance, n_vehicles: int) -> np.ndarray:
    """Angular sweep assignment, balanced by customer count."""
    coordinates = instance.coordinates
    delta = coordinates[1:] - coordinates[0]
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    order = np.argsort(angles, kind="stable")
    groups = np.array_split(order, n_vehicles)
    labels = np.zeros(instance.n_locations - 1, dtype=int)
    for vehicle, group in enumerate(groups):
        labels[group] = vehicle
    return labels


# ---------------------------------------------------------------------------
# Classical reservoir
# ---------------------------------------------------------------------------


class EchoStateNetwork:
    """Fixed classical ESN with a 3N feature vector (matched to the old ablation)."""

    def __init__(
        self,
        size: int,
        input_dim: int,
        seed: int,
        spectral_radius: float = 0.9,
        input_scale: float = 0.2,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.input_dim = input_dim
        self.input_weights = rng.uniform(
            -input_scale, input_scale, size=(size, input_dim)
        )
        recurrent = rng.uniform(-1.0, 1.0, size=(size, size))
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self.recurrent_weights = recurrent * (spectral_radius / max(radius, 1e-12))

    def features(self, distances: np.ndarray) -> np.ndarray:
        """Compute a fresh ESN feature vector for one instance."""
        scale = float(np.max(np.abs(distances)))
        normalised = distances / scale if scale > 0.0 else distances
        state = np.zeros(self.size, dtype=float)
        state = np.tanh(
            self.input_weights @ normalised.ravel() + self.recurrent_weights @ state
        )
        return np.concatenate((state, state**2, np.tanh(2.0 * state)))


# ---------------------------------------------------------------------------
# Quantum features
# ---------------------------------------------------------------------------


def measure_observables_fast(statevector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Measure X/Y/Z on every qubit plus 3 adjacent ZZ terms, without Qiskit."""
    state = np.asarray(statevector, dtype=complex)
    indices = np.arange(state.size)
    probabilities = np.abs(state) ** 2
    features: list[float] = []

    for qubit in range(n_qubits):
        flipped = indices ^ (1 << qubit)
        sign = 1.0 - 2.0 * ((indices >> qubit) & 1)
        x_value = np.vdot(state, state[flipped])
        y_value = -1j * np.sum(sign * np.conj(state) * state[flipped])
        z_value = np.sum(probabilities * sign)
        features.extend([float(x_value.real), float(y_value.real), float(z_value.real)])

    for qubit in range(min(3, n_qubits - 1)):
        left = (indices >> qubit) & 1
        right = (indices >> (qubit + 1)) & 1
        same = 1.0 - 2.0 * np.abs(left - right)
        features.append(float(np.sum(probabilities * same)))

    return np.asarray(features, dtype=float)


def quantum_features(
    reservoir: QuantumReservoir,
    distances: np.ndarray,
    n_vehicles: int,
    lcu_theta: float,
) -> np.ndarray:
    """Run the feature map, LCU layer, and reservoir, then measure features."""
    circuit = reservoir.build_architecture_from_scalar(
        distances, n_vehicles, lcu_theta=lcu_theta
    )
    state = Statevector.from_instruction(circuit).data
    reservoir.current_state = state / np.linalg.norm(state)
    return measure_observables_fast(reservoir.current_state, reservoir.n_qubits)


def deployment_input_sampler(
    n_qubits: int,
    angle_scale: float,
    use_depot_diagonal: bool,
    num_samples: int,
) -> Callable[[np.random.Generator], list]:
    """Return feature-map circuits from the real encode path for training."""
    from multi_objective_encoder import build_qrc_feature_map

    def sampler(rng: np.random.Generator) -> list:
        circuits = []
        for _ in range(num_samples):
            coordinates = rng.random((n_qubits, 2))
            deltas = coordinates[:, None, :] - coordinates[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            np.fill_diagonal(distances, 0.0)
            circuits.append(
                build_qrc_feature_map(
                    distances,
                    n_qubits,
                    angle_scale=angle_scale,
                    use_depot_diagonal=use_depot_diagonal,
                )
            )
        return circuits

    return sampler


def train_quantum_parameters(
    n_qubits: int,
    builder: Callable[[int, int], object],
    seed: int,
    angle_scale: float,
    use_depot_diagonal: bool,
    num_samples: int,
    maxiter: int,
) -> np.ndarray:
    """Train reservoir angles on deployment-distribution feature-map states."""
    return train_reservoir_offline(
        n_qubits,
        layers=1 if builder is build_shallow_local_reservoir else 2,
        maxiter=maxiter,
        circuit_builder=builder,  # type: ignore[arg-type]
        seed=seed,
        input_sampler=deployment_input_sampler(
            n_qubits, angle_scale, use_depot_diagonal, num_samples
        ),
        observables=("X", "Y", "Z"),
    )


# ---------------------------------------------------------------------------
# Readout data plumbing
# ---------------------------------------------------------------------------


def flat_feature_rows(features: Sequence[np.ndarray]) -> np.ndarray:
    """Stack one reservoir feature vector per instance."""
    return np.vstack([np.asarray(row, dtype=float) for row in features])


def node_feature_rows(
    features: Sequence[np.ndarray], n_qubits: int
) -> np.ndarray:
    """Stack per-customer feature rows for shared-weight node readouts."""
    from modules.readout import node_features

    return np.vstack([node_features(row, n_qubits) for row in features])


def fit_readout(
    method: str,
    train_features: Sequence[np.ndarray],
    train_assignment: Sequence[np.ndarray],
    n_locations: int,
    n_vehicles: int,
    readout_mode: str,
    seed: int,
    alphas: Sequence[float],
):
    """Fit the shared ridge readout for one method/size/seed."""
    from modules.readout import RidgeReadout

    if readout_mode == "node":
        features = node_feature_rows(train_features, n_locations)
        targets = np.vstack(
            [assignment for assignment in train_assignment]
        )
    else:
        features = flat_feature_rows(train_features)
        targets = np.vstack(
            [assignment.reshape(-1) for assignment in train_assignment]
        )
    readout = RidgeReadout(
        mode=readout_mode,
        alphas=tuple(alphas),
        cv_folds=5,
        seed=seed,
    )
    return readout.fit(features, targets)


def predict_assignment(
    readout,
    features: np.ndarray,
    n_locations: int,
    n_vehicles: int,
    readout_mode: str,
) -> np.ndarray:
    """Turn a readout prediction into one vehicle label per customer."""
    from modules.readout import assignments_from_logits, node_features

    if readout_mode == "node":
        rows = node_features(features, n_locations)
        logits = readout.predict(rows)
        return np.argmax(logits, axis=1)
    logits = readout.predict(features.reshape(1, -1))
    return assignments_from_logits(logits.reshape(-1), n_locations, n_vehicles)


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------


def _decode(
    solver: ReservoirVRPSolver,
    assignment: np.ndarray,
    instance: Instance,
    n_vehicles: int,
    ordering: str,
) -> list[list[int]]:
    return solver._decode_assignment(  # pylint: disable=protected-access
        assignment,
        n_vehicles,
        instance.distances,
        ordering=ordering,
        demands=instance.demands,
        capacity=instance.capacity,
    )


def _lcu_theta(n_qubits: int, n_vehicles: int, seed: int) -> float:
    from modules.lcu_constraint_flattener import sample_lcu_branch

    return sample_lcu_branch(
        n_qubits, min(n_vehicles, n_qubits), rng=np.random.default_rng(seed)
    )


def _collect_quantum_features(
    reservoir: QuantumReservoir,
    instances: Sequence[Instance],
    n_vehicles: int,
    outer_seed: int,
    timing: bool,
) -> tuple[list[np.ndarray], list[float]]:
    features = []
    times = []
    for instance in instances:
        started = time.perf_counter()
        theta = _lcu_theta(
            instance.n_locations,
            n_vehicles,
            instance.seed + outer_seed,
        )
        feature_vector = quantum_features(
            reservoir, instance.distances, n_vehicles, theta
        )
        elapsed = time.perf_counter() - started
        features.append(feature_vector)
        times.append(elapsed)
    return features, times


def _collect_esn_features(
    esn: EchoStateNetwork, instances: Sequence[Instance], timing: bool
) -> tuple[list[np.ndarray], list[float]]:
    features = []
    times = []
    for instance in instances:
        started = time.perf_counter()
        feature_vector = esn.features(instance.distances)
        elapsed = time.perf_counter() - started
        features.append(feature_vector)
        times.append(elapsed)
    return features, times


def evaluate_method(
    method: str,
    solver: ReservoirVRPSolver,
    instances: Sequence[Instance],
    train_assignment: Sequence[np.ndarray],
    n_vehicles: int,
    seed: int,
    alphas: Sequence[float],
    readout_mode: str,
    ordering: str,
    train_instances: Optional[Sequence[Instance]] = None,
    legacy_diagnostics: bool = False,
) -> tuple[list[dict], dict]:
    """Train (if needed) and evaluate one method on the test instances."""
    from modules.readout import canonicalize_routes

    test_features: list[np.ndarray]
    test_feature_times: list[float]
    readout = None

    if method in FEATURE_METHODS:
        train_features, _ = _method_features(
            method, solver, train_instances, n_vehicles, seed, timing=False
        )
        readout = fit_readout(
            method,
            train_features,
            train_assignment,
            instances[0].n_locations,
            n_vehicles,
            readout_mode,
            seed,
            alphas,
        )
        test_features, test_feature_times = _method_features(
            method, solver, instances, n_vehicles, seed, timing=True
        )
    else:
        test_features = []
        test_feature_times = [0.0] * len(instances)

    rows = []
    diagnostic = {}
    rng = np.random.default_rng(seed + 991)
    for index, instance in enumerate(instances):
        started = time.perf_counter()
        if method == "random_assignment":
            assignment = random_assignment(instance, n_vehicles, rng)
        elif method == "sweep_assignment":
            assignment = sweep_assignment(instance, n_vehicles)
        elif method in FEATURE_METHODS:
            assignment = predict_assignment(
                readout,
                test_features[index],
                instance.n_locations,
                n_vehicles,
                readout_mode,
            )
        else:
            raise ValueError(f"Unknown method {method}")

        routes = _decode(solver, assignment, instance, n_vehicles, ordering)
        cost = route_cost(routes, instance.distances)
        route_cost_index = cost
        if ordering != "index":
            index_routes = _decode(
                solver, assignment, instance, n_vehicles, "index"
            )
            route_cost_index = route_cost(index_routes, instance.distances)
        elapsed = time.perf_counter() - started + test_feature_times[index]
        rows.append(
            {
                "method": method,
                "cost": cost,
                "cost_index": route_cost_index,
                "time": elapsed,
                "feature_variance": float(np.var(test_features[index]))
                if method in FEATURE_METHODS
                else float("nan"),
                "feature_mean_abs": float(np.mean(np.abs(test_features[index])))
                if method in FEATURE_METHODS
                else float("nan"),
            }
        )

    if legacy_diagnostics and method == "quantum_ring_trained":
        legacy_costs = []
        trained_index_costs = []
        for index, instance in enumerate(instances):
            feature_vector = test_features[index]
            encoding_size = n_vehicles * instance.n_locations
            padded = np.pad(
                feature_vector, (0, max(0, encoding_size - len(feature_vector)))
            )[:encoding_size]
            legacy_routes = solver._decode_routes(  # pylint: disable=protected-access
                padded,
                n_vehicles,
                instance.n_locations,
                active_locations=instance.n_locations,
                distance_matrix=instance.distances,
                ordering="index",
            )
            legacy_costs.append(route_cost(legacy_routes, instance.distances))
            assignment = predict_assignment(
                readout,
                feature_vector,
                instance.n_locations,
                n_vehicles,
                readout_mode,
            )
            index_routes = _decode(
                solver, assignment, instance, n_vehicles, "index"
            )
            trained_index_costs.append(route_cost(index_routes, instance.distances))
        diagnostic = {
            "legacy_argmax_index": float(np.mean(legacy_costs)),
            "trained_readout_index": float(np.mean(trained_index_costs)),
            "trained_readout_nn2opt": float(np.mean([row["cost"] for row in rows])),
        }

    return rows, diagnostic


def _method_features(
    method: str,
    solver: ReservoirVRPSolver,
    instances: Sequence[Instance],
    n_vehicles: int,
    seed: int,
    timing: bool,
) -> tuple[list[np.ndarray], list[float]]:
    """Extract features for a feature-based method."""
    if method == "classical_esn":
        esn = getattr(solver, "_benchmark_esn", None)
        if esn is None:
            esn = EchoStateNetwork(
                solver.reservoir.n_qubits,
                solver.reservoir.n_qubits**2,
                seed=seed,
            )
            solver._benchmark_esn = esn  # type: ignore[attr-defined]
        return _collect_esn_features(esn, instances, timing)
    return _collect_quantum_features(
        solver.reservoir, instances, n_vehicles, seed, timing
    )


# ---------------------------------------------------------------------------
# Bootstrap and reporting
# ---------------------------------------------------------------------------


def bootstrap_ci(
    differences: np.ndarray,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile 95% bootstrap CI of paired differences."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(values), len(values))
        means[index] = values[sample].mean()
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def write_markdown_table(
    path: str,
    summary: Sequence[dict],
    comparisons: Sequence[dict],
    title: str,
) -> None:
    """Write a human-readable markdown report."""
    lines = [f"# {title}", ""]
    lines.append(
        "Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. "
        "Times are per-solve wall clock for the test split, excluding training."
    )
    lines.append("")
    lines.append(
        "| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {n_locations} | {method} | {mean_cost:.4f} | {mean_gap:.2%} | "
            "{median_time_ms:.3f} | {feature_mean_abs:.4f} | {feature_variance:.5f} |".format(
                **row
            )
        )

    lines.extend(["", "## Paired differences (bootstrap 95% CI)", ""])
    lines.append(
        "| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |"
    )
    lines.append("|---:|---|---:|---|---|---:|")
    for row in comparisons:
        lines.append(
            "| {n_locations} | {comparison} | {mean_diff:.4f} | "
            "[{ci_low:.4f}, {ci_high:.4f}] | {excludes_zero} | {n_pairs} |".format(
                **row
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_csv(path: str, rows: Sequence[dict]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full fair benchmark and write CSV/markdown outputs."""
    os.makedirs(args.output_dir, exist_ok=True)
    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    paired_costs: dict[tuple[int, str], list[float]] = {}

    for size in args.sizes:
        if size > 20:
            raise ValueError("Quantum runs are limited to at most 20 qubits")
        if size < 2:
            raise ValueError("n_locations must be at least two")

        for outer_seed in args.outer_seeds:
            train_instances, test_instances = make_split(
                outer_seed,
                size,
                args.per_size,
                args.train_per_size,
            )
            all_instances = list(train_instances) + list(test_instances)
            reference_routes = []
            reference_costs = []
            latency_costs = []
            reference_times = []
            latency_times = []
            for index, instance in enumerate(all_instances):
                routes, cost, elapsed = solve_ortools(
                    instance.distances,
                    args.vehicles,
                    args.ortools_seconds,
                    seed=outer_seed * 1000 + index,
                    demands=instance.demands,
                    capacity=instance.capacity,
                )
                reference_routes.append(routes)
                reference_costs.append(cost)
                reference_times.append(elapsed)

                latency_routes, latency_cost, latency_elapsed = solve_ortools(
                    instance.distances,
                    args.vehicles,
                    args.ortools_latency_seconds,
                    seed=outer_seed * 1000 + index + 7,
                    demands=instance.demands,
                    capacity=instance.capacity,
                )
                latency_costs.append(latency_cost)
                latency_times.append(latency_elapsed)

            assignments = canonical_assignments(
                reference_routes, all_instances, args.vehicles
            )
            train_assignments = assignments[: len(train_instances)]

            trainer_seed = args.trainer_seed_base + outer_seed
            params = {}
            params["quantum_ring_trained"] = train_quantum_parameters(
                size,
                build_parameterized_reservoir,
                trainer_seed,
                args.angle_scale,
                args.use_depot_diagonal,
                args.trainer_samples,
                args.trainer_maxiter,
            )
            params["quantum_shallow_trained"] = train_quantum_parameters(
                size,
                build_shallow_local_reservoir,
                trainer_seed + 1,
                args.angle_scale,
                args.use_depot_diagonal,
                args.trainer_samples,
                args.trainer_maxiter,
            )
            random_rng = np.random.default_rng(
                args.reservoir_seed_base + outer_seed * 100 + size
            )
            ring_layers = 2
            params["quantum_ring_random"] = random_rng.uniform(
                0.0, 2.0 * np.pi, size * ring_layers
            )

            for method in METHODS:
                if method in ("random_assignment", "sweep_assignment"):
                    solver = ReservoirVRPSolver(
                        n_reservoir_qubits=size,
                        trained_params=None,
                        legacy_decoder=False,
                        ordering="nn2opt",
                    )
                elif method == "classical_esn":
                    solver = ReservoirVRPSolver(
                        n_reservoir_qubits=size,
                        trained_params=None,
                        legacy_decoder=False,
                        ordering="nn2opt",
                    )
                elif method == "quantum_ring_trained":
                    solver = ReservoirVRPSolver(
                        n_reservoir_qubits=size,
                        trained_params=params[method],
                        legacy_decoder=False,
                        ordering="nn2opt",
                        angle_scale=args.angle_scale,
                        use_depot_diagonal=args.use_depot_diagonal,
                        random_seed=outer_seed,
                    )
                elif method == "quantum_shallow_trained":
                    solver = ReservoirVRPSolver(
                        n_reservoir_qubits=size,
                        trained_params=params[method],
                        legacy_decoder=False,
                        ordering="nn2opt",
                        angle_scale=args.angle_scale,
                        use_depot_diagonal=args.use_depot_diagonal,
                        random_seed=outer_seed,
                        reservoir_builder=build_shallow_local_reservoir,
                        reservoir_layers=1,
                    )
                elif method == "quantum_ring_random":
                    solver = ReservoirVRPSolver(
                        n_reservoir_qubits=size,
                        trained_params=params[method],
                        legacy_decoder=False,
                        ordering="nn2opt",
                        angle_scale=args.angle_scale,
                        use_depot_diagonal=args.use_depot_diagonal,
                        random_seed=outer_seed,
                    )
                else:
                    continue

                if method == "ortools_0.1s":
                    costs = latency_costs[len(train_instances) :]
                    times = latency_times[len(train_instances) :]
                    for index, (cost, elapsed, instance) in enumerate(
                        zip(costs, times, test_instances)
                    ):
                        run_rows.append(
                            {
                                "n_locations": size,
                                "outer_seed": outer_seed,
                                "instance_seed": instance.seed,
                                "split": "test",
                                "method": method,
                                "cost": cost,
                                "gap": (cost - reference_costs[len(train_instances) + index])
                                / reference_costs[len(train_instances) + index],
                                "time": elapsed,
                                "reference_cost": reference_costs[
                                    len(train_instances) + index
                                ],
                                "cost_index": cost,
                                "feature_mean_abs": float("nan"),
                                "feature_variance": float("nan"),
                            }
                        )
                        paired_costs.setdefault((size, method), []).append(
                            cost - reference_costs[len(train_instances) + index]
                        )
                    continue

                method_rows, diagnostics = evaluate_method(
                    method,
                    solver,
                    test_instances,
                    train_assignments,
                    args.vehicles,
                    seed=outer_seed * 10_000 + size,
                    alphas=args.alphas,
                    readout_mode=args.readout,
                    ordering=args.ordering,
                    train_instances=train_instances,
                    legacy_diagnostics=(method == "quantum_ring_trained"),
                )
                if diagnostics:
                    diagnostic_rows.append(
                        {
                            "n_locations": size,
                            "outer_seed": outer_seed,
                            "method": method,
                            **diagnostics,
                        }
                    )
                for index, (row, instance) in enumerate(
                    zip(method_rows, test_instances)
                ):
                    reference_cost = reference_costs[len(train_instances) + index]
                    run_rows.append(
                        {
                            "n_locations": size,
                            "outer_seed": outer_seed,
                            "instance_seed": instance.seed,
                            "split": "test",
                            "method": method,
                            "cost": row["cost"],
                            "gap": (row["cost"] - reference_cost) / reference_cost,
                            "time": row["time"],
                            "reference_cost": reference_cost,
                            "cost_index": row["cost_index"],
                            "feature_mean_abs": row["feature_mean_abs"],
                            "feature_variance": row["feature_variance"],
                        }
                    )
                    paired_costs.setdefault((size, method), []).append(
                        row["cost"] - reference_cost
                    )

            print(
                f"completed size={size} outer_seed={outer_seed} "
                f"({len(test_instances)} test instances)"
            )

        for method in METHODS:
            method_rows = [
                row
                for row in run_rows
                if row["n_locations"] == size and row["method"] == method
            ]
            if not method_rows:
                continue
            costs = np.array([row["cost"] for row in method_rows])
            gaps = np.array([row["gap"] for row in method_rows])
            times = np.array([row["time"] for row in method_rows])
            summary_rows.append(
                {
                    "n_locations": size,
                    "method": method,
                    "mean_cost": float(np.mean(costs)),
                    "mean_gap": float(np.mean(gaps)),
                    "median_time_ms": float(np.median(times) * 1000.0),
                    "feature_mean_abs": float(
                        np.nanmean([row["feature_mean_abs"] for row in method_rows])
                    ),
                    "feature_variance": float(
                        np.nanmean([row["feature_variance"] for row in method_rows])
                    ),
                }
            )

    pair_definitions = (
        ("quantum_ring_trained", "classical_esn"),
        ("quantum_shallow_trained", "classical_esn"),
        ("quantum_ring_trained", "quantum_ring_random"),
        ("quantum_ring_trained", "random_assignment"),
        ("quantum_ring_trained", "sweep_assignment"),
        ("quantum_ring_trained", "ortools_0.1s"),
        ("classical_esn", "sweep_assignment"),
        ("sweep_assignment", "random_assignment"),
        ("ortools_0.1s", "ortools_1s"),
    )
    for size in args.sizes:
        cost_lookup: dict[str, dict[int, float]] = {}
        for method in METHODS:
            cost_lookup[method] = {
                row["instance_seed"]: row["cost"]
                for row in run_rows
                if row["n_locations"] == size and row["method"] == method
            }
        reference_lookup = {
            row["instance_seed"]: row["reference_cost"]
            for row in run_rows
            if row["n_locations"] == size
        }
        cost_lookup["ortools_1s"] = reference_lookup
        for left, right in pair_definitions:
            common = sorted(set(cost_lookup.get(left, {})) & set(cost_lookup.get(right, {})))
            if not common:
                continue
            differences = np.array(
                [cost_lookup[left][key] - cost_lookup[right][key] for key in common]
            )
            mean_diff, low, high = bootstrap_ci(differences, seed=size)
            comparison_rows.append(
                {
                    "n_locations": size,
                    "comparison": f"{left} - {right}",
                    "mean_diff": mean_diff,
                    "ci_low": low,
                    "ci_high": high,
                    "excludes_zero": "yes" if low > 0.0 or high < 0.0 else "no",
                    "n_pairs": len(common),
                }
            )

    write_csv(os.path.join(args.output_dir, "fair_benchmark_runs.csv"), run_rows)
    write_csv(os.path.join(args.output_dir, "fair_benchmark_summary.csv"), summary_rows)
    write_csv(
        os.path.join(args.output_dir, "fair_benchmark_comparisons.csv"),
        comparison_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "fair_benchmark_diagnostics.csv"),
        diagnostic_rows,
    )
    write_markdown_table(
        os.path.join(args.output_dir, "fair_benchmark.md"),
        summary_rows,
        comparison_rows,
        "Fair quantum-versus-classical VRP benchmark",
    )

    with open(
        os.path.join(args.output_dir, "fair_benchmark_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Encoding sweep
# ---------------------------------------------------------------------------


def run_encoding_sweep(args: argparse.Namespace) -> None:
    """Sweep angle_scale and select by validation route cost."""
    os.makedirs(args.output_dir, exist_ok=True)
    rows: list[dict] = []

    for size in args.sizes:
        outer_seed = args.outer_seeds[0]
        train_instances, test_instances = make_split(
            outer_seed, size, args.per_size, args.train_per_size
        )
        reference_routes = []
        for index, instance in enumerate(train_instances):
            routes, _, _ = solve_ortools(
                instance.distances,
                args.vehicles,
                args.ortools_seconds,
                seed=outer_seed * 1000 + index,
                demands=instance.demands,
                capacity=instance.capacity,
            )
            reference_routes.append(routes)
        train_assignments = canonical_assignments(
            reference_routes, train_instances, args.vehicles
        )

        random_rng = np.random.default_rng(
            args.reservoir_seed_base + outer_seed * 100 + size
        )
        random_theta = random_rng.uniform(0.0, 2.0 * np.pi, size * 2)

        for angle_scale in args.angle_scales:
            solver = ReservoirVRPSolver(
                n_reservoir_qubits=size,
                trained_params=random_theta,
                legacy_decoder=False,
                ordering=args.ordering,
                angle_scale=angle_scale,
                use_depot_diagonal=args.use_depot_diagonal,
                random_seed=outer_seed,
            )
            all_features, _ = _collect_quantum_features(
                solver.reservoir, train_instances, args.vehicles, outer_seed, False
            )
            costs = _cross_validated_route_cost(
                all_features,
                train_assignments,
                train_instances,
                args.vehicles,
                n_qubits=size,
                seed=outer_seed * 10_000 + size,
                alphas=args.alphas,
                ordering=args.ordering,
            )
            feature_matrix = np.vstack(all_features)
            rows.append(
                {
                    "n_locations": size,
                    "outer_seed": outer_seed,
                    "angle_scale": angle_scale,
                    "validation_route_cost": float(np.mean(costs)),
                    "validation_route_cost_std": float(np.std(costs)),
                    "feature_mean_abs": float(np.mean(np.abs(feature_matrix))),
                    "feature_variance": float(np.sum(np.var(feature_matrix, axis=0))),
                }
            )
            print(
                f"size={size} angle_scale={angle_scale:.4f} "
                f"validation_cost={np.mean(costs):.4f}"
            )

    write_csv(os.path.join(args.output_dir, "encoding_sweep.csv"), rows)
    best = {}
    for size in args.sizes:
        size_rows = [row for row in rows if row["n_locations"] == size]
        if size_rows:
            best[size] = min(
                size_rows, key=lambda row: row["validation_route_cost"]
            )["angle_scale"]
    with open(
        os.path.join(args.output_dir, "encoding_sweep_selection.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({str(key): value for key, value in best.items()}, handle, indent=2)

    lines = ["# Encoding angle-scale sweep", ""]
    lines.append(
        "Selection criterion is validation route cost with a trained readout; "
        "feature statistics are diagnostics only."
    )
    lines.append("")
    lines.append(
        "| Size | Angle scale | Validation route cost | Feature mean abs | Feature variance | Selected |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {n_locations} | {angle_scale:.4f} | {validation_route_cost:.4f} | "
            "{feature_mean_abs:.4f} | {feature_variance:.5f} | {selected} |".format(
                selected="yes" if best.get(row["n_locations"]) == row["angle_scale"] else "",
                **row,
            )
        )
    with open(
        os.path.join(args.output_dir, "encoding_sweep.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")


def _cross_validated_route_cost(
    features: Sequence[np.ndarray],
    assignments: Sequence[np.ndarray],
    instances: Sequence[Instance],
    n_vehicles: int,
    n_qubits: int,
    seed: int,
    alphas: Sequence[float],
    ordering: str,
) -> list[float]:
    """Out-of-fold route cost used only for encoding-scale selection."""
    from modules.readout import RidgeReadout

    n_samples = len(features)
    folds = min(5, n_samples)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    costs = np.zeros(n_samples)
    solver = ReservoirVRPSolver(n_reservoir_qubits=n_qubits)
    for fold in np.array_split(indices, folds):
        mask = np.ones(n_samples, dtype=bool)
        mask[fold] = False
        readout = RidgeReadout(alphas=tuple(alphas), cv_folds=5, seed=seed)
        readout.fit(
            flat_feature_rows([features[index] for index in np.where(mask)[0]]),
            np.vstack(
                [assignments[index].reshape(-1) for index in np.where(mask)[0]]
            ),
        )
        for index in fold:
            logits = readout.predict(features[index].reshape(1, -1)).reshape(-1)
            assignment = np.argmax(
                logits.reshape(instances[index].n_locations - 1, n_vehicles), axis=1
            )
            routes = solver._decode_assignment(  # pylint: disable=protected-access
                assignment,
                n_vehicles,
                instances[index].distances,
                ordering=ordering,
                demands=instances[index].demands,
                capacity=instances[index].capacity,
            )
            costs[index] = route_cost(routes, instances[index].distances)
    return costs.tolist()


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "encoding-sweep"), default="benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--per-size", type=int, default=250)
    parser.add_argument("--train-per-size", type=int, default=200)
    parser.add_argument("--outer-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ortools-seconds", type=float, default=1.0)
    parser.add_argument("--ortools-latency-seconds", type=float, default=0.1)
    parser.add_argument("--readout", choices=("flat", "node"), default="flat")
    parser.add_argument("--ordering", choices=("index", "nn2opt"), default="nn2opt")
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--angle-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--use-depot-diagonal", action="store_true")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--trainer-seed-base", type=int, default=1000)
    parser.add_argument("--reservoir-seed-base", type=int, default=2000)
    parser.add_argument("--trainer-samples", type=int, default=20)
    parser.add_argument("--trainer-maxiter", type=int, default=100)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results"))
    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    if args.mode == "benchmark":
        run_benchmark(args)
    else:
        run_encoding_sweep(args)


if __name__ == "__main__":
    main()
