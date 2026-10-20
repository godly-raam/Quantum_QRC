"""Pair-readout diagnostics: angle-scale selection and readout-only controls.

This script reuses the archived feature cache, so it does not rerun finished
quantum simulations.  It solves OR-Tools only for the training instances
needed to build label-free co-assignment targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.fast_sim import FeatureCache  # noqa: E402
from modules.quantum_reservoir_vrp import ReservoirVRPSolver  # noqa: E402
from modules.readout import (  # noqa: E402
    RidgeReadout,
    cluster_similarity,
    customer_pairs,
    distance_node_features,
    pair_feature_matrix,
    pair_targets_from_routes,
    reservoir_pair_components,
)
from scripts.benchmark_fair import (  # noqa: E402
    EchoStateNetwork,
    _decode,
    _lcu_theta,
    _quantum_theta_hash,
    bootstrap_ci,
    make_split,
    quantum_features,
    route_cost,
    solve_ortools,
)


def build_random_ring_solver(
    size: int, outer_seed: int, angle_scale: float
) -> ReservoirVRPSolver:
    rng = np.random.default_rng(2000 + outer_seed * 100 + size)
    params = rng.uniform(0.0, 2.0 * np.pi, size * 2)
    return ReservoirVRPSolver(
        n_reservoir_qubits=size,
        trained_params=params,
        legacy_decoder=False,
        ordering="nn2opt",
        angle_scale=angle_scale,
        use_depot_diagonal=False,
        random_seed=outer_seed,
    )


def collect_quantum(
    cache: FeatureCache,
    solver: ReservoirVRPSolver,
    instances,
    n_vehicles: int,
    benchmark_seed: int,
):
    reservoir = solver.reservoir
    theta_hash = _quantum_theta_hash(reservoir, benchmark_seed)
    angle_scale = reservoir.angle_scale
    features = []
    for instance in instances:
        theta = _lcu_theta(
            instance.n_locations,
            n_vehicles,
            instance.seed + benchmark_seed,
        )
        features.append(
            cache.get_or_compute(
                instance.n_locations,
                instance.seed,
                angle_scale,
                theta_hash,
                lambda: quantum_features(
                    reservoir, instance.distances, n_vehicles, theta
                ),
            )
        )
    cache.flush()
    return features


def collect_esn(cache: FeatureCache, size: int, instances, benchmark_seed: int):
    esn = EchoStateNetwork(size, size**2, seed=benchmark_seed)
    theta_hash = f"esn-v2:{benchmark_seed}:{size}:{size**2}"
    features = []
    for instance in instances:
        features.append(
            cache.get_or_compute(
                size,
                instance.seed,
                0.0,
                theta_hash,
                lambda: esn.features(instance.distances),
            )
        )
    cache.flush()
    return features


def pair_rows(features, n_locations: int, include_zz: bool = True) -> np.ndarray:
    observables, zz_matrix = reservoir_pair_components(features, n_locations)
    return pair_feature_matrix(
        observables, n_locations, zz_matrix if include_zz else None
    )


def distance_rows(instance, include_dij: bool) -> np.ndarray:
    rows = pair_feature_matrix(
        distance_node_features(instance.distances), instance.n_locations
    )
    if include_dij:
        extra = np.asarray(
            [
                instance.distances[left, right]
                for left, right in customer_pairs(instance.n_locations)
            ],
            dtype=float,
        )[:, None]
        rows = np.hstack([rows, extra])
    return rows


def fit_reader(rows: np.ndarray, targets: np.ndarray, seed: int) -> RidgeReadout:
    return RidgeReadout(
        mode="pair",
        alphas=(1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0),
        cv_folds=5,
        seed=seed,
    ).fit(rows, targets[:, None])


def predict_labels(
    readout: RidgeReadout,
    rows: np.ndarray,
    n_locations: int,
    n_vehicles: int,
) -> np.ndarray:
    values = np.asarray(readout.predict(rows), dtype=float).reshape(-1)
    n_customers = n_locations - 1
    similarity = np.zeros((n_customers, n_customers), dtype=float)
    for (left, right), value in zip(customer_pairs(n_locations), values):
        similarity[left - 1, right - 1] = value
        similarity[right - 1, left - 1] = value
    np.fill_diagonal(similarity, 1.0)
    return cluster_similarity(similarity, n_vehicles)


def cost_for_labels(
    solver: ReservoirVRPSolver, labels: np.ndarray, instance
) -> float:
    routes = _decode(solver, labels, instance, 4, "nn2opt")
    return route_cost(routes, instance.distances)


def load_reference_costs() -> dict[tuple[int, int, int], float]:
    import csv

    references: dict[tuple[int, int, int], float] = {}
    path = "results/aggregate/fair_benchmark_runs.csv"
    if not os.path.exists(path):
        return references
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "random_assignment":
                continue
            references[
                (
                    int(row["n_locations"]),
                    int(row["outer_seed"]),
                    int(row["instance_seed"]),
                )
            ] = float(row["reference_cost"])
    return references


def solve_training_routes(size: int, outer_seed: int, instances, seconds: float):
    routes = []
    for index, instance in enumerate(instances):
        route, _, _ = solve_ortools(
            instance.distances,
            4,
            seconds,
            seed=outer_seed * 1000 + index,
            demands=instance.demands,
            capacity=instance.capacity,
        )
        routes.append(route)
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 12])
    parser.add_argument("--outer-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--per-size", type=int, default=250)
    parser.add_argument("--train-per-size", type=int, default=200)
    parser.add_argument("--ortools-seconds", type=float, default=0.5)
    parser.add_argument("--cache-dir", default="results/feature_cache")
    parser.add_argument("--output-dir", default="results/pairwise_diagnostics")
    args = parser.parse_args()

    cache = FeatureCache(args.cache_dir)
    references = load_reference_costs()
    os.makedirs(args.output_dir, exist_ok=True)

    scale_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    control_rows: list[dict] = []
    comparison_rows: list[dict] = []
    selections: dict[int, float] = {}

    for size in args.sizes:
        scales = []
        for scale in (1.0, 1.0 / math.sqrt(size), 1.0 / size, 0.25):
            if not any(abs(scale - existing) < 1e-12 for existing in scales):
                scales.append(scale)
        n_pairs = len(customer_pairs(size))

        training: dict[int, tuple] = {}
        for outer_seed in args.outer_seeds:
            train, test = make_split(
                outer_seed,
                size,
                args.per_size + args.train_per_size,
                args.train_per_size,
            )
            routes = solve_training_routes(
                size, outer_seed, train, args.ortools_seconds
            )
            targets = np.vstack(
                [pair_targets_from_routes(route, size) for route in routes]
            )
            training[outer_seed] = (train, test, routes, targets)

        validation_costs: dict[float, list[float]] = defaultdict(list)
        for scale in scales:
            for outer_seed in args.outer_seeds:
                train, _, _, targets = training[outer_seed]
                benchmark_seed = outer_seed * 10_000 + size
                solver = build_random_ring_solver(size, outer_seed, scale)
                features = collect_quantum(
                    cache, solver, train, 4, benchmark_seed
                )
                split = int(0.6 * len(train))
                train_rows = np.vstack(
                    [pair_rows(feature, size) for feature in features[:split]]
                )
                validation_rows = np.vstack(
                    [pair_rows(feature, size) for feature in features[split:]]
                )
                readout = fit_reader(
                    train_rows,
                    targets[:split].reshape(-1),
                    benchmark_seed,
                )
                costs = []
                for offset, instance in enumerate(train[split:]):
                    labels = predict_labels(
                        readout,
                        validation_rows[
                            offset * n_pairs : (offset + 1) * n_pairs
                        ],
                        size,
                        4,
                    )
                    costs.append(cost_for_labels(solver, labels, instance))
                validation_costs[scale].extend(costs)
            scale_rows.append(
                {
                    "n_locations": size,
                    "angle_scale": scale,
                    "validation_cost": float(np.mean(validation_costs[scale])),
                    "n_validation_runs": len(validation_costs[scale]),
                }
            )

        selected = min(scales, key=lambda scale: np.mean(validation_costs[scale]))
        selections[size] = selected

        for outer_seed in args.outer_seeds:
            train, test, _, targets = training[outer_seed]
            benchmark_seed = outer_seed * 10_000 + size
            solver = build_random_ring_solver(size, outer_seed, selected)
            features = collect_quantum(
                cache, solver, train + list(test), 4, benchmark_seed
            )
            train_rows = np.vstack(
                [pair_rows(feature, size) for feature in features[: len(train)]]
            )
            test_rows = np.vstack(
                [pair_rows(feature, size) for feature in features[len(train) :]]
            )
            readout = fit_reader(
                train_rows, targets.reshape(-1), benchmark_seed
            )
            costs = []
            gaps = []
            for offset, instance in enumerate(test):
                labels = predict_labels(
                    readout,
                    test_rows[offset * n_pairs : (offset + 1) * n_pairs],
                    size,
                    4,
                )
                cost = cost_for_labels(solver, labels, instance)
                reference = references.get((size, outer_seed, instance.seed))
                costs.append(cost)
                gaps.append(
                    (cost - reference) / reference if reference else float("nan")
                )
            evaluation_rows.append(
                {
                    "n_locations": size,
                    "outer_seed": outer_seed,
                    "angle_scale": selected,
                    "mean_cost": float(np.mean(costs)),
                    "mean_gap": float(np.nanmean(gaps)),
                    "n_instances": len(costs),
                }
            )

        for outer_seed in args.outer_seeds:
            train, test, _, targets = training[outer_seed]
            benchmark_seed = outer_seed * 10_000 + size
            ring_solver = build_random_ring_solver(size, outer_seed, 1.0)
            ring_features = collect_quantum(
                cache, ring_solver, train + list(test), 4, benchmark_seed
            )
            esn_features = collect_esn(
                cache, size, train + list(test), benchmark_seed
            )
            controls = {
                "quantum_with_zz": [
                    pair_rows(feature, size, include_zz=True)
                    for feature in ring_features
                ],
                "quantum_no_zz": [
                    pair_rows(feature, size, include_zz=False)
                    for feature in ring_features
                ],
                "esn_pair_products": [
                    pair_rows(feature, size, include_zz=True)
                    for feature in esn_features
                ],
            }
            for method, rows in controls.items():
                train_rows = np.vstack(rows[: len(train)])
                test_rows = np.vstack(rows[len(train) :])
                readout = fit_reader(
                    train_rows, targets.reshape(-1), benchmark_seed
                )
                costs = []
                for offset, instance in enumerate(test):
                    labels = predict_labels(
                        readout,
                        test_rows[offset * n_pairs : (offset + 1) * n_pairs],
                        size,
                        4,
                    )
                    costs.append(cost_for_labels(ring_solver, labels, instance))
                control_rows.append(
                    {
                        "n_locations": size,
                        "outer_seed": outer_seed,
                        "control": method,
                        "mean_cost": float(np.mean(costs)),
                        "n_instances": len(costs),
                    }
                )

            for method, include_dij in (
                ("distance_pair", False),
                ("distance_pair_dij", True),
            ):
                train_rows = np.vstack(
                    [distance_rows(instance, include_dij) for instance in train]
                )
                test_rows = np.vstack(
                    [distance_rows(instance, include_dij) for instance in test]
                )
                readout = fit_reader(
                    train_rows, targets.reshape(-1), benchmark_seed
                )
                costs = []
                for offset, instance in enumerate(test):
                    labels = predict_labels(
                        readout,
                        test_rows[offset * n_pairs : (offset + 1) * n_pairs],
                        size,
                        4,
                    )
                    costs.append(cost_for_labels(ring_solver, labels, instance))
                control_rows.append(
                    {
                        "n_locations": size,
                        "outer_seed": outer_seed,
                        "control": method,
                        "mean_cost": float(np.mean(costs)),
                        "n_instances": len(costs),
                    }
                )

    import csv

    def write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with open(os.path.join(args.output_dir, name), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write("angle_scale_validation.csv", scale_rows)
    write("angle_scale_evaluation.csv", evaluation_rows)
    write("readout_controls.csv", control_rows)
    with open(
        os.path.join(args.output_dir, "angle_scale_selection.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({str(key): value for key, value in selections.items()}, handle, indent=2)

    # Paired comparisons among controls, by size.
    for size in args.sizes:
        by_control: dict[str, dict[int, float]] = defaultdict(dict)
        for row in control_rows:
            if int(row["n_locations"]) == size:
                key = int(row["outer_seed"]) * 10_000 + int(row["n_instances"])
                by_control[row["control"]][key] = float(row["mean_cost"])
        # The per-seed aggregate is enough for a descriptive comparison.
        print(size, {control: np.mean(list(values.values())) for control, values in by_control.items()})

    print(json.dumps({"selections": selections}, indent=2))


if __name__ == "__main__":
    main()
