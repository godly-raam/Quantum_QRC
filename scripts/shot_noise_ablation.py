"""Shot-noise ablation on cached quantum features plus stronger classical baselines.

For each cached ring-reservoir feature vector we simulate Pauli expectation
estimates from ``S`` shots.  A Pauli expectation ``o`` is estimated by
``2 * Binomial(S, (1 + o) / 2) / S - 1``, whose variance is
``(1 - o**2) / S``.  The pair readout is retrained on noisy training features
and evaluated on independently noised test features.

The report is paired by ``(outer_seed, instance_seed)`` against the existing
random-assignment and matched-ESN runs, and against two stronger classical
feature maps with the same total feature count ``3N + N(N-1)/2``:

* a large ESN with that many reservoir units and three recurrent steps;
* random Fourier features with that many components.

Definitions are intentionally explicit: the quantum and classical vectors are
split into the same ``3N`` node slots and ``N(N-1)/2`` pair slots before the
shared pair readout is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.fast_sim import FeatureCache  # noqa: E402
from modules.readout import (  # noqa: E402
    customer_pairs,
    pair_feature_matrix,
    pair_targets_from_routes,
)
from scripts.benchmark_fair import (  # noqa: E402
    EchoStateNetwork,
    bootstrap_ci,
    make_split,
    solve_ortools,
)
from scripts.pairwise_diagnostics import (  # noqa: E402
    build_random_ring_solver,
    collect_quantum,
    cost_for_labels,
    fit_reader,
    predict_labels,
)


def target_path(root: str, size: int, outer_seed: int) -> str:
    return os.path.join(root, f"targets_n{size}_seed{outer_seed}.json")


def solve_targets_task(
    size: int, outer_seed: int, train_count: int, test_count: int, seconds: float
):
    train, _ = make_split(outer_seed, size, train_count + test_count, train_count)
    routes = []
    for index, instance in enumerate(train):
        route, _, _ = solve_ortools(
            instance.distances,
            4,
            seconds,
            seed=outer_seed * 1000 + index,
            demands=instance.demands,
            capacity=instance.capacity,
        )
        routes.append(route)
    return {
        "n_locations": size,
        "outer_seed": outer_seed,
        "instance_seeds": [instance.seed for instance in train],
        "routes": routes,
    }


def ensure_targets(
    root: str, sizes, seeds, train_count: int, test_count: int, seconds: float
) -> None:
    os.makedirs(root, exist_ok=True)
    missing = [
        (size, seed)
        for size in sizes
        for seed in seeds
        if not os.path.exists(target_path(root, size, seed))
    ]
    if not missing:
        return
    print(f"solving {len(missing)} target sets with 2 workers", flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {
            (size, seed): pool.submit(
                solve_targets_task,
                size,
                seed,
                train_count,
                test_count,
                seconds,
            )
            for size, seed in missing
        }
        for (size, seed), future in futures.items():
            payload = future.result()
            with open(target_path(root, size, seed), "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            print(f"targets size={size} seed={seed} ready", flush=True)


def load_targets(root: str, size: int, outer_seed: int):
    with open(target_path(root, size, outer_seed), encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["routes"]


def add_shot_noise(features: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    values = np.clip(np.asarray(features, dtype=float), -1.0, 1.0)
    probabilities = 0.5 * (1.0 + values)
    counts = rng.binomial(int(shots), probabilities)
    return 2.0 * counts / float(shots) - 1.0


def split_reservoir_vector(vector: np.ndarray, size: int):
    from modules.fast_sim import zz_pair_index

    vector = np.asarray(vector, dtype=float)
    observables = vector[: 3 * size].reshape(size, 3)
    pair_values = vector[3 * size :]
    zz_matrix = np.zeros((size, size), dtype=float)
    for (left, right), position in zz_pair_index(size).items():
        zz_matrix[left, right] = pair_values[position - 3 * size]
        zz_matrix[right, left] = pair_values[position - 3 * size]
    return observables, zz_matrix


def rows_from_vector(vector: np.ndarray, size: int) -> np.ndarray:
    observables, zz_matrix = split_reservoir_vector(vector, size)
    return pair_feature_matrix(observables, size, zz_matrix)


def rows_from_state(state: np.ndarray, size: int) -> np.ndarray:
    """Treat a same-length classical state as node and pair slots."""
    return rows_from_vector(state, size)


class LargeESN:
    """ESN with ``3N + N(N-1)/2`` reservoir units and three recurrent steps."""

    def __init__(
        self,
        size: int,
        seed: int,
        spectral_radius: float = 0.9,
        input_scale: float = 0.2,
        steps: int = 3,
    ) -> None:
        self.size = size
        self.units = 3 * size + size * (size - 1) // 2
        self.steps = steps
        rng = np.random.default_rng(seed)
        self.input_weights = rng.uniform(
            -input_scale, input_scale, size=(self.units, size * size)
        )
        recurrent = rng.uniform(-1.0, 1.0, size=(self.units, self.units))
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self.recurrent_weights = recurrent * (spectral_radius / max(radius, 1e-12))

    def feature_vector(self, distances: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(distances)))
        normalised = distances / scale if scale > 0.0 else distances
        state = np.zeros(self.units, dtype=float)
        for _ in range(self.steps):
            state = np.tanh(
                self.input_weights @ normalised.ravel()
                + self.recurrent_weights @ state
            )
        return state


class RandomFourierFeatures:
    """RBF random Fourier features with the same total feature count.

    This is the standard ``sqrt(2/M) * cos(W x + b)`` construction with
    ``W ~ N(0, 2 * gamma)`` and ``b ~ Uniform(0, 2*pi)``.  It is implemented
    directly so importing scikit-learn is not required by the benchmark.
    """

    def __init__(self, size: int, seed: int, gamma: float) -> None:
        self.size = size
        self.units = 3 * size + size * (size - 1) // 2
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            0.0, np.sqrt(2.0 * gamma), (self.units, size * size)
        )
        self.offset = rng.uniform(0.0, 2.0 * np.pi, self.units)

    def fit(self, instances) -> None:
        """Gamma is supplied by the caller; kept for a uniform interface."""

    def feature_vector(self, distances: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(distances)))
        normalised = distances / scale if scale > 0.0 else distances
        projection = self.weights @ normalised.ravel()
        return np.sqrt(2.0 / self.units) * np.cos(projection + self.offset)


def median_gamma(instances) -> float:
    matrix = np.vstack(
        [instance.distances.ravel() / np.max(instance.distances) for instance in instances]
    )
    rng = np.random.default_rng(0)
    left = rng.integers(0, len(matrix), 4000)
    right = rng.integers(0, len(matrix), 4000)
    squared = np.sum((matrix[left] - matrix[right]) ** 2, axis=1)
    positive = squared[squared > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    return 1.0 / (2.0 * median)


def evaluate_feature_rows(
    train_rows,
    test_rows,
    targets,
    test_instances,
    solver,
    seed: int,
):
    train_matrix = np.vstack(train_rows)
    test_matrix = np.vstack(test_rows)
    readout = fit_reader(train_matrix, targets.reshape(-1), seed)
    costs = []
    for offset, instance in enumerate(test_instances):
        n_pairs = len(customer_pairs(instance.n_locations))
        labels = predict_labels(
            readout,
            test_matrix[offset * n_pairs : (offset + 1) * n_pairs],
            instance.n_locations,
            4,
        )
        costs.append(cost_for_labels(solver, labels, instance))
    return costs


def load_run_costs(path: str):
    costs = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            key = (
                int(row["n_locations"]),
                int(row["outer_seed"]),
                int(row["instance_seed"]),
                row["method"],
            )
            costs[key] = float(row["cost"])
    return costs


def write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--outer-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--shots", nargs="+", type=int, default=[100, 1_000, 10_000, 100_000])
    parser.add_argument("--noise-reps", type=int, default=5)
    parser.add_argument("--per-size", type=int, default=250)
    parser.add_argument("--train-per-size", type=int, default=200)
    parser.add_argument("--ortools-seconds", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="results/feature_cache")
    parser.add_argument("--output-dir", default="results/shot_noise")
    parser.add_argument(
        "--runs-csv",
        default="results/aggregate/fair_benchmark_runs.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    targets_root = os.path.join(args.output_dir, "targets")
    ensure_targets(
        targets_root,
        args.sizes,
        args.outer_seeds,
        args.train_per_size,
        args.per_size,
        args.ortools_seconds,
    )

    cache = FeatureCache(args.cache_dir)
    run_costs = load_run_costs(args.runs_csv)

    baseline_rows: list[dict] = []
    baseline_costs: dict[tuple, float] = {}
    shot_rows: list[dict] = []
    repeated_rows: list[dict] = []
    per_instance_rows: list[dict] = []

    for size in args.sizes:
        print(f"size {size}", flush=True)
        for outer_seed in args.outer_seeds:
            train, test = make_split(
                outer_seed,
                size,
                args.per_size + args.train_per_size,
                args.train_per_size,
            )
            routes = load_targets(targets_root, size, outer_seed)
            targets = np.vstack(
                [pair_targets_from_routes(route, size) for route in routes]
            )
            solver = build_random_ring_solver(size, outer_seed, 1.0)
            benchmark_seed = outer_seed * 10_000 + size
            features = collect_quantum(
                cache, solver, train + list(test), 4, benchmark_seed
            )
            solver_seed = benchmark_seed

            # Classical baselines, all trained on the same target routes.
            matched_esn = EchoStateNetwork(size, size**2, seed=benchmark_seed)
            large_esn = LargeESN(size, seed=benchmark_seed + 77)
            rff = RandomFourierFeatures(
                size, seed=benchmark_seed + 91, gamma=median_gamma(train)
            )
            rff.fit(train)
            models = (
                ("matched_esn", matched_esn.features),
                ("large_esn", large_esn.feature_vector),
                ("rff", rff.feature_vector),
            )
            for name, feature_fn in models:
                train_rows = [
                    rows_from_vector(feature_fn(instance.distances), size)
                    for instance in train
                ]
                test_rows = [
                    rows_from_vector(feature_fn(instance.distances), size)
                    for instance in test
                ]
                costs = evaluate_feature_rows(
                    train_rows, test_rows, targets, test, solver, solver_seed
                )
                for instance, cost in zip(test, costs):
                    baseline_costs[
                        (size, outer_seed, instance.seed, name)
                    ] = float(cost)
                    baseline_rows.append(
                        {
                            "n_locations": size,
                            "outer_seed": outer_seed,
                            "instance_seed": instance.seed,
                            "method": name,
                            "cost": float(cost),
                        }
                    )

            train_vectors = features[: len(train)]
            test_vectors = features[len(train) :]
            for shots in args.shots:
                rep_costs = []
                for rep in range(args.noise_reps):
                    rng = np.random.default_rng(
                        size * 1_000_000
                        + outer_seed * 10_000
                        + shots
                        + rep * 97
                    )
                    noisy_train = [
                        add_shot_noise(vector, shots, rng) for vector in train_vectors
                    ]
                    noisy_test = [
                        add_shot_noise(vector, shots, rng) for vector in test_vectors
                    ]
                    train_rows = [rows_from_vector(vector, size) for vector in noisy_train]
                    test_rows = [rows_from_vector(vector, size) for vector in noisy_test]
                    costs = evaluate_feature_rows(
                        train_rows, test_rows, targets, test, solver, solver_seed
                    )
                    rep_costs.append(costs)
                    repeated_rows.append(
                        {
                            "n_locations": size,
                            "outer_seed": outer_seed,
                            "shots": shots,
                            "noise_rep": rep,
                            "mean_cost": float(np.mean(costs)),
                        }
                    )
                mean_costs = np.mean(np.asarray(rep_costs), axis=0)
                for instance, cost in zip(test, mean_costs):
                    per_instance_rows.append(
                        {
                            "n_locations": size,
                            "outer_seed": outer_seed,
                            "instance_seed": instance.seed,
                            "shots": shots,
                            "mean_cost": float(cost),
                        }
                    )

    write_csv(
        os.path.join(args.output_dir, "classical_baseline_runs.csv"),
        baseline_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "shot_noise_repetitions.csv"),
        repeated_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "shot_noise_per_instance.csv"),
        per_instance_rows,
    )

    baseline_summary = []
    for size in args.sizes:
        for name in ("matched_esn", "large_esn", "rff"):
            values = [
                cost
                for key, cost in baseline_costs.items()
                if key[0] == size and key[3] == name
            ]
            baseline_summary.append(
                {
                    "n_locations": size,
                    "method": name,
                    "n_runs": len(values),
                    "mean_cost": float(np.mean(values)),
                }
            )
    write_csv(
        os.path.join(args.output_dir, "classical_baseline_summary.csv"),
        baseline_summary,
    )

    lookup = {}
    for row in per_instance_rows:
        lookup[
            (
                row["n_locations"],
                row["outer_seed"],
                row["instance_seed"],
                row["shots"],
            )
        ] = row["mean_cost"]

    for size in args.sizes:
        for shots in args.shots:
            differences_random = []
            differences_esn = []
            differences_large = []
            differences_rff = []
            costs = []
            for key, noisy_cost in lookup.items():
                key_size, outer_seed, instance_seed, key_shots = key
                if key_size != size or key_shots != shots:
                    continue
                random_cost = run_costs.get(
                    (size, outer_seed, instance_seed, "random_assignment")
                )
                esn_cost = baseline_costs.get(
                    (size, outer_seed, instance_seed, "matched_esn")
                )
                if esn_cost is None:
                    esn_cost = run_costs.get(
                        (size, outer_seed, instance_seed, "classical_esn")
                    )
                if random_cost is None or esn_cost is None:
                    continue
                costs.append(noisy_cost)
                differences_random.append(noisy_cost - random_cost)
                differences_esn.append(noisy_cost - esn_cost)
                large = baseline_costs.get(
                    (size, outer_seed, instance_seed, "large_esn")
                )
                rff = baseline_costs.get((size, outer_seed, instance_seed, "rff"))
                if large is not None:
                    differences_large.append(noisy_cost - large)
                if rff is not None:
                    differences_rff.append(noisy_cost - rff)
            mean_r, low_r, high_r = bootstrap_ci(
                np.asarray(differences_random), seed=size * 100 + shots
            )
            mean_e, low_e, high_e = bootstrap_ci(
                np.asarray(differences_esn), seed=size * 100 + shots + 1
            )
            mean_l, low_l, high_l = bootstrap_ci(
                np.asarray(differences_large), seed=size * 100 + shots + 2
            )
            mean_f, low_f, high_f = bootstrap_ci(
                np.asarray(differences_rff), seed=size * 100 + shots + 3
            )
            shot_rows.append(
                {
                    "n_locations": size,
                    "shots": shots,
                    "n_pairs": len(differences_random),
                    "mean_cost": float(np.mean(costs)),
                    "diff_random": mean_r,
                    "ci_low_random": low_r,
                    "ci_high_random": high_r,
                    "beats_random": bool(high_r < 0.0),
                    "diff_esn": mean_e,
                    "ci_low_esn": low_e,
                    "ci_high_esn": high_e,
                    "beats_esn": bool(high_e < 0.0),
                    "diff_large_esn": mean_l,
                    "ci_low_large_esn": low_l,
                    "ci_high_large_esn": high_l,
                    "beats_large_esn": bool(high_l < 0.0),
                    "diff_rff": mean_f,
                    "ci_low_rff": low_f,
                    "ci_high_rff": high_f,
                    "beats_rff": bool(high_f < 0.0),
                }
            )

    write_csv(os.path.join(args.output_dir, "shot_noise_summary.csv"), shot_rows)

    threshold = {}
    for size in (10, 12):
        ordered = sorted(
            (row for row in shot_rows if row["n_locations"] == size),
            key=lambda row: row["shots"],
        )
        vanished = next(
            (row["shots"] for row in ordered if not row["beats_random"]), None
        )
        threshold[str(size)] = vanished

    with open(
        os.path.join(args.output_dir, "shot_noise_threshold.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(threshold, handle, indent=2)

    lines = [
        "# Shot-noise ablation",
        "",
        "Noise model: each Pauli expectation is estimated from `S` binomial shots, "
        "giving variance `(1 - <O>^2) / S`. The pair readout is retrained for every "
        "`(size, outer_seed, S, noise_rep)`.",
        "",
        "| Size | Shots | Mean cost | vs random (95% CI) | beats random | vs matched ESN (95% CI) | vs large ESN (95% CI) | vs RFF (95% CI) |",
        "|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in shot_rows:
        lines.append(
            "| {n_locations} | {shots} | {mean_cost:.4f} | "
            "{diff_random:+.4f} [{ci_low_random:+.4f}, {ci_high_random:+.4f}] | "
            "{beats_random} | {diff_esn:+.4f} [{ci_low_esn:+.4f}, {ci_high_esn:+.4f}] | "
            "{diff_large_esn:+.4f} [{ci_low_large_esn:+.4f}, {ci_high_large_esn:+.4f}] | "
            "{diff_rff:+.4f} [{ci_low_rff:+.4f}, {ci_high_rff:+.4f}] |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Smallest S where the edge over random disappears (CI includes 0 or cost is worse):",
            json.dumps(threshold, indent=2),
            "",
        ]
    )
    with open(
        os.path.join(args.output_dir, "shot_noise.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines))

    print(json.dumps(threshold, indent=2), flush=True)


if __name__ == "__main__":
    main()
