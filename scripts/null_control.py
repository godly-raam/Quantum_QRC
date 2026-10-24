"""Cache-only null controls for the label-free pairwise readout.

Every null is applied to the cached ``quantum_ring_trained`` feature vectors
and then passed through exactly the same pair construction, ridge/CV readout,
and ``nn2opt`` decoder used by the fair benchmark.  No quantum simulator and no
OR-Tools call is allowed here: features come from ``results/feature_cache``,
labels from a cached route JSON, and the random/ESN reference costs from the
benchmark checkpoints.

The three nulls are:

* ``shuffled``: independently permute each feature column across instances,
  separately for train and test;
* ``gaussian``: i.i.d. Gaussian noise whose per-column mean/std match the
  training features;
* ``constant``: an intercept-only readout (one constant feature).

The script also evaluates the unmodified quantum features and the matched ESN
features as references.  It is resumable: one row per ``(size, seed, variant)``
is appended as each combination completes, and existing rows are skipped.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.readout import customer_pairs, pair_targets_from_routes  # noqa: E402
from scripts.benchmark_fair import (  # noqa: E402
    EchoStateNetwork,
    bootstrap_ci,
    make_split,
)
from scripts.pairwise_diagnostics import (  # noqa: E402
    build_random_ring_solver,
    cost_for_labels,
    fit_reader,
    predict_labels,
)
from scripts.shot_noise_ablation import rows_from_vector  # noqa: E402

VARIANTS = ("quantum", "esn", "shuffled", "gaussian", "constant")
REFERENCES = ("random_assignment",) + VARIANTS
CHECKPOINT_METHODS = ("random_assignment", "classical_esn", "quantum_ring_trained")
MATCH_TOLERANCE = 0.5


def _forbidden(*_args, **_kwargs):
    raise RuntimeError(
        "null_control is cache-only: a Statevector, AerSimulator, or OR-Tools "
        "call was reached"
    )


def install_no_simulation_guards() -> None:
    """Make any simulator/OR-Tools call fail loudly."""
    import modules.fast_sim as fast_sim

    fast_sim.statevector_from_circuit = _forbidden  # type: ignore[assignment]

    for module_name in (
        "scripts.benchmark_fair",
        "scripts.shot_noise_ablation",
        "scripts.pairwise_diagnostics",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "solve_ortools"):
            module.solve_ortools = _forbidden  # type: ignore[assignment]

    try:
        from qiskit.quantum_info import Statevector

        Statevector.from_instruction = classmethod(_forbidden)  # type: ignore[assignment]
    except Exception:
        pass

    if not any(
        isinstance(finder, _BlockForbiddenImports) for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _BlockForbiddenImports())


class _BlockForbiddenImports:
    """Import hook that refuses simulator/OR-Tools imports inside the worker."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ortools" or fullname.startswith("ortools."):
            raise RuntimeError(
                "null_control is cache-only: an OR-Tools import was reached"
            )
        if fullname == "qiskit_aer" or fullname.startswith("qiskit_aer."):
            raise RuntimeError(
                "null_control is cache-only: an AerSimulator import was reached"
            )
        return None


def target_path(root: str, size: int, outer_seed: int) -> str:
    return os.path.join(root, f"targets_n{size}_seed{outer_seed}.json")


def load_cached_labels(
    targets_dir: str,
    size: int,
    outer_seed: int,
    train_instances,
) -> np.ndarray:
    """Load cached OR-Tools routes and convert them to pair labels."""
    path = target_path(targets_dir, size, outer_seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No cached OR-Tools labels for "
            f"size={size} seed={outer_seed}: {path}. The benchmark checkpoints "
            "store costs but not route labels. Provide a target JSON in this "
            "directory; null_control will not call OR-Tools."
        )
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = [instance.seed for instance in train_instances]
    if payload.get("instance_seeds") != expected:
        raise ValueError(
            f"cached labels in {path} do not match the train split for "
            f"size={size} seed={outer_seed}"
        )
    routes = payload["routes"]
    if len(routes) != len(train_instances):
        raise ValueError(f"expected {len(train_instances)} routes in {path}")
    return np.vstack(
        [pair_targets_from_routes(route, size) for route in routes]
    )


def _test_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return [row for row in payload.get("rows", []) if row.get("split") == "test"]


def load_checkpoints(
    size: int,
    outer_seed: int,
    test_instances,
) -> list[dict]:
    """Return usable benchmark checkpoints, most complete first."""
    pattern = os.path.join(
        PROJECT_ROOT,
        "results",
        "**",
        "checkpoints",
        f"size{size}_seed{outer_seed}.json",
    )
    test_seeds = {instance.seed for instance in test_instances}
    checkpoints = []
    for path in glob.glob(pattern, recursive=True):
        rows = _test_rows(path)
        methods = {row["method"] for row in rows}
        if not set(CHECKPOINT_METHODS).issubset(methods):
            continue
        by_method: dict[str, dict[int, float]] = {method: {} for method in CHECKPOINT_METHODS}
        reference: dict[int, float] = {}
        for row in rows:
            seed = int(row["instance_seed"])
            if seed not in test_seeds:
                continue
            method = row["method"]
            if method in by_method:
                by_method[method][seed] = float(row["cost"])
            reference[seed] = float(row["reference_cost"])
        if any(len(by_method[method]) != len(test_seeds) for method in CHECKPOINT_METHODS):
            continue
        checkpoints.append(
            {
                "path": path,
                "costs": by_method,
                "reference": reference,
                "n_rows": len(rows),
            }
        )
    checkpoints.sort(key=lambda item: (-item["n_rows"], item["path"]))
    return checkpoints


def load_candidate_shards(cache_dir: str, size: int, instance_seeds: Sequence[int]):
    """Load every full-coverage feature shard for a size/instance set."""
    wanted = set(int(seed) for seed in instance_seeds)
    candidates = []
    for path in sorted(glob.glob(os.path.join(cache_dir, f"n{size}", "*.npz"))):
        with np.load(path, allow_pickle=False) as data:
            ids = [int(value) for value in data["instance_ids"]]
            features = np.asarray(data["features"], dtype=float)
        if not wanted.issubset(set(ids)) or len(ids) != len(features):
            continue
        lookup = {instance_id: features[row] for row, instance_id in enumerate(ids)}
        ordered = np.vstack([lookup[seed] for seed in instance_seeds])
        candidates.append((path, ordered))
    return candidates


def _evaluate_rows(
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    labels: np.ndarray,
    test_instances,
    solver,
    seed: int,
) -> tuple[list[float], float, float]:
    started = time.perf_counter()
    readout = fit_reader(train_rows, labels.reshape(-1), seed)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    costs = []
    n_pairs = len(customer_pairs(test_instances[0].n_locations))
    for offset, instance in enumerate(test_instances):
        predicted = predict_labels(
            readout,
            test_rows[offset * n_pairs : (offset + 1) * n_pairs],
            instance.n_locations,
            4,
        )
        costs.append(cost_for_labels(solver, predicted, instance))
    decode_seconds = time.perf_counter() - started
    return costs, fit_seconds, decode_seconds


def shuffle_columns(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently permute each feature column, separately per split."""
    shuffled = []
    for split_index, matrix in enumerate((train, test)):
        rng = np.random.default_rng(seed * 2 + split_index)
        result = np.asarray(matrix, dtype=float).copy()
        for column in range(result.shape[1]):
            result[:, column] = rng.permutation(result[:, column])
        shuffled.append(result)
    return shuffled[0], shuffled[1]


def gaussian_like(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """I.i.d. noise with each column's training mean and standard deviation."""
    train = np.asarray(train, dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(mean, std, size=train.shape),
        rng.normal(mean, std, size=np.asarray(test).shape),
    )


def make_null_features(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build the shuffled and Gaussian null feature pairs deterministically."""
    shuffled = shuffle_columns(train, test, seed)
    gaussian = gaussian_like(train, test, seed)
    return {"shuffled": shuffled, "gaussian": gaussian}


def _build_rows(vectors, size: int) -> np.ndarray:
    return np.vstack([rows_from_vector(vector, size) for vector in vectors])


def _select_quantum_shard(
    candidates,
    checkpoints,
    train_instances,
    test_instances,
    labels,
    solver,
    seed: int,
):
    """Match cached shards to checkpoint costs; return the trained one."""
    evaluated = []
    for path, vectors in candidates:
        train_vectors = vectors[: len(train_instances)]
        test_vectors = vectors[len(train_instances) :]
        train_rows = _build_rows(train_vectors, len(train_instances[0].distances))
        test_rows = _build_rows(test_vectors, len(train_instances[0].distances))
        costs, fit_seconds, decode_seconds = _evaluate_rows(
            train_rows,
            test_rows,
            labels,
            test_instances,
            solver,
            seed,
        )
        evaluated.append(
            {
                "path": path,
                "vectors": vectors,
                "costs": costs,
                "fit_seconds": fit_seconds,
                "decode_seconds": decode_seconds,
            }
        )

    best = None
    for checkpoint in checkpoints:
        target = np.asarray(
            [checkpoint["costs"]["quantum_ring_trained"][instance.seed] for instance in test_instances]
        )
        for item in evaluated:
            score = float(np.mean(np.abs(np.asarray(item["costs"]) - target)))
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "checkpoint": checkpoint,
                    "shard": item,
                }
    if best is None:
        raise RuntimeError("no cached feature shard could be matched to a checkpoint")
    if best["score"] > MATCH_TOLERANCE:
        raise RuntimeError(
            "could not identify quantum_ring_trained features in the cache "
            f"(best mean-abs cost mismatch={best['score']:.3f} > {MATCH_TOLERANCE}); "
            "cached labels may not match the checkpoint"
        )
    return best


def run_combo(task) -> dict:
    size, outer_seed, config, needed = task
    install_no_simulation_guards()
    started = time.perf_counter()

    train, test = make_split(
        outer_seed,
        size,
        config["per_size"] + config["train_per_size"],
        config["train_per_size"],
    )
    labels = load_cached_labels(
        config["targets_dir"], size, outer_seed, train
    )
    checkpoints = load_checkpoints(size, outer_seed, test)
    if not checkpoints:
        raise RuntimeError(
            f"no usable checkpoint for size={size} seed={outer_seed}"
        )
    candidates = load_candidate_shards(
        config["cache_dir"], size, [instance.seed for instance in train + list(test)]
    )
    if not candidates:
        raise RuntimeError(
            f"no full-coverage feature shards for size={size} seed={outer_seed}"
        )
    load_seconds = time.perf_counter() - started

    benchmark_seed = outer_seed * 10_000 + size
    solver = build_random_ring_solver(size, outer_seed, 1.0)
    selection = _select_quantum_shard(
        candidates,
        checkpoints,
        train,
        test,
        labels,
        solver,
        benchmark_seed,
    )
    checkpoint = selection["checkpoint"]
    shard = selection["shard"]
    vectors = shard["vectors"]
    train_vectors = vectors[: len(train)]
    test_vectors = vectors[len(train) :]
    quantum_cost = shard["costs"]
    fit_seconds = shard["fit_seconds"]
    decode_seconds = shard["decode_seconds"]

    test_seeds = [instance.seed for instance in test]
    random_cost = [
        checkpoint["costs"]["random_assignment"][seed] for seed in test_seeds
    ]

    esn_cost = None
    esn_fit = 0.0
    esn_decode = 0.0
    if "esn" in needed:
        esn = EchoStateNetwork(size, size**2, seed=benchmark_seed)
        train_rows = _build_rows(
            [esn.features(instance.distances) for instance in train], size
        )
        test_rows = _build_rows(
            [esn.features(instance.distances) for instance in test], size
        )
        esn_cost, esn_fit, esn_decode = _evaluate_rows(
            train_rows, test_rows, labels, test, solver, benchmark_seed
        )
        checkpoint_esn = np.asarray(
            [checkpoint["costs"]["classical_esn"][seed] for seed in test_seeds]
        )
        mismatch = float(np.mean(np.abs(np.asarray(esn_cost) - checkpoint_esn)))
        if mismatch > MATCH_TOLERANCE:
            print(
                f"warning size={size} seed={outer_seed}: ESN checkpoint "
                f"mismatch {mismatch:.3f}",
                flush=True,
            )

    costs_by_variant = {"quantum": quantum_cost}
    if esn_cost is not None:
        costs_by_variant["esn"] = esn_cost
    for name in ("shuffled", "gaussian"):
        if name not in needed:
            continue
        null_train, null_test = make_null_features(
            train_vectors, test_vectors, benchmark_seed
        )[name]
        rows_train = _build_rows(null_train, size)
        rows_test = _build_rows(null_test, size)
        costs, null_fit, null_decode = _evaluate_rows(
            rows_train, rows_test, labels, test, solver, benchmark_seed
        )
        costs_by_variant[name] = costs
        fit_seconds += null_fit
        decode_seconds += null_decode
    if "constant" in needed:
        n_train_pairs = len(train) * len(customer_pairs(size))
        n_test_pairs = len(test) * len(customer_pairs(size))
        constant_train = np.zeros((n_train_pairs, 1), dtype=float)
        constant_test = np.zeros((n_test_pairs, 1), dtype=float)
        costs, constant_fit, constant_decode = _evaluate_rows(
            constant_train, constant_test, labels, test, solver, benchmark_seed
        )
        costs_by_variant["constant"] = costs
        fit_seconds += constant_fit
        decode_seconds += constant_decode
    if esn_cost is not None:
        fit_seconds += esn_fit
        decode_seconds += esn_decode

    run_rows = []
    per_instance_rows = []
    for variant in needed:
        costs = np.asarray(costs_by_variant[variant], dtype=float)
        differences = costs - np.asarray(random_cost, dtype=float)
        mean_diff, low, high = bootstrap_ci(
            differences, seed=size * 10_000 + outer_seed * 100 + VARIANTS.index(variant)
        )
        run_rows.append(
            {
                "n_locations": size,
                "outer_seed": outer_seed,
                "variant": variant,
                "n_instances": len(costs),
                "mean_cost": float(np.mean(costs)),
                "diff_random_assignment": mean_diff,
                "ci_low_random_assignment": low,
                "ci_high_random_assignment": high,
                "feature_source": (
                    os.path.basename(shard["path"])
                    if variant in ("quantum", "shuffled", "gaussian", "constant")
                    else "esn-v2"
                ),
            }
        )
        for instance, cost in zip(test, costs):
            per_instance_rows.append(
                {
                    "n_locations": size,
                    "outer_seed": outer_seed,
                    "instance_seed": instance.seed,
                    "variant": variant,
                    "cost": float(cost),
                }
            )

    # The random reference is stored once per combination for paired summaries.
    for instance, cost in zip(test, random_cost):
        per_instance_rows.append(
            {
                "n_locations": size,
                "outer_seed": outer_seed,
                "instance_seed": instance.seed,
                "variant": "random_assignment",
                "cost": float(cost),
            }
        )

    print(
        f"size={size} seed={outer_seed} load={load_seconds:.2f}s "
        f"fit={fit_seconds:.2f}s decode={decode_seconds:.2f}s "
        f"shard={os.path.basename(shard['path'])} match={selection['score']:.4f}",
        flush=True,
    )
    return {"run_rows": run_rows, "per_instance_rows": per_instance_rows}


def _write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(per_instance_rows: Sequence[dict], sizes: Sequence[int]) -> list[dict]:
    grouped: dict[tuple[int, str], dict[tuple[int, int], float]] = {}
    for row in per_instance_rows:
        key = (int(row["n_locations"]), row["variant"])
        grouped.setdefault(key, {})[
            (int(row["outer_seed"]), int(row["instance_seed"]))
        ] = float(row["cost"])

    summary = []
    for size in sizes:
        for variant in REFERENCES:
            costs = grouped.get((size, variant))
            if not costs:
                continue
            row = {
                "n_locations": size,
                "variant": variant,
                "n_instances": len(costs),
                "mean_cost": float(np.mean(list(costs.values()))),
            }
            for reference in REFERENCES:
                reference_costs = grouped.get((size, reference))
                if not reference_costs:
                    continue
                common = sorted(set(costs) & set(reference_costs))
                differences = np.asarray(
                    [costs[key] - reference_costs[key] for key in common]
                )
                mean_diff, low, high = bootstrap_ci(
                    differences,
                    seed=size * 100_000
                    + REFERENCES.index(variant) * 1_000
                    + REFERENCES.index(reference),
                )
                row[f"diff_{reference}"] = mean_diff
                row[f"ci_low_{reference}"] = low
                row[f"ci_high_{reference}"] = high
            summary.append(row)
    return summary


def _write_markdown(path: str, summary: Sequence[dict]) -> None:
    lines = [
        "# Null controls",
        "",
        "Lower cost is better. Values are paired mean differences (row variant "
        "minus reference) with 95% bootstrap CIs.",
        "",
        "| Size | Variant | Mean cost | vs random | vs quantum | vs shuffled | vs gaussian | vs constant | vs ESN |",
        "|---:|---|---:|---|---|---|---|---|---|",
    ]
    for row in summary:
        cells = [str(row["n_locations"]), row["variant"], f"{row['mean_cost']:.4f}"]
        for reference in REFERENCES:
            key = f"diff_{reference}"
            if key not in row:
                cells.append("")
                continue
            cells.append(
                f"{row[key]:+.4f} [{row[f'ci_low_{reference}']:+.4f}, "
                f"{row[f'ci_high_{reference}']:+.4f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 10, 12, 16])
    parser.add_argument(
        "--outer-seeds",
        "--seeds",
        dest="outer_seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="worker processes; use 1 for an in-process run (needed by cProfile)",
    )
    parser.add_argument("--per-size", type=int, default=250)
    parser.add_argument("--train-per-size", type=int, default=200)
    parser.add_argument("--cache-dir", default="results/feature_cache")
    parser.add_argument(
        "--targets-dir", default="results/shot_noise/targets"
    )
    parser.add_argument("--output-dir", default="results/null_control")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    runs_path = os.path.join(args.output_dir, "null_control_runs.csv")
    per_instance_path = os.path.join(
        args.output_dir, "null_control_per_instance.csv"
    )
    summary_path = os.path.join(args.output_dir, "null_control_summary.csv")

    run_rows = {
        (int(row["n_locations"]), int(row["outer_seed"]), row["variant"]): row
        for row in _read_csv(runs_path)
    }
    per_instance_rows = {
        (
            int(row["n_locations"]),
            int(row["outer_seed"]),
            int(row["instance_seed"]),
            row["variant"],
        ): row
        for row in _read_csv(per_instance_path)
    }

    tasks = []
    for size in args.sizes:
        for outer_seed in args.outer_seeds:
            needed = [
                variant
                for variant in VARIANTS
                if (size, outer_seed, variant) not in run_rows
            ]
            if not needed:
                continue
            tasks.append(
                (
                    size,
                    outer_seed,
                    {
                        "per_size": args.per_size,
                        "train_per_size": args.train_per_size,
                        "cache_dir": args.cache_dir,
                        "targets_dir": args.targets_dir,
                    },
                    needed,
                )
            )

    def consume(result: dict) -> None:
        for row in result["run_rows"]:
            key = (
                int(row["n_locations"]),
                int(row["outer_seed"]),
                row["variant"],
            )
            run_rows[key] = row
        for row in result["per_instance_rows"]:
            key = (
                int(row["n_locations"]),
                int(row["outer_seed"]),
                int(row["instance_seed"]),
                row["variant"],
            )
            per_instance_rows[key] = row
        _write_csv(per_instance_path, per_instance_rows.values())
        _write_csv(runs_path, run_rows.values())

    if tasks:
        if args.workers <= 1:
            for task in tasks:
                consume(run_combo(task))
        else:
            with mp.Pool(processes=args.workers) as pool:
                for result in pool.imap_unordered(run_combo, tasks):
                    consume(result)
    else:
        print("all requested rows already present", flush=True)

    available_sizes = sorted(
        {int(row["n_locations"]) for row in per_instance_rows.values()}
        | set(args.sizes)
    )
    summary = _summary(list(per_instance_rows.values()), available_sizes)
    _write_csv(summary_path, summary)
    _write_markdown(os.path.join(args.output_dir, "null_control.md"), summary)
    print(f"wrote {runs_path}", flush=True)
    print(f"wrote {per_instance_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
