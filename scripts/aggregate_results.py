"""Aggregate checkpointed fair-benchmark runs into one reproducible table."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark_fair import bootstrap_ci  # noqa: E402

DEFAULT_SOURCES = {
    4: "results/pairwise_4_12",
    8: "results/pairwise_8_16",
    12: "results/pairwise_4_12",
    16: "results/pairwise_8_16",
}

COMPARISONS = (
    ("distance_pair", "random_assignment"),
    ("distance_pair", "sweep_assignment"),
    ("distance_pair", "kmeans_distance"),
    ("kmeans_distance", "random_assignment"),
    ("sweep_assignment", "random_assignment"),
    ("classical_esn", "distance_pair"),
    ("quantum_ring_trained", "distance_pair"),
    ("quantum_shallow_trained", "distance_pair"),
    ("quantum_ring_random", "random_assignment"),
    ("quantum_ring_trained", "random_assignment"),
    ("quantum_ring_trained", "classical_esn"),
    ("quantum_shallow_trained", "classical_esn"),
    ("quantum_ring_trained", "quantum_ring_random"),
)


def load_checkpoints(directory: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "checkpoints", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.extend(payload.get("rows", []))
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        key = (
            int(row["n_locations"]),
            int(row["outer_seed"]),
            row["method"],
            int(row["instance_seed"]),
        )
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    summaries = []
    sizes = sorted({int(row["n_locations"]) for row in rows})
    methods = sorted({row["method"] for row in rows})
    for size in sizes:
        for method in methods:
            method_rows = [
                row
                for row in rows
                if int(row["n_locations"]) == size and row["method"] == method
            ]
            if not method_rows:
                continue
            costs = np.array([float(row["cost"]) for row in method_rows])
            gaps = np.array([float(row["gap"]) for row in method_rows])
            times = np.array([float(row["time"]) for row in method_rows])
            feature_abs = np.array(
                [float(row["feature_mean_abs"]) for row in method_rows]
            )
            feature_var = np.array(
                [float(row["feature_variance"]) for row in method_rows]
            )
            summaries.append(
                {
                    "n_locations": size,
                    "method": method,
                    "n_runs": len(method_rows),
                    "n_seeds": len({int(row["outer_seed"]) for row in method_rows}),
                    "mean_cost": float(costs.mean()),
                    "mean_gap": float(gaps.mean()),
                    "median_time_ms": float(np.median(times) * 1000.0),
                    "feature_mean_abs": float(np.nanmean(feature_abs))
                    if np.any(np.isfinite(feature_abs))
                    else float("nan"),
                    "feature_variance": float(np.nanmean(feature_var))
                    if np.any(np.isfinite(feature_var))
                    else float("nan"),
                }
            )
    return summaries


def comparisons(rows: list[dict]) -> list[dict]:
    results = []
    sizes = sorted({int(row["n_locations"]) for row in rows})
    for size in sizes:
        lookup: dict[str, dict[int, float]] = defaultdict(dict)
        for row in rows:
            if int(row["n_locations"]) == size:
                lookup[row["method"]][int(row["instance_seed"])] = float(
                    row["cost"]
                )
        for left, right in COMPARISONS:
            common = sorted(set(lookup[left]) & set(lookup[right]))
            if not common:
                continue
            differences = np.array(
                [lookup[left][key] - lookup[right][key] for key in common]
            )
            mean, low, high = bootstrap_ci(differences, seed=size)
            results.append(
                {
                    "n_locations": size,
                    "comparison": f"{left} - {right}",
                    "mean_diff": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "excludes_zero": "yes"
                    if low > 0.0 or high < 0.0
                    else "no",
                    "n_pairs": len(common),
                }
            )
    return results


def decoder_diagnostics(rows: list[dict]) -> list[dict]:
    diagnostics = []
    sizes = sorted({int(row["n_locations"]) for row in rows})
    methods = sorted({row["method"] for row in rows})
    for size in sizes:
        for method in methods:
            method_rows = [
                row
                for row in rows
                if int(row["n_locations"]) == size and row["method"] == method
            ]
            if not method_rows:
                continue
            costs = np.array([float(row["cost"]) for row in method_rows])
            index_costs = np.array([float(row["cost_index"]) for row in method_rows])
            improvement = (index_costs - costs) / np.maximum(index_costs, 1e-12)
            diagnostics.append(
                {
                    "n_locations": size,
                    "method": method,
                    "mean_cost_nn2opt": float(costs.mean()),
                    "mean_cost_index": float(index_costs.mean()),
                    "decoder_improvement_pct": float(100.0 * improvement.mean()),
                    "n_runs": len(method_rows),
                }
            )
    return diagnostics


def gate_duplicate_check(aggregate_rows: list[dict], gate_dir: str) -> list[dict]:
    gate_rows = deduplicate(load_checkpoints(gate_dir))
    pairwise_8 = {
        (int(row["outer_seed"]), row["method"], int(row["instance_seed"])): float(
            row["cost"]
        )
        for row in aggregate_rows
        if int(row["n_locations"]) == 8
    }
    differences = []
    for row in gate_rows:
        key = (
            int(row["outer_seed"]),
            row["method"],
            int(row["instance_seed"]),
        )
        if key in pairwise_8:
            differences.append(abs(pairwise_8[key] - float(row["cost"])))
    return [
        {
            "n_common_gate_runs": len(differences),
            "max_abs_cost_difference": float(max(differences)) if differences else float("nan"),
            "mean_abs_cost_difference": float(np.mean(differences)) if differences else float("nan"),
        }
    ]


def write_markdown(path: str, summary: list[dict], comps: list[dict], diagnostics: list[dict], notes: list[str]) -> None:
    lines = ["# Aggregated label-free fair benchmark", ""]
    lines.extend(notes)
    lines.extend(["", "## Summary", ""])
    lines.append(
        "| Size | Method | Seeds | Runs | Mean cost | Mean gap | Median time (ms) |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {n_locations} | {method} | {n_seeds} | {n_runs} | {mean_cost:.4f} | "
            "{mean_gap:.2%} | {median_time_ms:.3f} |".format(**row)
        )
    lines.extend(["", "## Paired differences (bootstrap 95% CI)", ""])
    lines.append("| Size | Comparison | Mean difference | 95% CI | Excludes 0 | N |")
    lines.append("|---:|---|---:|---|---|---:|")
    for row in comps:
        lines.append(
            "| {n_locations} | {comparison} | {mean_diff:.4f} | "
            "[{ci_low:.4f}, {ci_high:.4f}] | {excludes_zero} | {n_pairs} |".format(
                **row
            )
        )
    lines.extend(["", "## Decoder diagnostic (nn2opt vs index ordering)", ""])
    lines.append("| Size | Method | Cost nn2opt | Cost index | Decoder gain |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in diagnostics:
        lines.append(
            "| {n_locations} | {method} | {mean_cost_nn2opt:.4f} | "
            "{mean_cost_index:.4f} | {decoder_improvement_pct:.2f}% |".format(
                **row
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/aggregate")
    args = parser.parse_args()

    rows: list[dict] = []
    for size, directory in DEFAULT_SOURCES.items():
        size_rows = [
            row
            for row in load_checkpoints(directory)
            if int(row["n_locations"]) == size
        ]
        rows.extend(size_rows)
    rows = deduplicate(rows)

    summary = summarise(rows)
    comps = comparisons(rows)
    diagnostics = decoder_diagnostics(rows)
    duplicate = gate_duplicate_check(rows, "results/gate")

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "fair_benchmark_runs.csv"), rows)
    write_csv(os.path.join(args.output_dir, "fair_benchmark_summary.csv"), summary)
    write_csv(
        os.path.join(args.output_dir, "fair_benchmark_comparisons.csv"), comps
    )
    write_csv(
        os.path.join(args.output_dir, "fair_benchmark_decoder.csv"), diagnostics
    )
    write_csv(
        os.path.join(args.output_dir, "gate_duplicate_check.csv"), duplicate
    )

    seed_counts = {
        size: len(
            {
                int(row["outer_seed"])
                for row in rows
                if int(row["n_locations"]) == size
            }
        )
        for size in sorted({int(row["n_locations"]) for row in rows})
    }
    notes = [
        "N=4 is degenerate (three customers, four vehicles) and is excluded "
        "from any claim.",
        "N=8 uses the full-method-set run; the gate run is retained as a "
        "duplicate check.",
        "Seeds available per size: "
        + ", ".join(f"N={size}: {count}" for size, count in seed_counts.items())
        + ". N=16 is provisional until seeds 1-4 are run.",
        "Locations/gaps are relative to the OR-Tools 1 s reference.",
    ]
    write_markdown(
        os.path.join(args.output_dir, "fair_benchmark.md"),
        summary,
        comps,
        diagnostics,
        notes,
    )
    print(json.dumps({"seed_counts": seed_counts, "duplicate": duplicate}, indent=2))


if __name__ == "__main__":
    main()
