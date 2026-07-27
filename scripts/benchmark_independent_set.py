"""Run QRC maximum-independent-set benchmarks on QOBLIB instances.

The output CSV inherits its header verbatim from IBM's QOBLIB QAOA submission so
that the two result rows can be compared without schema translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

from typing import Any, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from modules.independent_set_qrc import (  # noqa: E402 # pylint: disable=wrong-import-position,import-error
    parse_dimacs_graph,
    parse_optimal_solution,
    sample_repaired_independent_sets,
)

DATA_ROOT = PROJECT_ROOT / "data" / "QOBLIB" / "07-independentset"
EGGER_SUBMISSION = DATA_ROOT / "submissions" / "20250115_QAOA_Egger"
HEADER_SOURCE = (
    EGGER_SUBMISSION
    / "mammalia-kangaroo-interactions"
    / "mammalia-kangaroo-interactions_summary.csv"
)


class RunResult(TypedDict):
    """Dictionary schema for individual benchmark run metrics."""

    run: int
    best_objective: int
    feasible_samples: int
    optimal_samples: int
    samples: int
    runtime_seconds: float


def _header() -> list[str]:
    with HEADER_SOURCE.open(newline="", encoding="utf-8") as source:
        return next(csv.reader(source))


def _benchmark_instance(  # pylint: disable=too-many-locals
    instance_name: str, runs: int, shots: int, seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    graph = parse_dimacs_graph(DATA_ROOT / "instances" / f"{instance_name}.gph")
    optimum_vertices = parse_optimal_solution(DATA_ROOT / "solutions" / f"{instance_name}.opt.sol")
    optimum = len(optimum_vertices)

    run_results: list[RunResult] = []
    total_start = time.perf_counter()
    for run_index in range(runs):
        run_start = time.perf_counter()
        samples = sample_repaired_independent_sets(
            graph, shots=shots, seed=seed + run_index
        )
        objectives = [len(sample) for sample in samples]
        best_obj = max(objectives)
        run_results.append(
            {
                "run": run_index + 1,
                "best_objective": best_obj,
                "feasible_samples": len(samples),
                "optimal_samples": sum(value == optimum for value in objectives),
                "samples": len(samples),
                "runtime_seconds": time.perf_counter() - run_start,
            }
        )

    total_runtime = time.perf_counter() - total_start
    best_objective = max(run["best_objective"] for run in run_results)
    successful_runs = sum(run["best_objective"] == optimum for run in run_results)
    feasible_runs = sum(run["feasible_samples"] > 0 for run in run_results)
    total_samples = sum(run["samples"] for run in run_results)
    optimal_samples = sum(run["optimal_samples"] for run in run_results)
    optimum_rate = optimal_samples / total_samples if total_samples else 0.0
    time_to_solution = next(
        (
            run["runtime_seconds"]
            for run in run_results
            if run["best_objective"] == optimum
        ),
        None,
    )

    workflow = (
        "1) Construct QOBLIB maximum-independent-set QUBO and constant-free Ising cost layer. "
        "2) Apply the fixed QuantumReservoir circuit family instead of a QAOA mixer. "
        "3) Sample with Aer matrix-product-state simulation. "
        "4) Greedily repair every violated edge and score repaired independent sets."
    )
    row = {
        "Problem": instance_name,
        "Submitter": "Entangle Minds Team",
        "Affiliation": "Q-Fleet",
        "Date": date.today().isoformat(),
        "Reference": "This repository; scripts/benchmark_independent_set.py",
        "Best Objective Value": str(best_objective),
        "Optimality Bound": str(optimum),
        "Modeling Approach": "QUBO converted to Ising; QuantumReservoir ansatz",
        "# Decision Variables": str(graph.num_vertices),
        "# Binary Variables": str(graph.num_vertices),
        "# Integer Variables": "N/A",
        "# Continuous Variables": "N/A",
        "# Non-Zero Coefficients": str(graph.num_vertices + len(graph.edges)),
        "Coefficients Type": "Continuous",
        "Coefficients Range": "<-1,2>",
        "Workflow": workflow,
        "Algorithm Type": "Stochastic",
        "Paradigm": "Quantum Simulator",
        "# Runs": str(runs),
        "# Feasible Runs": str(feasible_runs),
        "# Successful Runs": str(successful_runs),
        "Success Threshold": "0",
        "Hardware Specifications": (
            "Local AerSimulator (matrix_product_state); see runtime environment"
        ),
        "Total Runtime": f"{total_runtime:.6f}",
        "Time to Solution": "N/A" if time_to_solution is None else f"{time_to_solution:.6f}",
        "CPU Runtime": f"{total_runtime:.6f}",
        "GPU Runtime": "N/A",
        "QPU Runtime": "N/A",
        "Other HW Runtime": "N/A",
        "Remarks": (
            f"All samples were greedily repaired. Optimal repaired-sample rate: {optimum_rate:.2%} "
            f"({optimal_samples}/{total_samples}); this is a simulator result, not QPU performance."
        ),
    }
    summary = {
        "instance": instance_name,
        "vertices": graph.num_vertices,
        "edges": len(graph.edges),
        "optimum_from_solution_file": optimum,
        "best_objective": best_objective,
        "successful_runs": successful_runs,
        "runs": runs,
        "optimal_repaired_sample_rate": optimum_rate,
        "optimal_repaired_samples": optimal_samples,
        "repaired_samples": total_samples,
        "total_runtime_seconds": total_runtime,
        "run_results": run_results,
    }
    return row, summary


def main() -> None:
    """Run benchmark experiments on QOBLIB maximum-independent-set instances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmark_results" / "independent_set_qrc_summary.csv",
    )
    args = parser.parse_args()
    if args.runs != 5:
        parser.error("Use exactly five runs to match the QOBLIB Egger submission methodology")

    rows: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    for instance_name in ("mammalia-kangaroo-interactions", "aves-sparrow-social"):
        row, summary = _benchmark_instance(instance_name, args.runs, args.shots, args.seed)
        rows.append(row)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=_header())
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote QOBLIB-aligned CSV: {args.output}")


if __name__ == "__main__":
    main()
