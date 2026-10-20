"""Decoder ceiling diagnostic.

Solve OR-Tools for a subset of test instances, convert its routes into
per-customer vehicle labels, and decode those same labels again.  If the
re-decoded cost is close to the OR-Tools cost, the decoder is not the
bottleneck; a remaining gap must come from the learned co-assignment.

The script also records how many routes OR-Tools actually uses.  Routes are
allowed to be empty, so this checks whether forcing exactly ``n_vehicles``
clusters would discard a valid solution.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark_fair import (  # noqa: E402
    _decode,
    make_split,
    route_cost,
    solve_ortools,
)
from scripts.pairwise_diagnostics import build_random_ring_solver  # noqa: E402


def labels_from_routes(routes, n_locations: int) -> np.ndarray:
    labels = np.full(n_locations - 1, -1, dtype=int)
    for vehicle, route in enumerate(routes):
        for node in route:
            if node != 0:
                labels[node - 1] = vehicle
    if np.any(labels < 0):
        raise ValueError("OR-Tools route omitted a customer")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 12])
    parser.add_argument("--outer-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--per-size", type=int, default=250)
    parser.add_argument("--train-per-size", type=int, default=200)
    parser.add_argument("--ortools-seconds", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/decoder_diagnostic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows: list[dict] = []

    for size in args.sizes:
        for outer_seed in args.outer_seeds:
            _, test = make_split(
                outer_seed,
                size,
                args.per_size + args.train_per_size,
                args.train_per_size,
            )
            solver = build_random_ring_solver(size, outer_seed, 1.0)
            for index, instance in enumerate(test[: args.instances]):
                routes, _, _ = solve_ortools(
                    instance.distances,
                    4,
                    args.ortools_seconds,
                    seed=outer_seed * 1000 + args.train_per_size + index,
                    demands=instance.demands,
                    capacity=instance.capacity,
                )
                nonempty = [route for route in routes if len(route) > 2]
                labels = labels_from_routes(routes, size)
                oracle_cost = route_cost(routes, instance.distances)
                decoded = _decode(solver, labels, instance, 4, "nn2opt")
                decoded_cost = route_cost(decoded, instance.distances)
                indexed = _decode(solver, labels, instance, 4, "index")
                indexed_cost = route_cost(indexed, instance.distances)
                rows.append(
                    {
                        "n_locations": size,
                        "outer_seed": outer_seed,
                        "instance_seed": instance.seed,
                        "ortools_routes": len(nonempty),
                        "ortools_cost": oracle_cost,
                        "decoded_cost": decoded_cost,
                        "decoded_index_cost": indexed_cost,
                        "decoded_gap": (decoded_cost - oracle_cost) / oracle_cost,
                        "index_gap": (indexed_cost - oracle_cost) / oracle_cost,
                    }
                )

    path = os.path.join(args.output_dir, "decoder_ceiling.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for size in args.sizes:
        subset = [row for row in rows if row["n_locations"] == size]
        route_counts = sorted({row["ortools_routes"] for row in subset})
        print(
            size,
            {
                "runs": len(subset),
                "ortools_route_counts": route_counts,
                "mean_decoded_gap": float(
                    np.mean([row["decoded_gap"] for row in subset])
                ),
                "max_decoded_gap": float(
                    np.max([row["decoded_gap"] for row in subset])
                ),
                "mean_index_gap": float(
                    np.mean([row["index_gap"] for row in subset])
                ),
            },
        )


if __name__ == "__main__":
    main()
