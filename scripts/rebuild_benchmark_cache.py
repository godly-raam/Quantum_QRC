"""Regenerate benchmark_fair quantum/ESN feature shards for requested sizes."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.fast_sim import FeatureCache  # noqa: E402
from modules.quantum_reservoir_vrp import ReservoirVRPSolver  # noqa: E402
from modules.reservoir_trainer import build_parameterized_reservoir  # noqa: E402
from scripts.benchmark_fair import (  # noqa: E402
    EchoStateNetwork,
    _collect_esn_features,
    _collect_quantum_features,
    make_split,
    train_quantum_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-cache", default="results/feature_cache")
    args = parser.parse_args()

    cache = FeatureCache(args.output_cache)
    for size in args.sizes:
        for outer_seed in args.seeds:
            train, test = make_split(outer_seed, size, 450, 200)
            instances = train + test
            benchmark_seed = outer_seed * 10_000 + size
            trainer_seed = 1000 + outer_seed
            print(f"N={size} seed={outer_seed}: training ring reservoir (seed={trainer_seed})", flush=True)
            trained = train_quantum_parameters(
                size,
                build_parameterized_reservoir,
                trainer_seed,
                angle_scale=1.0,
                use_depot_diagonal=False,
                num_samples=20,
                maxiter=100,
            )
            solver = ReservoirVRPSolver(
                n_reservoir_qubits=size,
                trained_params=trained,
                legacy_decoder=False,
                ordering="nn2opt",
                angle_scale=1.0,
                use_depot_diagonal=False,
                random_seed=outer_seed,
                reservoir_layers=2,
                reservoir_builder=build_parameterized_reservoir,
            )
            _collect_quantum_features(
                solver.reservoir,
                instances,
                n_vehicles=4,
                outer_seed=benchmark_seed,
                timing=False,
                cache=cache,
            )
            esn = EchoStateNetwork(size, size**2, seed=benchmark_seed)
            _collect_esn_features(
                esn,
                instances,
                timing=False,
                cache=cache,
                seed=benchmark_seed,
            )
            print(f"N={size} seed={outer_seed}: cached {len(instances)} instances", flush=True)


if __name__ == "__main__":
    main()
