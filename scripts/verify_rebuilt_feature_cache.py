"""Compare rebuilt quantum/ESN feature shards with fair-benchmark checkpoints."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.fast_sim import FeatureCache  # noqa: E402
from scripts.benchmark_fair import EchoStateNetwork, _collect_esn_features, make_split  # noqa: E402
from scripts.null_control import (  # noqa: E402
    _build_rows,
    _evaluate_rows,
    _select_quantum_shard,
    install_no_simulation_guards,
    load_cached_labels,
    load_candidate_shards,
    load_checkpoints,
)
from scripts.pairwise_diagnostics import build_random_ring_solver  # noqa: E402


def main() -> None:
    install_no_simulation_guards()
    cache_root = "results/feature_cache"
    feature_cache = FeatureCache(cache_root)
    for size in (10,):
        totals = {"quantum": [], "esn": [], "quantum_ref": [], "esn_ref": []}
        for outer_seed in range(5):
            train, test = make_split(outer_seed, size, 450, 200)
            targets = load_cached_labels("results/shot_noise/targets", size, outer_seed, train)
            checkpoints = load_checkpoints(size, outer_seed, test)
            if not checkpoints:
                raise RuntimeError(f"missing checkpoint N={size}, seed={outer_seed}")
            candidates = load_candidate_shards(cache_root, size, [x.seed for x in train + test])
            solver = build_random_ring_solver(size, outer_seed, 1.0)
            benchmark_seed = outer_seed * 10_000 + size
            selected = _select_quantum_shard(candidates, checkpoints, train, test, targets, solver, benchmark_seed)
            vectors = selected["shard"]["vectors"]
            q_rows = _build_rows(vectors[:len(train)], size), _build_rows(vectors[len(train):], size)
            q_cost, _, _ = _evaluate_rows(q_rows[0], q_rows[1], targets, test, solver, benchmark_seed)

            esn = EchoStateNetwork(size, size**2, seed=benchmark_seed)
            esn_vectors, _ = _collect_esn_features(esn, train + test, timing=False, cache=feature_cache, seed=benchmark_seed)
            e_rows = _build_rows(esn_vectors[:len(train)], size), _build_rows(esn_vectors[len(train):], size)
            e_cost, _, _ = _evaluate_rows(e_rows[0], e_rows[1], targets, test, solver, benchmark_seed)
            checkpoint = selected["checkpoint"]
            test_ids = [instance.seed for instance in test]
            q_ref = np.asarray([checkpoint["costs"]["quantum_ring_trained"][key] for key in test_ids])
            e_ref = np.asarray([checkpoint["costs"]["classical_esn"][key] for key in test_ids])
            q_mean, e_mean = float(np.mean(q_cost)), float(np.mean(e_cost))
            q_ref_mean, e_ref_mean = float(np.mean(q_ref)), float(np.mean(e_ref))
            totals["quantum"].extend(q_cost); totals["esn"].extend(e_cost)
            totals["quantum_ref"].extend(q_ref); totals["esn_ref"].extend(e_ref)
            print(f"N={size} seed={outer_seed}: QRC {q_mean:.4f} (checkpoint {q_ref_mean:.4f}; Δ={q_mean-q_ref_mean:+.4f}), ESN {e_mean:.4f} (checkpoint {e_ref_mean:.4f}; Δ={e_mean-e_ref_mean:+.4f}), feature shard={os.path.basename(selected['shard']['path'])}", flush=True)
        for method, ref in (("quantum", "quantum_ref"), ("esn", "esn_ref")):
            got, expected = float(np.mean(totals[method])), float(np.mean(totals[ref]))
            print(f"N={size} aggregate {method}: {got:.6f} vs checkpoint {expected:.6f}; Δ={got-expected:+.6f}", flush=True)
            if size == 10 and abs(got - ({"quantum": 6.3779, "esn": 6.4161}[method])) > 0.01:
                raise SystemExit(f"D5 stop: N=10 {method} mean {got:.5f} failed ±0.01 target")


if __name__ == "__main__":
    main()
