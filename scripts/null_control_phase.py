"""Cache-only D0/D2/D3/D4 diagnostics for one requested benchmark size."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.readout import customer_pairs, pair_targets_from_routes  # noqa: E402
from scripts.benchmark_fair import _decode, bootstrap_ci, make_split, random_assignment, route_cost  # noqa: E402
from scripts.null_control import (  # noqa: E402
    _build_rows,
    _evaluate_rows,
    _select_quantum_shard,
    install_no_simulation_guards,
    load_cached_labels,
    load_candidate_shards,
    load_checkpoints,
    make_null_features,
)
from scripts.pairwise_diagnostics import build_random_ring_solver, fit_reader, predict_labels  # noqa: E402
from scripts.shot_noise_ablation import add_shot_noise, evaluate_feature_rows, load_targets  # noqa: E402


def balanced_assignment(n_customers: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.empty(n_customers, dtype=int)
    order = rng.permutation(n_customers)
    for label, group in enumerate(np.array_split(order, 4)):
        labels[group] = label
    return labels


def cost_and_routes(labels: np.ndarray, instance, solver) -> tuple[float, int, float]:
    routes = _decode(solver, labels, instance, 4, "nn2opt")
    nonempty = sum(any(int(stop) != 0 for stop in route) for route in routes)
    counts = np.bincount(labels, minlength=4)
    return route_cost(routes, instance.distances), nonempty, float(np.std(counts))


def paired_ci(differences: np.ndarray, seed: int) -> str:
    mean, low, high = bootstrap_ci(differences, seed=seed)
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--cache-dir", default="results/feature_cache")
    parser.add_argument("--targets-dir", default="results/shot_noise/targets")
    parser.add_argument("--output", default="results/null_control/diagnostic.md")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.size not in (10, 12):
        parser.error("this diagnostic is configured for N=10 or N=12")
    install_no_simulation_guards()

    size = args.size
    seeds = (0, 1, 2, 3, 4)
    variants = ("quantum_ring_trained", "constant", "shuffled", "gaussian", "S=1", "S=100000")
    costs = {name: [] for name in variants}
    metrics = {name: {key: [] for key in ("nonempty", "cluster_std", "lambda", "output_var")} for name in variants}
    random_reference, balanced_reference = [], []
    random_nonempty, random_cluster_std = [], []
    balanced_nonempty, balanced_cluster_std = [], []
    quantum_and_shot_equivalence = []
    constant_partitions = []
    seed_lines = []

    for outer_seed in seeds:
        train, test = make_split(outer_seed, size, 450, 200)
        routes = load_targets(args.targets_dir, size, outer_seed)
        assert routes == load_targets(args.targets_dir, size, outer_seed)
        cached_targets = load_cached_labels(args.targets_dir, size, outer_seed, train)
        targets = np.vstack([pair_targets_from_routes(route, size) for route in routes])
        assert np.array_equal(cached_targets, targets)
        checkpoints = load_checkpoints(size, outer_seed, test)
        if not checkpoints:
            raise RuntimeError(f"missing checkpoint N={size}, seed={outer_seed}")
        candidates = load_candidate_shards(args.cache_dir, size, [x.seed for x in train + test])
        if not candidates:
            raise RuntimeError(f"missing complete feature shard N={size}, seed={outer_seed}")
        solver = build_random_ring_solver(size, outer_seed, 1.0)
        benchmark_seed = outer_seed * 10_000 + size
        selection = _select_quantum_shard(candidates, checkpoints, train, test, targets, solver, benchmark_seed)
        vectors = selection["shard"]["vectors"]
        train_vectors, test_vectors = vectors[:len(train)], vectors[len(train):]
        train_rows, test_rows = _build_rows(train_vectors, size), _build_rows(test_vectors, size)
        n_pairs = len(customer_pairs(size))
        train_split = [train_rows[i*n_pairs:(i+1)*n_pairs] for i in range(len(train))]
        test_split = [test_rows[i*n_pairs:(i+1)*n_pairs] for i in range(len(test))]
        no_noise_a = _evaluate_rows(train_rows, test_rows, targets, test, solver, benchmark_seed)[0]
        no_noise_b = evaluate_feature_rows(train_split, test_split, targets, test, solver, benchmark_seed)
        assert np.allclose(no_noise_a, no_noise_b, atol=1e-10, rtol=0)
        quantum_and_shot_equivalence.extend(np.asarray(no_noise_a) - np.asarray(no_noise_b))

        # Expected random and uniformly balanced partitions, 100 draws per test instance.
        rand_values = {x.seed: [] for x in test}
        bal_values = {x.seed: [] for x in test}
        for draw in range(100):
            rng = np.random.default_rng(7_000_000 + size * 100_000 + outer_seed * 1_000 + draw)
            for instance in test:
                labels = random_assignment(instance, 4, rng)
                rand_values[instance.seed].append(cost_and_routes(labels, instance, solver))
                labels_b = balanced_assignment(size - 1, rng)
                bal_values[instance.seed].append(cost_and_routes(labels_b, instance, solver))
        rand_means = {key: np.mean(value, axis=0) for key, value in rand_values.items()}
        bal_means = {key: np.mean(value, axis=0) for key, value in bal_values.items()}
        random_reference.extend([rand_means[x.seed][0] for x in test])
        balanced_reference.extend([bal_means[x.seed][0] for x in test])
        random_nonempty.extend([rand_means[x.seed][1] for x in test])
        random_cluster_std.extend([rand_means[x.seed][2] for x in test])
        balanced_nonempty.extend([bal_means[x.seed][1] for x in test])
        balanced_cluster_std.extend([bal_means[x.seed][2] for x in test])

        nulls = make_null_features(train_vectors, test_vectors, benchmark_seed)
        feature_sets = {
            "quantum_ring_trained": (train_rows, test_rows),
            "constant": (np.zeros((len(train) * n_pairs, 1)), np.zeros((len(test) * n_pairs, 1))),
            "shuffled": (_build_rows(nulls["shuffled"][0], size), _build_rows(nulls["shuffled"][1], size)),
            "gaussian": (_build_rows(nulls["gaussian"][0], size), _build_rows(nulls["gaussian"][1], size)),
        }
        for name, (tr, te) in feature_sets.items():
            readout = fit_reader(tr, targets.reshape(-1), benchmark_seed)
            scores = np.asarray(readout.predict(te)).reshape(-1)
            predicted_costs = _evaluate_rows(tr, te, targets, test, solver, benchmark_seed)[0]
            costs[name].extend(predicted_costs)
            metrics[name]["lambda"].append(float(readout.best_alpha))
            metrics[name]["output_var"].append(float(np.var(scores)))
            for i, instance in enumerate(test):
                rows = te[i*n_pairs:(i+1)*n_pairs]
                labels = predict_labels(readout, rows, size, 4)
                _, occupied, spread = cost_and_routes(labels, instance, solver)
                metrics[name]["nonempty"].append(occupied)
                metrics[name]["cluster_std"].append(spread)
            if name == "constant" and outer_seed == 0:
                for i, instance in enumerate(test[:3]):
                    labels = predict_labels(readout, te[i*n_pairs:(i+1)*n_pairs], size, 4)
                    partition = [np.flatnonzero(labels == label).tolist() for label in np.unique(labels)]
                    constant_partitions.append((instance.seed, partition, readout.best_alpha))

        for shots, name in ((1, "S=1"), (100_000, "S=100000")):
            rep_costs, rep_routes, rep_spread, rep_lambdas, rep_vars = [], [], [], [], []
            for rep in range(5):
                rng = np.random.default_rng(size * 1_000_000 + outer_seed * 10_000 + shots + rep * 97)
                noisy_train = np.asarray([add_shot_noise(v, shots, rng) for v in train_vectors])
                noisy_test = np.asarray([add_shot_noise(v, shots, rng) for v in test_vectors])
                tr, te = _build_rows(noisy_train, size), _build_rows(noisy_test, size)
                readout = fit_reader(tr, targets.reshape(-1), benchmark_seed)
                scores = np.asarray(readout.predict(te)).reshape(-1)
                rep_costs.append(_evaluate_rows(tr, te, targets, test, solver, benchmark_seed)[0])
                rep_lambdas.append(float(readout.best_alpha))
                rep_vars.append(float(np.var(scores)))
                rcount, spread = [], []
                for i, instance in enumerate(test):
                    labels = predict_labels(readout, te[i*n_pairs:(i+1)*n_pairs], size, 4)
                    _, occupied, std = cost_and_routes(labels, instance, solver)
                    rcount.append(occupied); spread.append(std)
                rep_routes.append(rcount); rep_spread.append(spread)
            costs[name].extend(np.mean(rep_costs, axis=0))
            metrics[name]["lambda"].append(Counter(rep_lambdas).most_common(1)[0][0])
            metrics[name]["output_var"].append(float(np.mean(rep_vars)))
            metrics[name]["nonempty"].extend(np.mean(rep_routes, axis=0))
            metrics[name]["cluster_std"].extend(np.mean(rep_spread, axis=0))

        checkpoint = selection["checkpoint"]
        test_seeds = [instance.seed for instance in test]
        q_ref = np.asarray([checkpoint["costs"]["quantum_ring_trained"][s] for s in test_seeds])
        random_ref = np.asarray([checkpoint["costs"]["random_assignment"][s] for s in test_seeds])
        seed_lines.append((outer_seed, float(np.mean(no_noise_a)), float(np.mean(q_ref)), float(np.mean(random_ref))))


    expected_quantum = {10: 6.3779, 12: 6.9706}[size]
    expected_random = {10: 6.4711, 12: 7.1301}[size]
    measured_quantum = float(np.mean(costs["quantum_ring_trained"]))
    measured_random = float(np.mean([row[3] for row in seed_lines]))
    if abs(measured_quantum - expected_quantum) > 0.01:
        raise AssertionError(f"D5 quantum mismatch N={size}: {measured_quantum:.4f}")
    if abs(measured_random - expected_random) > 0.01:
        raise AssertionError(f"D5 random mismatch N={size}: {measured_random:.4f}")
    rand = np.asarray(random_reference)
    bal = np.asarray(balanced_reference)
    lines = []
    if not args.append or not os.path.exists(args.output):
        lines += [
            "# Null-control diagnostic",
            "",
            "Cache-only; lower route cost is better. Five outer seeds, 200 labeled training and 250 test instances per seed. No simulator or OR-Tools call was permitted in this phase.",
            "",
        ]
    lines += [f"## N={size}", "", "### D5: consistency check", "", "The unmodified-quantum null-control path and no-noise shot-noise path used the same `make_split` IDs, target JSON routes, pair targets, cached feature shard, readout, and `nn2opt` decoder. Per-instance costs were asserted equal within 1e-10.", "", "| Seed | Quantum no-noise | Quantum checkpoint | Random checkpoint | Δ quantum | Δ random |", "|---:|---:|---:|---:|---:|---:|"]
    for seed, got, ref, rand_ref in seed_lines:
        lines.append(f"| {seed} | {got:.4f} | {ref:.4f} | {rand_ref:.4f} | {got-ref:+.4f} | {rand_ref-({10:6.4711,12:7.1301}[size]):+.4f} |")
    mean_q = np.mean([x[1] for x in seed_lines])
    mean_q_ref = np.mean([x[2] for x in seed_lines])
    mean_rand_ref = np.mean([x[3] for x in seed_lines])
    lines += [f"", f"Mean quantum {mean_q:.4f}; checkpoint mean {mean_q_ref:.4f}; difference {mean_q-mean_q_ref:+.4f}. Random checkpoint mean {mean_rand_ref:.4f}; published fair benchmark reference { {10:6.4711,12:7.1301}[size]:.4f}.", "", "### D0/D2/D3/D4 and label-free D1", "", "D0 differences use one common 100-draw random-assignment expectation per test instance; D3 uses 100 uniform balanced partitions (four groups, sizes differ by at most one). CIs are paired bootstrap 95% intervals over 1,250 test instances. D1 entries report only the label-free structural metrics requested here. D4 lambda is the 5-fold CV ridge alpha; output variance is the variance of raw predicted pair scores on test pairs.", "", "| Variant | Mean cost | vs random (95% CI) | vs balanced (95% CI) | Mean non-empty routes | Mean cluster-size SD | CV λ (mode) | Test-score variance |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for name in variants:
        values = np.asarray(costs[name])
        lines.append(f"| {name} | {np.mean(values):.4f} | {paired_ci(values-rand, size*10+1)} | {paired_ci(values-bal, size*10+2)} | {np.mean(metrics[name]['nonempty']):.3f} | {np.mean(metrics[name]['cluster_std']):.3f} | {Counter(metrics[name]['lambda']).most_common(1)[0][0]:g} | {np.mean(metrics[name]['output_var']):.6g} |")
    lines += [
        f"| random_assignment (100 draws) | {np.mean(rand):.4f} | reference | {paired_ci(rand-bal, size*10+3)} | {np.mean(random_nonempty):.3f} | {np.mean(random_cluster_std):.3f} | n/a | n/a |",
        f"| random_balanced_assignment (100 draws) | {np.mean(bal):.4f} | {paired_ci(bal-rand, size*10+4)} | reference | {np.mean(balanced_nonempty):.3f} | {np.mean(balanced_cluster_std):.3f} | n/a | n/a |",
        "",
        "Baseline route counts are the mean number of non-empty decoded routes over the 100 assignment draws per instance; cluster-size SD is computed from the assigned customer counts before decoding.",
        "",
        "#### D2 constant-readout decoder partitions",
        "",
    ]
    for seed, partition, alpha in constant_partitions:
        lines.append(f"- Test instance seed `{seed}`: `{partition}` (selected λ={alpha:g}). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.")
    lines += ["", "#### No-noise path equivalence", "", f"Maximum per-instance absolute cost difference between the null-control implementation and shot-noise no-noise implementation: `{np.max(np.abs(quantum_and_shot_equivalence)):.3g}`.", ""]
    if args.append and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as handle:
            existing = handle.read()
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(existing.rstrip() + "\n\n" + "\n".join(lines))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
