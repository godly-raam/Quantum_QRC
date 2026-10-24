# Null-control diagnostic

Cache-only; lower route cost is better. Five outer seeds, 200 labeled training and 250 test instances per seed. No simulator or OR-Tools call was permitted in this phase.

## N=12

### D5: consistency check

The unmodified-quantum null-control path and no-noise shot-noise path used the same `make_split` IDs, target JSON routes, pair targets, cached feature shard, readout, and `nn2opt` decoder. Per-instance costs were asserted equal within 1e-10.

| Seed | Quantum no-noise | Quantum checkpoint | Random checkpoint | Δ quantum | Δ random |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.8204 | 6.8192 | 6.9854 | +0.0012 | -0.1447 |
| 1 | 6.9742 | 6.9826 | 7.1003 | -0.0085 | -0.0298 |
| 2 | 7.0639 | 7.0508 | 7.2439 | +0.0131 | +0.1138 |
| 3 | 7.0509 | 7.0754 | 7.2202 | -0.0245 | +0.0901 |
| 4 | 6.9125 | 6.9252 | 7.1007 | -0.0127 | -0.0294 |

Mean quantum 6.9644; checkpoint mean 6.9706; difference -0.0063. Random checkpoint mean 7.1301; published fair benchmark reference 7.1301.

### D0/D2/D3/D4 and label-free D1

D0 differences use one common 100-draw random-assignment expectation per test instance; D3 uses 100 uniform balanced partitions (four groups, sizes differ by at most one). CIs are paired bootstrap 95% intervals over 1,250 test instances. D1 entries report only the label-free structural metrics requested here. D4 lambda is the 5-fold CV ridge alpha; output variance is the variance of raw predicted pair scores on test pairs.

| Variant | Mean cost | vs random (95% CI) | vs balanced (95% CI) | Mean non-empty routes | Mean cluster-size SD | CV λ (mode) | Test-score variance |
|---|---:|---|---|---:|---:|---:|---:|
| quantum_ring_trained | 6.9644 | -0.1724 [-0.1991, -0.1451] | -0.2371 [-0.2647, -0.2096] | 4.000 | 2.314 | 100 | 0.000165164 |
| constant | 7.0326 | -0.1041 [-0.1307, -0.0770] | -0.1688 [-0.1976, -0.1411] | 4.000 | 3.031 | 1e-08 | 2.31112e-33 |
| shuffled | 7.0595 | -0.0773 [-0.1041, -0.0496] | -0.1420 [-0.1714, -0.1142] | 4.000 | 2.510 | 100 | 6.17419e-05 |
| gaussian | 7.0511 | -0.0857 [-0.1116, -0.0585] | -0.1504 [-0.1786, -0.1210] | 4.000 | 2.616 | 100 | 0.000101269 |
| S=1 | 7.0867 | -0.0501 [-0.0628, -0.0377] | -0.1148 [-0.1290, -0.1003] | 4.000 | 2.111 | 100 | 0.00010095 |
| S=100000 | 6.9632 | -0.1735 [-0.1986, -0.1495] | -0.2382 [-0.2645, -0.2130] | 4.000 | 2.317 | 100 | 0.000165598 |
| random_assignment (100 draws) | 7.1367 | reference | -0.0647 [-0.0700, -0.0590] | 4.000 | 1.332 | n/a | n/a |
| random_balanced_assignment (100 draws) | 7.2014 | +0.0647 [+0.0593, +0.0705] | reference | 4.000 | 0.433 | n/a | n/a |

Baseline route counts are the mean number of non-empty decoded routes over the 100 assignment draws per instance; cluster-size SD is computed from the assigned customer counts before decoding.

#### D2 constant-readout decoder partitions

- Test instance seed `10120200`: `[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9], [10]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.
- Test instance seed `10120201`: `[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9], [10]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.
- Test instance seed `10120202`: `[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9], [10]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.

#### No-noise path equivalence

Maximum per-instance absolute cost difference between the null-control implementation and shot-noise no-noise implementation: `0`.

## N=10

### D5: consistency check

The unmodified-quantum null-control path and no-noise shot-noise path used the same `make_split` IDs, target JSON routes, pair targets, cached feature shard, readout, and `nn2opt` decoder. Per-instance costs were asserted equal within 1e-10.

| Seed | Quantum no-noise | Quantum checkpoint | Random checkpoint | Δ quantum | Δ random |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.4041 | 6.4041 | 6.4532 | +0.0000 | -0.0179 |
| 1 | 6.3679 | 6.3679 | 6.4869 | +0.0000 | +0.0158 |
| 2 | 6.4044 | 6.4044 | 6.4964 | +0.0000 | +0.0253 |
| 3 | 6.3402 | 6.3402 | 6.4516 | +0.0000 | -0.0195 |
| 4 | 6.3744 | 6.3731 | 6.4676 | +0.0012 | -0.0035 |

Mean quantum 6.3782; checkpoint mean 6.3779; difference +0.0002. Random checkpoint mean 6.4711; published fair benchmark reference 6.4711.

### D0/D2/D3/D4 and label-free D1

D0 differences use one common 100-draw random-assignment expectation per test instance; D3 uses 100 uniform balanced partitions (four groups, sizes differ by at most one). CIs are paired bootstrap 95% intervals over 1,250 test instances. D1 entries report only the label-free structural metrics requested here. D4 lambda is the 5-fold CV ridge alpha; output variance is the variance of raw predicted pair scores on test pairs.

| Variant | Mean cost | vs random (95% CI) | vs balanced (95% CI) | Mean non-empty routes | Mean cluster-size SD | CV λ (mode) | Test-score variance |
|---|---:|---|---|---:|---:|---:|---:|
| quantum_ring_trained | 6.3782 | -0.0779 [-0.1047, -0.0522] | -0.1499 [-0.1763, -0.1237] | 4.000 | 1.694 | 100 | 0.000361503 |
| constant | 6.4229 | -0.0332 [-0.0594, -0.0071] | -0.1052 [-0.1312, -0.0789] | 4.000 | 2.165 | 1e-08 | 4.62223e-34 |
| shuffled | 6.4238 | -0.0322 [-0.0573, -0.0075] | -0.1042 [-0.1280, -0.0767] | 4.000 | 1.811 | 100 | 0.0001164 |
| gaussian | 6.4008 | -0.0553 [-0.0811, -0.0309] | -0.1273 [-0.1523, -0.1016] | 4.000 | 2.038 | 100 | 0.000163148 |
| S=1 | 6.4590 | +0.0029 [-0.0088, +0.0144] | -0.0691 [-0.0811, -0.0563] | 4.000 | 1.466 | 100 | 0.000131751 |
| S=100000 | 6.3766 | -0.0794 [-0.1054, -0.0547] | -0.1514 [-0.1775, -0.1260] | 4.000 | 1.694 | 100 | 0.000360168 |
| random_assignment (100 draws) | 6.4561 | reference | -0.0720 [-0.0776, -0.0665] | 4.000 | 1.207 | n/a | n/a |
| random_balanced_assignment (100 draws) | 6.5281 | +0.0720 [+0.0663, +0.0779] | reference | 4.000 | 0.433 | n/a | n/a |

Baseline route counts are the mean number of non-empty decoded routes over the 100 assignment draws per instance; cluster-size SD is computed from the assigned customer counts before decoding.

#### D2 constant-readout decoder partitions

- Test instance seed `10100200`: `[[0, 1, 2, 3, 4, 5], [6], [7], [8]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.
- Test instance seed `10100201`: `[[0, 1, 2, 3, 4, 5], [6], [7], [8]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.
- Test instance seed `10100202`: `[[0, 1, 2, 3, 4, 5], [6], [7], [8]]` (selected λ=1e-08). This is a giant prefix cluster plus singleton clusters. Equal pair scores break ties in `cluster_similarity` by scanning cluster pairs in current list order and updating only on strictly larger score.

#### No-noise path equivalence

Maximum per-instance absolute cost difference between the null-control implementation and shot-noise no-noise implementation: `0`.

## Rebuilt feature-cache D5 and archive

N=6 and N=10 quantum and ESN feature vectors were rebuilt with FastSim/Aer using `benchmark_fair` settings: deterministic `make_split` instance IDs, 450 instances per outer seed (200 training / 250 test), ring trainer seed `1000 + outer_seed`, 20 trainer samples, 100 optimizer iterations, angle scale 1.0, and no depot diagonal. The generated ESN used seed `outer_seed*10000 + size`. New shards include train and test feature vectors for all five seeds.

N=10 D5 was checked against `results/pairwise_6_10/checkpoints` before running the full N=10 diagnostic:

| Seed | QRC mean | QRC checkpoint | ESN mean | ESN checkpoint |
|---:|---:|---:|---:|---:|
| 0 | 6.4041 | 6.4041 | 6.4085 | 6.4085 |
| 1 | 6.3679 | 6.3679 | 6.4778 | 6.4778 |
| 2 | 6.4044 | 6.4044 | 6.3967 | 6.3967 |
| 3 | 6.3402 | 6.3402 | 6.3749 | 6.3749 |
| 4 | 6.3744 | 6.3731 | 6.4257 | 6.4226 |
| **Mean** | **6.378197** | **6.377948** | **6.416721** | **6.416105** |

Both N=10 D5 checks passed the requested ±0.01 tolerance (QRC Δ=+0.000250; ESN Δ=+0.000616). N=6 quantum/ESN feature shards were generated, but a readout-level D5 check was not possible: there are no `targets_n6_seed*.json` files containing the training OR-Tools routes, and OR-Tools was not permitted before optional step 5.

The rebuilt `results/feature_cache.tar.gz` contains feature shards for N=4, 6, 8, 10, 12, and 16.
