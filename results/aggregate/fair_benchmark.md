# Aggregated label-free fair benchmark

N=4 is degenerate (three customers, four vehicles) and is excluded from any claim.
N=8 uses the full-method-set run; the gate run is retained as a duplicate check.
Seeds available per size: N=4: 5, N=8: 5, N=12: 5, N=16: 1. N=16 is provisional until seeds 1-4 are run.
Locations/gaps are relative to the OR-Tools 1 s reference.

## Summary

| Size | Method | Seeds | Runs | Mean cost | Mean gap | Median time (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 4 | classical_esn | 5 | 1250 | 3.1723 | 10.32% | 0.269 |
| 4 | distance_pair | 5 | 1250 | 3.1723 | 10.32% | 0.246 |
| 4 | kmeans_distance | 5 | 1250 | 3.1723 | 10.32% | 82.233 |
| 4 | quantum_ring_random | 5 | 1250 | 3.1723 | 10.32% | 4.093 |
| 4 | quantum_ring_trained | 5 | 1250 | 3.1723 | 10.32% | 4.089 |
| 4 | quantum_shallow_trained | 5 | 1250 | 3.1723 | 10.32% | 3.252 |
| 4 | random_assignment | 5 | 1250 | 3.0954 | 7.41% | 0.130 |
| 4 | sweep_assignment | 5 | 1250 | 3.1723 | 10.32% | 0.126 |
| 8 | classical_esn | 5 | 1250 | 5.6545 | 19.85% | 2.035 |
| 8 | distance_pair | 5 | 1250 | 5.2883 | 11.77% | 2.048 |
| 8 | kmeans_distance | 5 | 1250 | 5.3596 | 13.39% | 83.887 |
| 8 | quantum_ring_random | 5 | 1250 | 5.6454 | 19.69% | 2.348 |
| 8 | quantum_ring_trained | 5 | 1250 | 5.6330 | 19.40% | 2.357 |
| 8 | quantum_shallow_trained | 5 | 1250 | 5.6155 | 19.07% | 2.357 |
| 8 | random_assignment | 5 | 1250 | 5.6240 | 19.20% | 0.212 |
| 8 | sweep_assignment | 5 | 1250 | 5.3744 | 13.59% | 0.215 |
| 12 | classical_esn | 5 | 1250 | 7.0720 | 34.76% | 7.845 |
| 12 | distance_pair | 5 | 1250 | 6.5077 | 23.70% | 7.628 |
| 12 | kmeans_distance | 5 | 1250 | 6.4395 | 22.51% | 30.885 |
| 12 | quantum_ring_random | 5 | 1250 | 6.9987 | 33.31% | 37.330 |
| 12 | quantum_ring_trained | 5 | 1250 | 6.9706 | 32.72% | 37.015 |
| 12 | quantum_shallow_trained | 5 | 1250 | 6.9910 | 33.17% | 33.525 |
| 12 | random_assignment | 5 | 1250 | 7.1301 | 35.90% | 0.282 |
| 12 | sweep_assignment | 5 | 1250 | 6.2668 | 19.03% | 0.273 |
| 16 | classical_esn | 1 | 250 | 8.1284 | 43.63% | 18.982 |
| 16 | distance_pair | 1 | 250 | 7.3985 | 30.47% | 18.760 |
| 16 | kmeans_distance | 1 | 250 | 7.2706 | 28.36% | 12.858 |
| 16 | quantum_ring_random | 1 | 250 | 7.9646 | 40.73% | 529.601 |
| 16 | quantum_ring_trained | 1 | 250 | 7.9596 | 40.58% | 536.395 |
| 16 | quantum_shallow_trained | 1 | 250 | 8.0066 | 41.43% | 497.663 |
| 16 | random_assignment | 1 | 250 | 8.2218 | 45.43% | 0.418 |
| 16 | sweep_assignment | 1 | 250 | 6.7291 | 18.34% | 0.383 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | Excludes 0 | N |
|---:|---|---:|---|---|---:|
| 4 | distance_pair - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | distance_pair - sweep_assignment | -0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | distance_pair - kmeans_distance | -0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | kmeans_distance - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | sweep_assignment - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | classical_esn - distance_pair | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - distance_pair | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_shallow_trained - distance_pair | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_random - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | quantum_ring_trained - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | quantum_ring_trained - classical_esn | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_shallow_trained - classical_esn | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - quantum_ring_random | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 8 | distance_pair - random_assignment | -0.3357 | [-0.3678, -0.3006] | yes | 1250 |
| 8 | distance_pair - sweep_assignment | -0.0861 | [-0.1184, -0.0537] | yes | 1250 |
| 8 | distance_pair - kmeans_distance | -0.0713 | [-0.1019, -0.0407] | yes | 1250 |
| 8 | kmeans_distance - random_assignment | -0.2644 | [-0.2956, -0.2330] | yes | 1250 |
| 8 | sweep_assignment - random_assignment | -0.2497 | [-0.2775, -0.2188] | yes | 1250 |
| 8 | classical_esn - distance_pair | 0.3662 | [0.3346, 0.3979] | yes | 1250 |
| 8 | quantum_ring_trained - distance_pair | 0.3447 | [0.3112, 0.3778] | yes | 1250 |
| 8 | quantum_shallow_trained - distance_pair | 0.3272 | [0.2940, 0.3608] | yes | 1250 |
| 8 | quantum_ring_random - random_assignment | 0.0213 | [-0.0100, 0.0513] | no | 1250 |
| 8 | quantum_ring_trained - random_assignment | 0.0089 | [-0.0208, 0.0419] | no | 1250 |
| 8 | quantum_ring_trained - classical_esn | -0.0215 | [-0.0516, 0.0088] | no | 1250 |
| 8 | quantum_shallow_trained - classical_esn | -0.0390 | [-0.0701, -0.0082] | yes | 1250 |
| 8 | quantum_ring_trained - quantum_ring_random | -0.0124 | [-0.0432, 0.0176] | no | 1250 |
| 12 | distance_pair - random_assignment | -0.6224 | [-0.6609, -0.5809] | yes | 1250 |
| 12 | distance_pair - sweep_assignment | 0.2409 | [0.2022, 0.2808] | yes | 1250 |
| 12 | distance_pair - kmeans_distance | 0.0682 | [0.0309, 0.1047] | yes | 1250 |
| 12 | kmeans_distance - random_assignment | -0.6906 | [-0.7297, -0.6505] | yes | 1250 |
| 12 | sweep_assignment - random_assignment | -0.8633 | [-0.8995, -0.8260] | yes | 1250 |
| 12 | classical_esn - distance_pair | 0.5643 | [0.5251, 0.6015] | yes | 1250 |
| 12 | quantum_ring_trained - distance_pair | 0.4629 | [0.4258, 0.4995] | yes | 1250 |
| 12 | quantum_shallow_trained - distance_pair | 0.4833 | [0.4451, 0.5217] | yes | 1250 |
| 12 | quantum_ring_random - random_assignment | -0.1314 | [-0.1671, -0.0951] | yes | 1250 |
| 12 | quantum_ring_trained - random_assignment | -0.1594 | [-0.1948, -0.1241] | yes | 1250 |
| 12 | quantum_ring_trained - classical_esn | -0.1014 | [-0.1359, -0.0657] | yes | 1250 |
| 12 | quantum_shallow_trained - classical_esn | -0.0810 | [-0.1184, -0.0445] | yes | 1250 |
| 12 | quantum_ring_trained - quantum_ring_random | -0.0281 | [-0.0653, 0.0057] | no | 1250 |
| 16 | distance_pair - random_assignment | -0.8233 | [-0.9139, -0.7335] | yes | 250 |
| 16 | distance_pair - sweep_assignment | 0.6693 | [0.5740, 0.7694] | yes | 250 |
| 16 | distance_pair - kmeans_distance | 0.1279 | [0.0389, 0.2167] | yes | 250 |
| 16 | kmeans_distance - random_assignment | -0.9512 | [-1.0403, -0.8585] | yes | 250 |
| 16 | sweep_assignment - random_assignment | -1.4926 | [-1.5880, -1.3970] | yes | 250 |
| 16 | classical_esn - distance_pair | 0.7300 | [0.6434, 0.8158] | yes | 250 |
| 16 | quantum_ring_trained - distance_pair | 0.5611 | [0.4722, 0.6469] | yes | 250 |
| 16 | quantum_shallow_trained - distance_pair | 0.6081 | [0.5261, 0.6959] | yes | 250 |
| 16 | quantum_ring_random - random_assignment | -0.2571 | [-0.3450, -0.1696] | yes | 250 |
| 16 | quantum_ring_trained - random_assignment | -0.2622 | [-0.3459, -0.1790] | yes | 250 |
| 16 | quantum_ring_trained - classical_esn | -0.1689 | [-0.2422, -0.0945] | yes | 250 |
| 16 | quantum_shallow_trained - classical_esn | -0.1219 | [-0.1984, -0.0405] | yes | 250 |
| 16 | quantum_ring_trained - quantum_ring_random | -0.0051 | [-0.0804, 0.0751] | no | 250 |

## Decoder diagnostic (nn2opt vs index ordering)

| Size | Method | Cost nn2opt | Cost index | Decoder gain |
|---:|---|---:|---:|---:|
| 4 | classical_esn | 3.1723 | 3.1723 | 0.00% |
| 4 | distance_pair | 3.1723 | 3.1723 | 0.00% |
| 4 | kmeans_distance | 3.1723 | 3.1723 | 0.00% |
| 4 | quantum_ring_random | 3.1723 | 3.1723 | 0.00% |
| 4 | quantum_ring_trained | 3.1723 | 3.1723 | 0.00% |
| 4 | quantum_shallow_trained | 3.1723 | 3.1723 | 0.00% |
| 4 | random_assignment | 3.0954 | 3.0954 | 0.00% |
| 4 | sweep_assignment | 3.1723 | 3.1723 | 0.00% |
| 8 | classical_esn | 5.6545 | 5.7213 | 1.12% |
| 8 | distance_pair | 5.2883 | 5.3432 | 1.04% |
| 8 | kmeans_distance | 5.3596 | 5.4062 | 0.83% |
| 8 | quantum_ring_random | 5.6454 | 5.7085 | 1.06% |
| 8 | quantum_ring_trained | 5.6330 | 5.6994 | 1.11% |
| 8 | quantum_shallow_trained | 5.6155 | 5.6850 | 1.17% |
| 8 | random_assignment | 5.6240 | 5.7082 | 1.38% |
| 8 | sweep_assignment | 5.3744 | 5.4157 | 0.71% |
| 12 | classical_esn | 7.0720 | 7.7855 | 8.88% |
| 12 | distance_pair | 6.5077 | 7.1769 | 9.15% |
| 12 | kmeans_distance | 6.4395 | 6.9804 | 7.53% |
| 12 | quantum_ring_random | 6.9987 | 7.7556 | 9.49% |
| 12 | quantum_ring_trained | 6.9706 | 7.7871 | 10.17% |
| 12 | quantum_shallow_trained | 6.9910 | 7.7428 | 9.43% |
| 12 | random_assignment | 7.1301 | 7.8070 | 8.34% |
| 12 | sweep_assignment | 6.2668 | 6.7817 | 7.28% |
| 16 | classical_esn | 8.1284 | 9.9602 | 18.08% |
| 16 | distance_pair | 7.3985 | 9.0368 | 17.84% |
| 16 | kmeans_distance | 7.2706 | 8.6960 | 15.94% |
| 16 | quantum_ring_random | 7.9646 | 9.9131 | 19.39% |
| 16 | quantum_ring_trained | 7.9596 | 9.9887 | 20.01% |
| 16 | quantum_shallow_trained | 8.0066 | 9.9389 | 19.05% |
| 16 | random_assignment | 8.2218 | 9.9464 | 17.06% |
| 16 | sweep_assignment | 6.7291 | 7.9658 | 15.14% |
