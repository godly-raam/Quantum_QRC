# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 4 | random_assignment | 3.0954 | 7.41% | 0.130 | nan | nan |
| 4 | sweep_assignment | 3.1723 | 10.32% | 0.126 | nan | nan |
| 4 | kmeans_distance | 3.1723 | 10.32% | 82.233 | nan | nan |
| 4 | distance_pair | 3.1723 | 10.32% | 0.246 | 0.4909 | 0.06142 |
| 4 | classical_esn | 3.1723 | 10.32% | 0.269 | 0.1489 | 0.05220 |
| 4 | quantum_ring_trained | 3.1723 | 10.32% | 4.089 | 0.1853 | 0.05116 |
| 4 | quantum_shallow_trained | 3.1723 | 10.32% | 3.252 | 0.1847 | 0.05763 |
| 4 | quantum_ring_random | 3.1723 | 10.32% | 4.093 | 0.1931 | 0.05093 |
| 12 | random_assignment | 7.1301 | 35.90% | 0.282 | nan | nan |
| 12 | sweep_assignment | 6.2668 | 19.03% | 0.273 | nan | nan |
| 12 | kmeans_distance | 6.4395 | 22.51% | 30.885 | nan | nan |
| 12 | distance_pair | 6.5077 | 23.70% | 7.628 | 0.5175 | 0.06006 |
| 12 | classical_esn | 7.0720 | 34.76% | 7.845 | 0.2777 | 0.14087 |
| 12 | quantum_ring_trained | 6.9706 | 32.72% | 37.015 | 0.1349 | 0.02975 |
| 12 | quantum_shallow_trained | 6.9910 | 33.17% | 33.525 | 0.1411 | 0.03506 |
| 12 | quantum_ring_random | 6.9987 | 33.31% | 37.330 | 0.0720 | 0.01037 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 4 | distance_pair - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | distance_pair - sweep_assignment | -0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | distance_pair - kmeans_distance | -0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | classical_esn - distance_pair | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - distance_pair | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - classical_esn | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_shallow_trained - classical_esn | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - quantum_ring_random | 0.0000 | [0.0000, 0.0000] | no | 1250 |
| 4 | quantum_ring_trained - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 4 | quantum_ring_trained - sweep_assignment | -0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | kmeans_distance - random_assignment | 0.0769 | [0.0635, 0.0904] | yes | 1250 |
| 12 | distance_pair - random_assignment | -0.6224 | [-0.6609, -0.5809] | yes | 1250 |
| 12 | distance_pair - sweep_assignment | 0.2409 | [0.2022, 0.2808] | yes | 1250 |
| 12 | distance_pair - kmeans_distance | 0.0682 | [0.0309, 0.1047] | yes | 1250 |
| 12 | classical_esn - distance_pair | 0.5643 | [0.5251, 0.6015] | yes | 1250 |
| 12 | quantum_ring_trained - distance_pair | 0.4629 | [0.4258, 0.4995] | yes | 1250 |
| 12 | quantum_ring_trained - classical_esn | -0.1014 | [-0.1359, -0.0657] | yes | 1250 |
| 12 | quantum_shallow_trained - classical_esn | -0.0810 | [-0.1184, -0.0445] | yes | 1250 |
| 12 | quantum_ring_trained - quantum_ring_random | -0.0281 | [-0.0653, 0.0057] | no | 1250 |
| 12 | quantum_ring_trained - random_assignment | -0.1594 | [-0.1948, -0.1241] | yes | 1250 |
| 12 | quantum_ring_trained - sweep_assignment | 0.7038 | [0.6660, 0.7408] | yes | 1250 |
| 12 | kmeans_distance - random_assignment | -0.6906 | [-0.7297, -0.6505] | yes | 1250 |
