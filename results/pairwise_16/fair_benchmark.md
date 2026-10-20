# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 16 | random_assignment | 8.2223 | 45.67% | 0.399 | nan | nan |
| 16 | sweep_assignment | 6.7653 | 19.31% | 0.377 | nan | nan |
| 16 | kmeans_distance | 7.2230 | 27.68% | 12.561 | nan | nan |
| 16 | distance_pair | 7.3993 | 30.63% | 18.708 | 0.5223 | 0.06054 |
| 16 | classical_esn | 8.0919 | 43.26% | 19.181 | 0.3704 | 0.22038 |
| 16 | quantum_ring_trained | 7.9141 | 40.04% | 316.871 | 0.1003 | 0.01837 |
| 16 | quantum_shallow_trained | 7.9155 | 40.03% | 285.069 | 0.1223 | 0.02786 |
| 16 | quantum_ring_random | 7.9724 | 41.02% | 316.175 | 0.0590 | 0.00744 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 16 | distance_pair - random_assignment | -0.8230 | [-0.8688, -0.7773] | yes | 1000 |
| 16 | distance_pair - sweep_assignment | 0.6341 | [0.5857, 0.6781] | yes | 1000 |
| 16 | distance_pair - kmeans_distance | 0.1764 | [0.1302, 0.2217] | yes | 1000 |
| 16 | classical_esn - distance_pair | 0.6926 | [0.6498, 0.7401] | yes | 1000 |
| 16 | quantum_ring_trained - distance_pair | 0.5148 | [0.4720, 0.5570] | yes | 1000 |
| 16 | quantum_ring_trained - classical_esn | -0.1778 | [-0.2175, -0.1390] | yes | 1000 |
| 16 | quantum_shallow_trained - classical_esn | -0.1763 | [-0.2195, -0.1385] | yes | 1000 |
| 16 | quantum_ring_trained - quantum_ring_random | -0.0584 | [-0.0976, -0.0186] | yes | 1000 |
| 16 | quantum_ring_trained - random_assignment | -0.3083 | [-0.3488, -0.2690] | yes | 1000 |
| 16 | quantum_ring_trained - sweep_assignment | 1.1488 | [1.1055, 1.1938] | yes | 1000 |
| 16 | kmeans_distance - random_assignment | -0.9994 | [-1.0450, -0.9522] | yes | 1000 |
