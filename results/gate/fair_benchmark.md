# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 8 | random_assignment | 5.6240 | 19.20% | 0.207 | nan | nan |
| 8 | distance_pair | 5.2883 | 11.77% | 2.018 | 0.5125 | 0.05943 |
| 8 | classical_esn | 5.6545 | 19.85% | 2.099 | 0.2387 | 0.11607 |
| 8 | quantum_ring_trained | 5.6330 | 19.40% | 12.840 | 0.1237 | 0.02703 |
| 8 | quantum_shallow_trained | 5.6155 | 19.07% | 10.800 | 0.1680 | 0.04611 |
| 8 | quantum_ring_random | 5.6454 | 19.69% | 12.737 | 0.1104 | 0.02115 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 8 | distance_pair - random_assignment | -0.3357 | [-0.3678, -0.3006] | yes | 1250 |
| 8 | classical_esn - distance_pair | 0.3662 | [0.3346, 0.3979] | yes | 1250 |
| 8 | quantum_ring_trained - distance_pair | 0.3447 | [0.3112, 0.3778] | yes | 1250 |
| 8 | quantum_ring_trained - classical_esn | -0.0215 | [-0.0516, 0.0088] | no | 1250 |
| 8 | quantum_shallow_trained - classical_esn | -0.0390 | [-0.0701, -0.0082] | yes | 1250 |
| 8 | quantum_ring_trained - quantum_ring_random | -0.0124 | [-0.0432, 0.0176] | no | 1250 |
| 8 | quantum_ring_trained - random_assignment | 0.0089 | [-0.0208, 0.0419] | no | 1250 |
