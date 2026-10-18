# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 4 | random_assignment | 3.0816 | 8.35% | 0.129 | nan | nan |
| 4 | sweep_assignment | 3.1806 | 12.14% | 0.124 | nan | nan |
| 4 | classical_esn | 3.0171 | 6.02% | 0.194 | 0.1729 | 0.06289 |
| 4 | quantum_ring_trained | 3.0326 | 6.32% | 3.918 | 0.2504 | 0.09461 |
| 4 | quantum_shallow_trained | 3.0449 | 7.07% | 3.106 | 0.2809 | 0.11740 |
| 4 | quantum_ring_random | 3.0317 | 6.47% | 3.928 | 0.2578 | 0.10531 |
| 8 | random_assignment | 5.6332 | 19.26% | 0.210 | nan | nan |
| 8 | sweep_assignment | 5.3571 | 12.89% | 0.221 | nan | nan |
| 8 | classical_esn | 5.5915 | 18.13% | 0.304 | 0.3412 | 0.18612 |
| 8 | quantum_ring_trained | 5.5841 | 17.93% | 10.955 | 0.1225 | 0.02522 |
| 8 | quantum_shallow_trained | 5.5738 | 17.92% | 9.074 | 0.1140 | 0.02525 |
| 8 | quantum_ring_random | 5.5517 | 17.46% | 11.079 | 0.0952 | 0.01688 |
| 12 | random_assignment | 7.1563 | 35.51% | 0.278 | nan | nan |
| 12 | sweep_assignment | 6.3334 | 19.44% | 0.285 | nan | nan |
| 12 | classical_esn | 6.9977 | 32.30% | 0.467 | 0.4160 | 0.24915 |
| 12 | quantum_ring_trained | 7.0813 | 34.21% | 29.011 | 0.2321 | 0.09106 |
| 12 | quantum_shallow_trained | 7.0155 | 32.69% | 24.320 | 0.2217 | 0.09386 |
| 12 | quantum_ring_random | 6.9866 | 32.17% | 28.882 | 0.2027 | 0.07620 |
| 16 | random_assignment | 8.1953 | 45.49% | 0.402 | nan | nan |
| 16 | sweep_assignment | 6.6919 | 18.13% | 0.383 | nan | nan |
| 16 | classical_esn | 8.0048 | 41.79% | 0.675 | 0.5257 | 0.36298 |
| 16 | quantum_ring_trained | 7.9672 | 41.21% | 1408.233 | 0.1999 | 0.07871 |
| 16 | quantum_shallow_trained | 7.9632 | 41.18% | 1196.068 | 0.2095 | 0.10018 |
| 16 | quantum_ring_random | 7.9434 | 40.89% | 1928.863 | 0.2436 | 0.10513 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 4 | quantum_ring_trained - classical_esn | 0.0154 | [-0.0346, 0.0644] | no | 250 |
| 4 | quantum_shallow_trained - classical_esn | 0.0278 | [-0.0185, 0.0788] | no | 250 |
| 4 | quantum_ring_trained - quantum_ring_random | 0.0009 | [-0.0486, 0.0490] | no | 250 |
| 4 | quantum_ring_trained - random_assignment | -0.0490 | [-0.0944, -0.0048] | yes | 250 |
| 4 | quantum_ring_trained - sweep_assignment | -0.1480 | [-0.1891, -0.1093] | yes | 250 |
| 4 | classical_esn - sweep_assignment | -0.1635 | [-0.2073, -0.1216] | yes | 250 |
| 4 | sweep_assignment - random_assignment | 0.0990 | [0.0668, 0.1353] | yes | 250 |
| 8 | quantum_ring_trained - classical_esn | -0.0074 | [-0.0727, 0.0557] | no | 250 |
| 8 | quantum_shallow_trained - classical_esn | -0.0177 | [-0.0829, 0.0479] | no | 250 |
| 8 | quantum_ring_trained - quantum_ring_random | 0.0324 | [-0.0258, 0.0927] | no | 250 |
| 8 | quantum_ring_trained - random_assignment | -0.0491 | [-0.1143, 0.0190] | no | 250 |
| 8 | quantum_ring_trained - sweep_assignment | 0.2270 | [0.1630, 0.2935] | yes | 250 |
| 8 | classical_esn - sweep_assignment | 0.2344 | [0.1721, 0.3016] | yes | 250 |
| 8 | sweep_assignment - random_assignment | -0.2761 | [-0.3430, -0.2059] | yes | 250 |
| 12 | quantum_ring_trained - classical_esn | 0.0836 | [0.0006, 0.1631] | yes | 250 |
| 12 | quantum_shallow_trained - classical_esn | 0.0178 | [-0.0560, 0.0883] | no | 250 |
| 12 | quantum_ring_trained - quantum_ring_random | 0.0947 | [0.0200, 0.1727] | yes | 250 |
| 12 | quantum_ring_trained - random_assignment | -0.0751 | [-0.1541, 0.0056] | no | 250 |
| 12 | quantum_ring_trained - sweep_assignment | 0.7479 | [0.6566, 0.8424] | yes | 250 |
| 12 | classical_esn - sweep_assignment | 0.6643 | [0.5763, 0.7491] | yes | 250 |
| 12 | sweep_assignment - random_assignment | -0.8229 | [-0.9020, -0.7435] | yes | 250 |
| 16 | quantum_ring_trained - classical_esn | -0.0376 | [-0.1174, 0.0357] | no | 250 |
| 16 | quantum_shallow_trained - classical_esn | -0.0416 | [-0.1155, 0.0406] | no | 250 |
| 16 | quantum_ring_trained - quantum_ring_random | 0.0238 | [-0.0578, 0.1026] | no | 250 |
| 16 | quantum_ring_trained - random_assignment | -0.2282 | [-0.3097, -0.1481] | yes | 250 |
| 16 | quantum_ring_trained - sweep_assignment | 1.2753 | [1.1777, 1.3659] | yes | 250 |
| 16 | classical_esn - sweep_assignment | 1.3129 | [1.2265, 1.3997] | yes | 250 |
| 16 | sweep_assignment - random_assignment | -1.5034 | [-1.5885, -1.4184] | yes | 250 |
