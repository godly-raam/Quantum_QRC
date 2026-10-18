# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 4 | random_assignment | 3.0816 | 8.35% | 0.130 | nan | nan |
| 4 | sweep_assignment | 3.1806 | 12.14% | 0.124 | nan | nan |
| 4 | classical_esn | 3.0557 | 7.33% | 0.187 | 0.1729 | 0.06289 |
| 4 | quantum_ring_trained | 3.0437 | 6.74% | 3.980 | 0.2504 | 0.09461 |
| 4 | quantum_shallow_trained | 3.0516 | 7.20% | 3.145 | 0.2809 | 0.11740 |
| 4 | quantum_ring_random | 3.0326 | 6.85% | 3.962 | 0.2578 | 0.10531 |
| 8 | random_assignment | 5.6332 | 19.26% | 0.213 | nan | nan |
| 8 | sweep_assignment | 5.3571 | 12.89% | 0.217 | nan | nan |
| 8 | classical_esn | 5.5514 | 17.51% | 0.293 | 0.3412 | 0.18612 |
| 8 | quantum_ring_trained | 5.6572 | 19.62% | 11.124 | 0.1225 | 0.02522 |
| 8 | quantum_shallow_trained | 5.5977 | 18.48% | 9.511 | 0.1140 | 0.02525 |
| 8 | quantum_ring_random | 5.6135 | 18.78% | 11.362 | 0.0952 | 0.01688 |
| 12 | random_assignment | 7.1563 | 35.52% | 0.281 | nan | nan |
| 12 | sweep_assignment | 6.3334 | 19.45% | 0.277 | nan | nan |
| 12 | classical_esn | 7.1322 | 34.97% | 0.373 | 0.4160 | 0.24915 |
| 12 | quantum_ring_trained | 7.1197 | 34.74% | 28.238 | 0.2321 | 0.09106 |
| 12 | quantum_shallow_trained | 7.0954 | 34.29% | 24.224 | 0.2217 | 0.09386 |
| 12 | quantum_ring_random | 7.1353 | 35.08% | 28.418 | 0.2027 | 0.07620 |
| 16 | random_assignment | 8.1953 | 45.46% | 0.398 | nan | nan |
| 16 | sweep_assignment | 6.6919 | 18.10% | 0.384 | nan | nan |
| 16 | classical_esn | 8.1754 | 44.92% | 0.513 | 0.5257 | 0.36298 |
| 16 | quantum_ring_trained | 8.1564 | 44.70% | 1394.592 | 0.1999 | 0.07871 |
| 16 | quantum_shallow_trained | 8.0973 | 43.62% | 1200.857 | 0.2095 | 0.10018 |
| 16 | quantum_ring_random | 8.1581 | 44.77% | 1349.248 | 0.2436 | 0.10513 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 4 | quantum_ring_trained - classical_esn | -0.0120 | [-0.0623, 0.0396] | no | 250 |
| 4 | quantum_shallow_trained - classical_esn | -0.0041 | [-0.0522, 0.0459] | no | 250 |
| 4 | quantum_ring_trained - quantum_ring_random | 0.0111 | [-0.0275, 0.0546] | no | 250 |
| 4 | quantum_ring_trained - random_assignment | -0.0379 | [-0.0796, 0.0033] | no | 250 |
| 4 | quantum_ring_trained - sweep_assignment | -0.1369 | [-0.1768, -0.0985] | yes | 250 |
| 4 | classical_esn - sweep_assignment | -0.1249 | [-0.1689, -0.0862] | yes | 250 |
| 4 | sweep_assignment - random_assignment | 0.0990 | [0.0668, 0.1353] | yes | 250 |
| 8 | quantum_ring_trained - classical_esn | 0.1058 | [0.0395, 0.1760] | yes | 250 |
| 8 | quantum_shallow_trained - classical_esn | 0.0463 | [-0.0214, 0.1181] | no | 250 |
| 8 | quantum_ring_trained - quantum_ring_random | 0.0437 | [-0.0206, 0.1060] | no | 250 |
| 8 | quantum_ring_trained - random_assignment | 0.0240 | [-0.0438, 0.0898] | no | 250 |
| 8 | quantum_ring_trained - sweep_assignment | 0.3001 | [0.2307, 0.3715] | yes | 250 |
| 8 | classical_esn - sweep_assignment | 0.1943 | [0.1273, 0.2619] | yes | 250 |
| 8 | sweep_assignment - random_assignment | -0.2761 | [-0.3430, -0.2059] | yes | 250 |
| 12 | quantum_ring_trained - classical_esn | -0.0125 | [-0.0964, 0.0683] | no | 250 |
| 12 | quantum_shallow_trained - classical_esn | -0.0368 | [-0.1168, 0.0430] | no | 250 |
| 12 | quantum_ring_trained - quantum_ring_random | -0.0157 | [-0.0728, 0.0415] | no | 250 |
| 12 | quantum_ring_trained - random_assignment | -0.0367 | [-0.1226, 0.0458] | no | 250 |
| 12 | quantum_ring_trained - sweep_assignment | 0.7863 | [0.6993, 0.8653] | yes | 250 |
| 12 | classical_esn - sweep_assignment | 0.7988 | [0.7201, 0.8826] | yes | 250 |
| 12 | sweep_assignment - random_assignment | -0.8229 | [-0.9020, -0.7435] | yes | 250 |
| 16 | quantum_ring_trained - classical_esn | -0.0190 | [-0.0932, 0.0576] | no | 250 |
| 16 | quantum_shallow_trained - classical_esn | -0.0781 | [-0.1526, 0.0021] | no | 250 |
| 16 | quantum_ring_trained - quantum_ring_random | -0.0018 | [-0.0601, 0.0562] | no | 250 |
| 16 | quantum_ring_trained - random_assignment | -0.0390 | [-0.1187, 0.0436] | no | 250 |
| 16 | quantum_ring_trained - sweep_assignment | 1.4645 | [1.3776, 1.5522] | yes | 250 |
| 16 | classical_esn - sweep_assignment | 1.4835 | [1.3900, 1.5712] | yes | 250 |
| 16 | sweep_assignment - random_assignment | -1.5034 | [-1.5885, -1.4184] | yes | 250 |
