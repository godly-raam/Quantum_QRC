# Fair quantum-versus-classical VRP benchmark

Lower route cost is better. Gap is relative to the OR-Tools 1 s reference. Times are per-solve wall clock for the test split, excluding training.

| Size | Method | Mean cost | Mean gap | Median time (ms) | Feature mean abs | Feature variance |
|---:|---|---:|---:|---:|---:|---:|
| 6 | random_assignment | 4.5913 | 11.49% | 0.176 | nan | nan |
| 6 | sweep_assignment | 4.6083 | 11.83% | 0.189 | nan | nan |
| 6 | kmeans_distance | 4.5451 | 10.18% | 12.130 | nan | nan |
| 6 | distance_pair | 4.5011 | 9.18% | 0.758 | 0.5028 | 0.05952 |
| 6 | classical_esn | 4.6095 | 11.96% | 0.806 | 0.1922 | 0.07807 |
| 6 | quantum_ring_trained | 4.6237 | 12.28% | 8.031 | 0.1327 | 0.02792 |
| 6 | quantum_shallow_trained | 4.6325 | 12.58% | 6.553 | 0.1806 | 0.05211 |
| 6 | quantum_ring_random | 4.6164 | 12.07% | 7.958 | 0.1467 | 0.03478 |
| 10 | random_assignment | 6.4711 | 29.67% | 0.239 | nan | nan |
| 10 | sweep_assignment | 5.9269 | 18.46% | 0.241 | nan | nan |
| 10 | kmeans_distance | 5.9421 | 18.96% | 83.301 | nan | nan |
| 10 | distance_pair | 5.9246 | 18.42% | 4.246 | 0.5158 | 0.05972 |
| 10 | classical_esn | 6.4161 | 28.54% | 4.275 | 0.2416 | 0.11582 |
| 10 | quantum_ring_trained | 6.3779 | 27.80% | 4.697 | 0.1289 | 0.02757 |
| 10 | quantum_shallow_trained | 6.3929 | 28.09% | 4.691 | 0.1561 | 0.04118 |
| 10 | quantum_ring_random | 6.3807 | 27.88% | 4.704 | 0.0903 | 0.01515 |

## Paired differences (bootstrap 95% CI)

| Size | Comparison | Mean difference | 95% CI | CI excludes 0 | N |
|---:|---|---:|---|---|---:|
| 6 | distance_pair - random_assignment | -0.0902 | [-0.1118, -0.0689] | yes | 1250 |
| 6 | distance_pair - sweep_assignment | -0.1072 | [-0.1276, -0.0876] | yes | 1250 |
| 6 | distance_pair - kmeans_distance | -0.0439 | [-0.0627, -0.0255] | yes | 1250 |
| 6 | classical_esn - distance_pair | 0.1083 | [0.0896, 0.1291] | yes | 1250 |
| 6 | quantum_ring_trained - distance_pair | 0.1226 | [0.1031, 0.1416] | yes | 1250 |
| 6 | quantum_ring_trained - classical_esn | 0.0142 | [-0.0050, 0.0337] | no | 1250 |
| 6 | quantum_shallow_trained - classical_esn | 0.0231 | [0.0048, 0.0422] | yes | 1250 |
| 6 | quantum_ring_trained - quantum_ring_random | 0.0073 | [-0.0111, 0.0260] | no | 1250 |
| 6 | quantum_ring_trained - random_assignment | 0.0323 | [0.0104, 0.0546] | yes | 1250 |
| 6 | quantum_ring_trained - sweep_assignment | 0.0153 | [-0.0037, 0.0333] | no | 1250 |
| 6 | kmeans_distance - random_assignment | -0.0463 | [-0.0664, -0.0252] | yes | 1250 |
| 10 | distance_pair - random_assignment | -0.5465 | [-0.5864, -0.5080] | yes | 1250 |
| 10 | distance_pair - sweep_assignment | -0.0023 | [-0.0420, 0.0386] | no | 1250 |
| 10 | distance_pair - kmeans_distance | -0.0175 | [-0.0539, 0.0189] | no | 1250 |
| 10 | classical_esn - distance_pair | 0.4915 | [0.4527, 0.5306] | yes | 1250 |
| 10 | quantum_ring_trained - distance_pair | 0.4533 | [0.4156, 0.4908] | yes | 1250 |
| 10 | quantum_ring_trained - classical_esn | -0.0382 | [-0.0737, -0.0003] | yes | 1250 |
| 10 | quantum_shallow_trained - classical_esn | -0.0232 | [-0.0578, 0.0127] | no | 1250 |
| 10 | quantum_ring_trained - quantum_ring_random | -0.0028 | [-0.0393, 0.0341] | no | 1250 |
| 10 | quantum_ring_trained - random_assignment | -0.0932 | [-0.1280, -0.0564] | yes | 1250 |
| 10 | quantum_ring_trained - sweep_assignment | 0.4511 | [0.4141, 0.4888] | yes | 1250 |
| 10 | kmeans_distance - random_assignment | -0.5290 | [-0.5678, -0.4905] | yes | 1250 |
