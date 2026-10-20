# Aggregated label-free fair benchmark

N=4 is degenerate (three customers, four vehicles) and is excluded from any claim.
N=8 uses the full-method-set run; the gate run is retained as a duplicate check.
Seeds available per size: N=4: 5, N=6: 5, N=8: 5, N=10: 5, N=12: 5, N=16: 5. N=16 is a secondary size and is not used for any advantage claim.
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
| 6 | classical_esn | 5 | 1250 | 4.6095 | 11.96% | 0.806 |
| 6 | distance_pair | 5 | 1250 | 4.5011 | 9.18% | 0.758 |
| 6 | kmeans_distance | 5 | 1250 | 4.5451 | 10.18% | 12.130 |
| 6 | quantum_ring_random | 5 | 1250 | 4.6164 | 12.07% | 7.958 |
| 6 | quantum_ring_trained | 5 | 1250 | 4.6237 | 12.28% | 8.031 |
| 6 | quantum_shallow_trained | 5 | 1250 | 4.6325 | 12.58% | 6.553 |
| 6 | random_assignment | 5 | 1250 | 4.5913 | 11.49% | 0.176 |
| 6 | sweep_assignment | 5 | 1250 | 4.6083 | 11.83% | 0.189 |
| 8 | classical_esn | 5 | 1250 | 5.6545 | 19.85% | 2.035 |
| 8 | distance_pair | 5 | 1250 | 5.2883 | 11.77% | 2.048 |
| 8 | kmeans_distance | 5 | 1250 | 5.3596 | 13.39% | 83.887 |
| 8 | quantum_ring_random | 5 | 1250 | 5.6454 | 19.69% | 2.348 |
| 8 | quantum_ring_trained | 5 | 1250 | 5.6330 | 19.40% | 2.357 |
| 8 | quantum_shallow_trained | 5 | 1250 | 5.6155 | 19.07% | 2.357 |
| 8 | random_assignment | 5 | 1250 | 5.6240 | 19.20% | 0.212 |
| 8 | sweep_assignment | 5 | 1250 | 5.3744 | 13.59% | 0.215 |
| 10 | classical_esn | 5 | 1250 | 6.4161 | 28.54% | 4.275 |
| 10 | distance_pair | 5 | 1250 | 5.9246 | 18.42% | 4.246 |
| 10 | kmeans_distance | 5 | 1250 | 5.9421 | 18.96% | 83.301 |
| 10 | quantum_ring_random | 5 | 1250 | 6.3807 | 27.88% | 4.704 |
| 10 | quantum_ring_trained | 5 | 1250 | 6.3779 | 27.80% | 4.697 |
| 10 | quantum_shallow_trained | 5 | 1250 | 6.3929 | 28.09% | 4.691 |
| 10 | random_assignment | 5 | 1250 | 6.4711 | 29.67% | 0.239 |
| 10 | sweep_assignment | 5 | 1250 | 5.9269 | 18.46% | 0.241 |
| 12 | classical_esn | 5 | 1250 | 7.0720 | 34.76% | 7.845 |
| 12 | distance_pair | 5 | 1250 | 6.5077 | 23.70% | 7.628 |
| 12 | kmeans_distance | 5 | 1250 | 6.4395 | 22.51% | 30.885 |
| 12 | quantum_ring_random | 5 | 1250 | 6.9987 | 33.31% | 37.330 |
| 12 | quantum_ring_trained | 5 | 1250 | 6.9706 | 32.72% | 37.015 |
| 12 | quantum_shallow_trained | 5 | 1250 | 6.9910 | 33.17% | 33.525 |
| 12 | random_assignment | 5 | 1250 | 7.1301 | 35.90% | 0.282 |
| 12 | sweep_assignment | 5 | 1250 | 6.2668 | 19.03% | 0.273 |
| 16 | classical_esn | 5 | 1250 | 8.0992 | 43.33% | 19.133 |
| 16 | distance_pair | 5 | 1250 | 7.3992 | 30.60% | 18.724 |
| 16 | kmeans_distance | 5 | 1250 | 7.2325 | 27.81% | 12.619 |
| 16 | quantum_ring_random | 5 | 1250 | 7.9709 | 40.96% | 318.352 |
| 16 | quantum_ring_trained | 5 | 1250 | 7.9232 | 40.15% | 318.978 |
| 16 | quantum_shallow_trained | 5 | 1250 | 7.9337 | 40.31% | 288.030 |
| 16 | random_assignment | 5 | 1250 | 8.2222 | 45.62% | 0.401 |
| 16 | sweep_assignment | 5 | 1250 | 6.7580 | 19.12% | 0.378 |

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
| 4 | sweep_assignment - quantum_ring_trained | 0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 4 | kmeans_distance - quantum_ring_trained | 0.0000 | [-0.0000, 0.0000] | no | 1250 |
| 6 | distance_pair - random_assignment | -0.0902 | [-0.1118, -0.0689] | yes | 1250 |
| 6 | distance_pair - sweep_assignment | -0.1072 | [-0.1276, -0.0876] | yes | 1250 |
| 6 | distance_pair - kmeans_distance | -0.0439 | [-0.0627, -0.0255] | yes | 1250 |
| 6 | kmeans_distance - random_assignment | -0.0463 | [-0.0664, -0.0252] | yes | 1250 |
| 6 | sweep_assignment - random_assignment | 0.0170 | [-0.0046, 0.0397] | no | 1250 |
| 6 | classical_esn - distance_pair | 0.1083 | [0.0896, 0.1291] | yes | 1250 |
| 6 | quantum_ring_trained - distance_pair | 0.1226 | [0.1031, 0.1416] | yes | 1250 |
| 6 | quantum_shallow_trained - distance_pair | 0.1314 | [0.1115, 0.1513] | yes | 1250 |
| 6 | quantum_ring_random - random_assignment | 0.0251 | [0.0036, 0.0468] | yes | 1250 |
| 6 | quantum_ring_trained - random_assignment | 0.0323 | [0.0104, 0.0546] | yes | 1250 |
| 6 | quantum_ring_trained - classical_esn | 0.0142 | [-0.0050, 0.0337] | no | 1250 |
| 6 | quantum_shallow_trained - classical_esn | 0.0231 | [0.0048, 0.0422] | yes | 1250 |
| 6 | quantum_ring_trained - quantum_ring_random | 0.0073 | [-0.0111, 0.0260] | no | 1250 |
| 6 | sweep_assignment - quantum_ring_trained | -0.0153 | [-0.0333, 0.0037] | no | 1250 |
| 6 | kmeans_distance - quantum_ring_trained | -0.0786 | [-0.0988, -0.0586] | yes | 1250 |
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
| 8 | sweep_assignment - quantum_ring_trained | -0.2586 | [-0.2896, -0.2277] | yes | 1250 |
| 8 | kmeans_distance - quantum_ring_trained | -0.2733 | [-0.3053, -0.2406] | yes | 1250 |
| 10 | distance_pair - random_assignment | -0.5465 | [-0.5864, -0.5080] | yes | 1250 |
| 10 | distance_pair - sweep_assignment | -0.0023 | [-0.0420, 0.0386] | no | 1250 |
| 10 | distance_pair - kmeans_distance | -0.0175 | [-0.0539, 0.0189] | no | 1250 |
| 10 | kmeans_distance - random_assignment | -0.5290 | [-0.5678, -0.4905] | yes | 1250 |
| 10 | sweep_assignment - random_assignment | -0.5443 | [-0.5809, -0.5080] | yes | 1250 |
| 10 | classical_esn - distance_pair | 0.4915 | [0.4527, 0.5306] | yes | 1250 |
| 10 | quantum_ring_trained - distance_pair | 0.4533 | [0.4156, 0.4908] | yes | 1250 |
| 10 | quantum_shallow_trained - distance_pair | 0.4683 | [0.4322, 0.5047] | yes | 1250 |
| 10 | quantum_ring_random - random_assignment | -0.0904 | [-0.1258, -0.0539] | yes | 1250 |
| 10 | quantum_ring_trained - random_assignment | -0.0932 | [-0.1280, -0.0564] | yes | 1250 |
| 10 | quantum_ring_trained - classical_esn | -0.0382 | [-0.0737, -0.0003] | yes | 1250 |
| 10 | quantum_shallow_trained - classical_esn | -0.0232 | [-0.0578, 0.0127] | no | 1250 |
| 10 | quantum_ring_trained - quantum_ring_random | -0.0028 | [-0.0393, 0.0341] | no | 1250 |
| 10 | sweep_assignment - quantum_ring_trained | -0.4511 | [-0.4888, -0.4141] | yes | 1250 |
| 10 | kmeans_distance - quantum_ring_trained | -0.4358 | [-0.4750, -0.3965] | yes | 1250 |
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
| 12 | sweep_assignment - quantum_ring_trained | -0.7038 | [-0.7408, -0.6660] | yes | 1250 |
| 12 | kmeans_distance - quantum_ring_trained | -0.5311 | [-0.5707, -0.4896] | yes | 1250 |
| 16 | distance_pair - random_assignment | -0.8231 | [-0.8633, -0.7838] | yes | 1250 |
| 16 | distance_pair - sweep_assignment | 0.6411 | [0.5983, 0.6822] | yes | 1250 |
| 16 | distance_pair - kmeans_distance | 0.1667 | [0.1259, 0.2078] | yes | 1250 |
| 16 | kmeans_distance - random_assignment | -0.9897 | [-1.0310, -0.9471] | yes | 1250 |
| 16 | sweep_assignment - random_assignment | -1.4642 | [-1.5041, -1.4247] | yes | 1250 |
| 16 | classical_esn - distance_pair | 0.7000 | [0.6615, 0.7406] | yes | 1250 |
| 16 | quantum_ring_trained - distance_pair | 0.5240 | [0.4874, 0.5621] | yes | 1250 |
| 16 | quantum_shallow_trained - distance_pair | 0.5346 | [0.4979, 0.5725] | yes | 1250 |
| 16 | quantum_ring_random - random_assignment | -0.2513 | [-0.2899, -0.2157] | yes | 1250 |
| 16 | quantum_ring_trained - random_assignment | -0.2990 | [-0.3352, -0.2608] | yes | 1250 |
| 16 | quantum_ring_trained - classical_esn | -0.1760 | [-0.2115, -0.1389] | yes | 1250 |
| 16 | quantum_shallow_trained - classical_esn | -0.1655 | [-0.2023, -0.1298] | yes | 1250 |
| 16 | quantum_ring_trained - quantum_ring_random | -0.0477 | [-0.0833, -0.0126] | yes | 1250 |
| 16 | sweep_assignment - quantum_ring_trained | -1.1651 | [-1.2068, -1.1248] | yes | 1250 |
| 16 | kmeans_distance - quantum_ring_trained | -0.6907 | [-0.7335, -0.6505] | yes | 1250 |

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
| 6 | classical_esn | 4.6095 | 4.6095 | -0.00% |
| 6 | distance_pair | 4.5011 | 4.5011 | 0.00% |
| 6 | kmeans_distance | 4.5451 | 4.5451 | -0.00% |
| 6 | quantum_ring_random | 4.6164 | 4.6164 | -0.00% |
| 6 | quantum_ring_trained | 4.6237 | 4.6237 | -0.00% |
| 6 | quantum_shallow_trained | 4.6325 | 4.6325 | -0.00% |
| 6 | random_assignment | 4.5913 | 4.5968 | 0.13% |
| 6 | sweep_assignment | 4.6083 | 4.6083 | -0.00% |
| 8 | classical_esn | 5.6545 | 5.7213 | 1.12% |
| 8 | distance_pair | 5.2883 | 5.3432 | 1.04% |
| 8 | kmeans_distance | 5.3596 | 5.4062 | 0.83% |
| 8 | quantum_ring_random | 5.6454 | 5.7085 | 1.06% |
| 8 | quantum_ring_trained | 5.6330 | 5.6994 | 1.11% |
| 8 | quantum_shallow_trained | 5.6155 | 5.6850 | 1.17% |
| 8 | random_assignment | 5.6240 | 5.7082 | 1.38% |
| 8 | sweep_assignment | 5.3744 | 5.4157 | 0.71% |
| 10 | classical_esn | 6.4161 | 6.7208 | 4.30% |
| 10 | distance_pair | 5.9246 | 6.2103 | 4.50% |
| 10 | kmeans_distance | 5.9421 | 6.1619 | 3.42% |
| 10 | quantum_ring_random | 6.3807 | 6.7445 | 5.20% |
| 10 | quantum_ring_trained | 6.3779 | 6.7162 | 4.83% |
| 10 | quantum_shallow_trained | 6.3929 | 6.7328 | 4.84% |
| 10 | random_assignment | 6.4711 | 6.7822 | 4.38% |
| 10 | sweep_assignment | 5.9269 | 6.1471 | 3.37% |
| 12 | classical_esn | 7.0720 | 7.7855 | 8.88% |
| 12 | distance_pair | 6.5077 | 7.1769 | 9.15% |
| 12 | kmeans_distance | 6.4395 | 6.9804 | 7.53% |
| 12 | quantum_ring_random | 6.9987 | 7.7556 | 9.49% |
| 12 | quantum_ring_trained | 6.9706 | 7.7871 | 10.17% |
| 12 | quantum_shallow_trained | 6.9910 | 7.7428 | 9.43% |
| 12 | random_assignment | 7.1301 | 7.8070 | 8.34% |
| 12 | sweep_assignment | 6.2668 | 6.7817 | 7.28% |
| 16 | classical_esn | 8.0992 | 9.9606 | 18.34% |
| 16 | distance_pair | 7.3992 | 9.0792 | 18.16% |
| 16 | kmeans_distance | 7.2325 | 8.6806 | 16.26% |
| 16 | quantum_ring_random | 7.9709 | 9.8958 | 19.11% |
| 16 | quantum_ring_trained | 7.9232 | 9.9102 | 19.71% |
| 16 | quantum_shallow_trained | 7.9337 | 9.9184 | 19.62% |
| 16 | random_assignment | 8.2222 | 9.9656 | 17.10% |
| 16 | sweep_assignment | 6.7580 | 8.0270 | 15.35% |
