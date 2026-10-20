# Q-Fleet: Quantum Reservoir VRP Backend

## What this project actually shows

This project investigates whether a fixed-parameter quantum reservoir provides a practical advantage over classical methods for vehicle routing problems (VRP) and related combinatorial optimization tasks. The results collected so far are mixed, and are reported here without adjustment:

- On a QOBLIB VRP instance, the quantum-reservoir solver trades solution quality for adaptation speed (139% optimality gap vs. a classical baseline, but sub-second re-routing) — see the Benchmark Results section. QOBLIB's own authors note that current VRP instances in their library are solvable to classical optimality, so this result should be read as a latency benchmark, not a quantum-advantage claim.
- On QOBLIB's Independent Set problem class (a class the benchmark's authors specifically flag as near-term quantum-amenable), we report a direct comparison against IBM's published QAOA baseline on the same instance — see the Independent Set Benchmark section.
- A controlled ablation against a classical Echo State Network of matched size found the quantum reservoir performs worse on route cost at every size tested, and its expressivity collapses as qubit count grows — consistent with the barren plateau phenomenon — see the Classical Reservoir Ablation section.
- In the label-free pairwise benchmark, no reservoir method beat a simple distance-based baseline. At 12 qubits the quantum reservoir beat the random assignment and the matched ESN, but remained far behind sweep, k-means, and distance-pair baselines — see the Fair Label-Free Pairwise Benchmark section.

We have not yet found a result in which this architecture outperforms a matched classical baseline. We're documenting this openly because we think the negative results, and the process of catching and fixing the bugs that produced misleading earlier numbers, are as useful a record of the work as a positive result would have been. Real quantum-hardware execution (beyond classical simulation) is the next open step — see "Next Steps" below.

Quantum Reservoir Computing (QRC) trades solution quality for sub-second re-adaptation latency versus full re-optimization. It is intended as a dynamic re-planner that operates alongside, rather than replaces, a classical routing solver.

## Quick Start & Environment Setup

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/godly-raam/Quantum_QRC.git
cd Quantum_QRC
pip install -r requirements.txt
```

**2. Download the QOBLIB Benchmark Data**

Create the data directory and download the VRP instances:

```bash
mkdir -p data/QOBLIB/09-routing/instances
# Place the XSH-n20-k4-01.vrp dataset into this folder
```

**3. Generate the Offline Quantum Weights (Crucial)**

Before starting the API, you must pre-train and lock the reservoir dynamics:

```bash
python scripts/run_offline_training.py
```

**4. Start the Redis Server**

The API requires Redis for real-time state management:

```bash
docker run -p 6379:6379 redis
```

**5. Start the FastAPI Backend**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

This project is the backend API for Q-Fleet, a quantum-classical system for the Vehicle Routing Problem (VRP). Its core claim is operational rather than an optimality claim: QRC trades solution quality for sub-second re-adaptation latency versus a full re-optimization, while a classical solver remains responsible for full optimization.

[QOBLIB's Vehicle Routing benchmark](https://github.com/ZIB-AOPT/QOBLIB/tree/main/09-routing) flags VRP instances of this scale as classically solvable to optimality today. Accordingly, this benchmark should be read as a stress test of real-time adaptation latency, not as a claim of quantum advantage.

This approach is informed by research into quantum reservoirs, including work by QuEra Computing (arXiv:2407.02553v1).

## Key Features

* **Quantum Reservoir Computing (QRC):** A fixed, random quantum system acts as a "reservoir" to capture complex patterns. A classical readout layer is then trained to map reservoir states to optimal routes.
* **Real-Time Adaptation:** The QRC solver can adapt to new traffic jams or add priority deliveries in under a second.
* **Hybrid Solver:** The API can intelligently switch between the ultra-fast QRC solver and a traditional (but slower) QAOA solver for full-scale optimization.
* **Self-Training:** On startup, the API automatically generates synthetic VRP data and trains the QRC model, making it instantly ready for requests.


## Key API Endpoints

* `POST /api/optimize`: Solves a new VRP problem from scratch. Set `"use_qrc": true` in the request body to use the QRC solver.
* `POST /api/traffic-jam`: Takes a list of jammed locations and instantly returns a new, adapted route using QRC.
* `POST /api/priority-delivery`: Takes a new priority location and instantly returns a new route that includes it.
* `GET /api/health`: A simple health check endpoint used by Render to verify the service is running.
* `GET /`: Root endpoint with API status and documentation links.

## Benchmark Results
The following disclosure follows [QOBLIB's submission schema](https://github.com/ZIB-AOPT/QOBLIB/blob/main/CONTRIBUTING.md). The recorded 27-qubit result uses the `> 20`-qubit random-vector bypass; it is **classical pipeline validation only — not a quantum result** and must not be used as a QRC-versus-classical quality comparison.

| QOBLIB submission field | Recorded value |
| :--- | :--- |
| **Identifier** | `XSH-n20-k4-01.vrp` |
| **Submitter** | Entangle Minds Team |
| **Date** | Not recorded |
| **Reference** | This repository; `scripts/benchmark_qoblib.py` |
| **Best objective value** | 826.98 — **Classical pipeline validation only — not a quantum result** |
| **Optimality bound** | N/A; no bound was established |
| **Modeling approach** | Fixed 27-feature reservoir/readout workflow with topology padding; **Classical pipeline validation only — not a quantum result** because the `> 20`-qubit random-vector bypass was used |
| **# decision variables** | N/A; no QOBLIB optimization-model variable count was produced |
| **# binary variables** | N/A |
| **# integer variables** | N/A |
| **# continuous variables** | N/A |
| **# non-zero coefficients** | N/A; no QOBLIB model export was produced |
| **Coefficient type/range** | N/A |
| **Workflow description** | Parse and pad the instance, inject randomized feature vectors in place of a statevector, apply the classical readout and Pareto filtering; **Classical pipeline validation only — not a quantum result** |
| **Algorithm type** | Stochastic (random-vector bypass) |
| **# runs** | 1 recorded run |
| **# feasible runs** | Not evaluated with the QOBLIB feasibility checker |
| **# successful runs + success threshold ε** | N/A; no success criterion was evaluated |
| **Hardware specification** | Not recorded; no QPU was used |
| **Total runtime** | 0.4643 s |
| **CPU runtime** | Not separately measured |
| **GPU runtime** | N/A |
| **QPU runtime** | N/A — no QPU execution; **Classical pipeline validation only — not a quantum result** |
| **Other runtime** | Not separately measured |

> **Note on Local Execution:** When simulating a reservoir > 20 qubits using a classical `Statevector`, memory overhead exceeds standard hardware capacities (e.g., $2^{27}$ requires tracking 134M amplitudes). To ensure local benchmark testing completes cleanly, a safety bypass activates for reservoirs `> 20` to inject randomized feature vectors. This bypass purely validates the classical optimization and Pareto filtering mechanics. It should be disabled when deploying the engine to a high-memory computing cluster or a native quantum device.

**Note on the live `/api/optimize` endpoint's classical-fallback path:** Prior to 2026-08-06, the QAOA branch used by this endpoint when QRC is unavailable, untrained, or rejects a problem as too large (`modules/quantum_solver.py`) failed silently due to a Qiskit Aer Sampler/SamplerV2 API incompatibility, and transparently fell back to an OR-Tools route on every call — correctly reporting `IS_QUANTUM=False` in each case, so no result was ever mislabeled as quantum. This affected only this specific endpoint branch; it did not affect the QOBLIB benchmark results documented above, the Independent Set benchmark, or the classical reservoir ablation, none of which import `modules/quantum_solver.py` (see commit `5aeff470` for the fix and verification).

## Fair Label-Free Pairwise Benchmark

The earlier ablations used a flat/node readout trained on route-centroid angles. Those labels depend on absolute coordinates, so a readout can recover geometry without learning a route-invariant assignment. The benchmark below replaces that target with a pairwise co-assignment label: two customers are positive if and only if OR-Tools puts them on the same route. This target is invariant to vehicle permutation and rigid motion, and the reservoir pair features contain no explicit customer-to-customer distance term; any distance information reaches them only through the fixed encoding.

Design: synthetic capacitated VRP instances with `n_locations == n_qubits`, four vehicles, and a capacity that requires roughly all four vehicles; 200 training and 250 test instances per outer seed; five outer seeds per size; identical `nn2opt` decoder for every method; paired bootstrap 95% CIs over the 1,250 test runs per size. The reference is OR-Tools with a 1 s time limit. N=4 is excluded because it has only three customers for four vehicles. Raw runs and comparisons are in `results/aggregate/`.

| Size (qubits) | Seeds | random | sweep | k-means | distance-pair | ESN | ring random | ring trained | shallow trained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 5 | 4.5913 | 4.6083 | 4.5451 | **4.5011** | 4.6095 | 4.6164 | 4.6237 | 4.6325 |
| 8 | 5 | 5.6240 | 5.3744 | 5.3596 | **5.2883** | 5.6545 | 5.6454 | 5.6330 | 5.6155 |
| 10 | 5 | 6.4711 | 5.9269 | 5.9421 | **5.9246** | 6.4161 | 6.3807 | 6.3779 | 6.3929 |
| 12 | 5 | 7.1301 | **6.2668** | 6.4395 | 6.5077 | 7.0720 | 6.9987 | 6.9706 | 6.9910 |
| 16 (provisional) | 1 | 8.2218 | **6.7291** | 7.2706 | 7.3985 | 8.1284 | 7.9646 | 7.9596 | 8.0066 |

Mean route cost, lower is better. N=16 has only seed 0 and is not used for any claim.

Key paired differences (quantum and best classical methods):

| Size | Comparison | Mean difference | 95% CI | Excludes 0 |
|---:|---|---:|---|---|
| 6 | ring trained − random | +0.0323 | [+0.0104, +0.0546] | yes (worse) |
| 8 | ring trained − random | +0.0089 | [−0.0208, +0.0419] | no |
| 8 | ring trained − ESN | −0.0215 | [−0.0516, +0.0088] | no |
| 8 | distance-pair − ring trained | −0.3447 | [−0.3781, −0.3112] | yes |
| 8 | sweep − ring trained | −0.2586 | [−0.2894, −0.2279] | yes |
| 10 | ring trained − random | −0.0932 | [−0.1280, −0.0564] | yes |
| 10 | ring trained − ESN | −0.0382 | [−0.0737, −0.0003] | yes |
| 10 | distance-pair − ring trained | −0.4533 | [−0.4908, −0.4156] | yes |
| 10 | sweep − ring trained | −0.4511 | [−0.4888, −0.4141] | yes |
| 12 | ring trained − random | −0.1594 | [−0.1948, −0.1241] | yes |
| 12 | ring trained − ESN | −0.1014 | [−0.1359, −0.0657] | yes |
| 12 | distance-pair − ring trained | −0.4629 | [−0.4999, −0.4269] | yes |
| 12 | sweep − ring trained | −0.7038 | [−0.7399, −0.6681] | yes |

**Findings:**

- No quantum advantage was found. At 6 qubits the trained reservoir is significantly worse than random assignment, and at 8 qubits it is statistically indistinguishable from random assignment and from the matched ESN. Every distance-based classical baseline is significantly better at both sizes.
- At 10 and 12 qubits the reservoir does learn something: it beats random assignment and the matched ESN with CIs excluding zero. It still loses to sweep by 0.45 and 0.70, and to distance-pair by 0.45 and 0.46, all with CIs excluding zero. At 10 qubits the best classical baselines (sweep, distance-pair, k-means) are statistically tied with each other and clearly ahead of the reservoir.
- The best cost at every size with five seeds comes from a simple classical method, not from the reservoir. The reservoir's edge over the ESN at 10 and 12 qubits is therefore not evidence of a useful quantum feature map; it is a small improvement inside a group that is already far behind classical routing heuristics.
- The N=16 seed-0 result points the same way, but is provisional because only one seed exists.

### Readout-only controls

These controls reuse the same cached reservoir features and the same training targets. Only the readout inputs change, so they isolate whether the quantum feature content helps.

| Size | quantum (ZZ) | quantum (no ZZ) | ESN pair products | distance-pair | distance + `D_ij` |
|---:|---:|---:|---:|---:|---:|
| 8 | 5.6447 | 5.6375 | 5.6211 | 5.2874 | **5.1088** |
| 12 | 6.9866 | 7.0315 | 7.0667 | 6.5098 | **5.8614** |

Removing all ZZ features does not hurt at 8 qubits and costs only 0.045 at 12 qubits, so the entangling features are not carrying the result. Adding explicit customer-to-customer distances to the distance-pair control gives the best cost at both sizes, well ahead of every reservoir variant. The angle-scale validation selected `1/sqrt(8)` at N=8 and `1.0` at N=12; the selected-scale test costs (5.648 and 6.987) are within noise of the default, so angle tuning does not change the conclusion. Per-seed controls are in `results/pairwise_diagnostics/`.

### Decoder ceiling

To separate readout error from decoder error, OR-Tools routes were converted to vehicle labels and decoded again with the same `nn2opt` decoder. The mean re-decoded gap is 0.013% at N=8 and 0.009% at N=12 (maximum 3.2% and 0.9%). OR-Tools uses three non-empty routes on 31/250 N=8 cases and 7/250 N=12 cases; the decoder already handles fewer than four routes when the labels contain fewer. Index ordering instead of `nn2opt` costs 3.1% at N=8 and 14.5% at N=12, which is why `nn2opt` is used for every method. The decoder is therefore not the source of the learned-method gap; the co-assignment predictions are.

### Decision rule

A quantum advantage would be reported only if the paired 95% CI of (quantum − best classical baseline) excluded zero on at least 50 test instances, at least 5 seeds, and at least 3 sizes. No comparison in this benchmark meets that rule. Negative and inconclusive results are reported as such.

## Classical Reservoir Ablation

> **Superseded for label-free comparisons.** The ablation below uses the legacy flat/node readout with coordinate-dependent route-centroid labels. It is retained for provenance only; use the fair label-free benchmark above. The pairwise target and readout change is in commit `f0e402c1`.

Before attributing any result to "quantum" advantage, we tested whether the quantum reservoir provides any benefit over a classical Echo State Network (ESN) of matched feature dimension, using the same decoder and identical synthetic VRP instances (`scripts/benchmark_classical_reservoir.py`). This is not a claim of quantum advantage — it isolates whether the quantum feature map is doing anything a classical random projection can't.

| Qubits | Classical ESN expressivity | Quantum reservoir expressivity | Classical route cost | Quantum route cost | Quantum vs. classical |
|---|---|---|---|---|---|
| 4 | 0.141 | 0.714 | 2.455 | 2.595 | +5.7% worse |
| 8 | 0.830 | 0.976 | 4.826 | 5.491 | +13.8% worse |
| 12 | 1.615 | 0.364 | 6.759 | 7.949 | +17.6% worse |
| 16 | 4.321 | 0.178 | 8.845 | 9.759 | +10.3% worse |
| 20 | 5.216 | 0.118 | 11.209 | 12.078 | +7.8% worse |

**Findings, stated plainly:**

- The quantum reservoir is **worse than the classical ESN on route cost at every size tested**, with no exceptions.
- The classical ESN's expressivity grows smoothly and monotonically with size (0.141 → 5.216). The quantum reservoir's expressivity peaks around 8 qubits (0.976) and then **collapses monotonically** through 12, 16, and 20 qubits (down to 0.118) — even after fixing a training bug that had previously made this measurement meaningless (see commit 70e7efec).
- This collapse pattern — expectation values of local observables concentrating toward zero as circuit width grows — is consistent with the **barren plateau phenomenon** documented in variational quantum circuits (McClean, J. R. et al. "Barren plateaus in quantum neural network training landscapes." *Nature Communications* 9, 4812 (2018)). We have not confirmed this is the specific mechanism at play here; it is offered as the most likely explanation consistent with the observed trend, not a proven diagnosis.
- We do not currently have a result in which the quantum reservoir outperforms the matched classical baseline on either metric. This ablation should be read as an open problem for the project's core architecture, not a settled negative — barren-plateau mitigation strategies (e.g. problem-informed ansatz design, layerwise training, or restricting to shallower/more local entangling structure) are a concrete direction for future work rather than a dead end.

Raw JSON outputs for all five runs are available via `scripts/benchmark_classical_reservoir.py --qubits {4,8,12,16,20} --instances 20 --vehicles 4`.

### Ansatz comparison at small scale (4 qubits)

> **Superseded for label-free comparisons.** This table also uses the legacy flat readout and coordinate-dependent route-centroid labels; it is retained for provenance only. See the fair label-free benchmark above.

The ring-topology results above use a specific entangling pattern (global CZ ring, 2 layers). We tested whether a shallower, purely local-entanglement ansatz (1 RX layer + nearest-neighbor-only CZ, no ring closure — see `build_shallow_local_reservoir()` in `modules/reservoir_trainer.py`) changes the picture, across 5 seeds at 4 qubits:

| Seed | Shallow-local route cost | Ring route cost | Classical ESN route cost | Shallow-local vs. classical |
|---|---|---|---|---|
| 1 | 2.393 | 2.668 | 2.425 | 1.3% better |
| 7 | 2.398 | 2.801 | 2.492 | 3.8% better |
| 42 | 2.287 | 2.595 | 2.455 | 6.8% better |
| 99 | 2.336 | 2.625 | 2.424 | 3.6% better |
| 123 | 2.571 | 3.081 | 2.479 | 3.7% worse |

**Findings:**

- The shallow-local ansatz outperformed the classical ESN in 4 of 5 seeds at 4 qubits (worse in 1/5). The ring ansatz was worse than the classical ESN in all 5 seeds, by consistently larger margins.
- This is the only regime found so far (across all ansätze and all qubit counts 4–20 tested) where a quantum-reservoir variant is competitive with or ahead of the classical baseline on route cost.
- The advantage does not persist to larger sizes: shallow-local's expressivity collapses faster than ring's as qubit count grows (see table below), so whatever helps at 4 qubits does not scale.
- **Methodological limitation**: `train_reservoir_offline()` currently fixes its own internal training RNG to a constant seed (42) regardless of the `--seed` flag used for the outer benchmark. The 5-seed sweep above varies the synthetic VRP instances, ESN initialization, and quantum sampling — but not the trained ansatz parameters themselves, which stay identical across all 5 runs. A fully independent confirmation would also vary the trainer seed. We report this result with that caveat rather than treating it as fully established; it is evidence the effect is not purely a one-off, not proof of a robust general advantage.

| Qubits | Shallow-local expressivity | Ring expressivity | Classical ESN expressivity |
|---|---|---|---|
| 4 | 0.313 | 0.714 | 0.141 |
| 8 | 0.144 | 0.976 | 0.830 |
| 12 | 0.049 | 0.364 | 1.615 |
| 16 | 0.021 | 0.178 | 4.321 |
| 20 | 0.011 | 0.118 | 5.216 |
