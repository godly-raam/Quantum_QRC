# Q-Fleet: Quantum Reservoir VRP Backend

**Five-line results summary**

- N=8: no detectable reservoir signal over random assignment.
- N=10: quantum features improve route cost by about 0.02–0.05 over decoder-matched nulls, but the comparison is marginal and not significant against the Gaussian null.
- N=12: quantum features improve route cost by about 0.07–0.10 over decoder-matched nulls, with paired 95% CIs excluding zero.
- The decoder pipeline alone beats random assignment by about 0.03–0.10; matched ESN features do not beat the nulls.
- Reservoirs remain 0.45–1.17 behind the stronger classical heuristics at N=10–16; these circuits are exactly classically simulable through N=16, so no quantum-advantage claim is testable here.

## What this project actually shows

This project investigates whether a fixed-parameter quantum reservoir provides a practical advantage over classical methods for vehicle routing problems (VRP) and related combinatorial optimization tasks. The results collected so far are mixed, and are reported here without adjustment:

- On a QOBLIB VRP instance, the quantum-reservoir solver trades solution quality for adaptation speed (139% optimality gap vs. a classical baseline, but sub-second re-routing) — see the Benchmark Results section. QOBLIB's own authors note that current VRP instances in their library are solvable to classical optimality, so this result should be read as a latency benchmark, not a quantum-advantage claim.
- On QOBLIB's Independent Set problem class (a class the benchmark's authors specifically flag as near-term quantum-amenable), we report a direct comparison against IBM's published QAOA baseline on the same instance — see the Independent Set Benchmark section.
- A controlled ablation against a classical Echo State Network of matched size found the quantum reservoir performs worse on route cost at every size tested, and its expressivity collapses as qubit count grows — consistent with the barren plateau phenomenon — see the Classical Reservoir Ablation section.
- In the label-free pairwise benchmark, decoder-matched null controls show a quantum-feature improvement at N=12 and a smaller, marginal improvement at N=10; neither changes the conclusion that reservoir methods remain far behind the strongest classical baselines — see the Fair Label-Free Pairwise Benchmark section.

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
| 16 | 5 | 8.2222 | **6.7580** | 7.2325 | 7.3992 | 8.0992 | 7.9709 | 7.9232 | 7.9337 |

Mean route cost, lower is better. All sizes shown have five seeds. N=16 is a secondary size because its simulation is much slower, and it is not used for any advantage claim.

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
| 16 | ring trained − random | −0.2990 | [−0.3352, −0.2608] | yes |
| 16 | ring trained − ESN | −0.1760 | [−0.2115, −0.1389] | yes |
| 16 | distance-pair − ring trained | −0.5240 | [−0.5621, −0.4874] | yes |
| 16 | sweep − ring trained | −1.1651 | [−1.2068, −1.1248] | yes |

**Findings:**

- No quantum advantage was found. At 6 qubits, the trained reservoir is significantly worse than random assignment; k-means and distance-pair significantly beat it, while sweep is statistically indistinguishable from it. At 8 qubits, the reservoir is statistically indistinguishable from random assignment and the matched ESN, while the distance-based baselines beat it.
- Relative to decoder-matched nulls (constant, shuffled and Gaussian features run through the same readout, clustering and capacity-repair decoder), the quantum features reduce route cost by 0.07–0.10 at N=12 (paired 95% CIs exclude 0) and 0.02–0.05 at N=10 (marginal; not significant vs the Gaussian null); there is no signal at N=8. The matched ESN's features do not beat any null. Random assignment is not a valid reference: the decoder alone, via capacity repair and tie breaking, beats it by 0.03–0.10. All reservoir variants remain 0.45–1.17 behind sweep, k-means and distance-pair at N=10–16. At N≤16 the circuits are exactly classically simulable, so no quantum-advantage claim is testable here.
- The best cost at every size with five seeds comes from a simple classical method, not from the reservoir. The N=10 ring-trained versus matched-ESN comparison is borderline: its 95% CI only barely excludes zero. The null-control gains are small relative to the gap to classical routing heuristics.
- At 16 qubits the pattern repeats: the reservoir beats random assignment and the ESN, but loses to k-means by 0.69, distance-pair by 0.52, and sweep by 1.17, all with CIs excluding zero.

### Readout-only controls

These controls reuse the same cached reservoir features and the same training targets. Only the readout inputs change, so they isolate whether the quantum feature content helps.

| Size | quantum (ZZ) | quantum (no ZZ) | ESN pair products | distance-pair | distance + `D_ij` |
|---:|---:|---:|---:|---:|---:|
| 8 | 5.6447 | 5.6375 | 5.6211 | 5.2874 | **5.1088** |
| 12 | 6.9866 | 7.0315 | 7.0667 | 6.5098 | **5.8614** |

Removing all ZZ features does not hurt at 8 qubits and costs only 0.045 at 12 qubits, so the entangling features are not carrying the result. Adding explicit customer-to-customer distances to the distance-pair control gives the best cost at both sizes, well ahead of every reservoir variant. The angle-scale validation selected `1/sqrt(8)` at N=8 and `1.0` at N=12; the selected-scale test costs (5.648 and 6.987) are within noise of the default, so angle tuning does not change the conclusion. Per-seed controls are in `results/pairwise_diagnostics/`.

### Decoder ceiling

To separate readout error from decoder error, OR-Tools routes were converted to vehicle labels and decoded again with the same `nn2opt` decoder. The mean re-decoded gap is 0.013% at N=8 and 0.009% at N=12 (maximum 3.2% and 0.9%). OR-Tools uses three non-empty routes on 31/250 N=8 cases and 7/250 N=12 cases; the decoder already handles fewer than four routes when the labels contain fewer. Index ordering instead of `nn2opt` costs 3.1% at N=8 and 14.5% at N=12, which is why `nn2opt` is used for every method. The decoder is therefore not the source of the learned-method gap; the co-assignment predictions are.

### Shot-noise ablation

The cached 1.0-angle ring-reservoir features were resampled with binomial shot noise. For each Pauli expectation `o`, the estimate is `2 * Binomial(S, (1 + o) / 2) / S - 1`, with variance `(1 - o^2) / S`. Both training and test features are noised independently, the pair readout is retrained, and results are averaged over five noise repetitions per outer seed. Full details and all shot counts are in `results/shot_noise/`.

Paired cost difference versus random assignment (negative is better than random):

| Size | S=100 | S=1,000 | S=10,000 | S=100,000 |
|---:|---|---|---|---|
| 8 | +0.0219 [−0.0037, +0.0472] | +0.0131 [−0.0142, +0.0387] | +0.0141 [−0.0148, +0.0432] | +0.0132 [−0.0162, +0.0429] |
| 10 | −0.0486 [−0.0783, −0.0205] | −0.0743 [−0.1054, −0.0440] | −0.0849 [−0.1193, −0.0511] | −0.0908 [−0.1259, −0.0547] |
| 12 | −0.0749 [−0.1024, −0.0476] | −0.0815 [−0.1110, −0.0520] | −0.1211 [−0.1531, −0.0896] | −0.1276 [−0.1615, −0.0940] |

The edge over random does **not** disappear at any requested shot count. At the low end, N=10 is not significant at S=1 (−0.0230 [−0.0491, +0.0047]) or S=3 (−0.0226 [−0.0485, +0.0059]), becomes significant at S=10, and stays significant through S=100,000. N=12 is already significant at S=1 (−0.0487 [−0.0760, −0.0212]). The edge therefore does not vanish monotonically: N=10 lacks significance only in the extreme 1–3 shot regime, and N=12 never vanishes in the tested range.

Against the matched ESN the picture is more fragile: N=10 is significant only at S=100,000 (−0.0364 [−0.0726, −0.0005]); N=12 becomes significant at S=10,000 and 100,000 (−0.0552 and −0.0618). At S≤3 the reservoir is worse than the matched ESN at N=10; from S=10 through S=1,000 the difference is not significant at either N=10 or N=12.

The null-control comparison shows that the readout-plus-decoder pipeline itself accounts for part of the improvement over random assignment. At N=10, the information-free shuffled, Gaussian, and constant controls improve mean cost over random by 0.047–0.070; at N=12 they improve it by 0.071–0.097. The quantum features improve on those controls by a further 0.023–0.046 at N=10 and 0.068–0.095 at N=12. At N=8, comparisons against the null controls show no clear signal. In the shot-noise ablation, this additional quantum-feature signal is only statistically distinguishable from the matched ESN at high sampling levels: S=100,000 at N=10, and S=10,000 or more at N=12. Thus the measured edge is not simply attributable to quantum features, and the remaining feature-dependent component has a substantial sampling cost. Full paired comparisons are in `results/null_control/null_control.md` and the shot-count analysis in `results/shot_noise/`.

The requested larger classical baselines, both using the same total feature count `3N + N(N-1)/2`, did not turn out to be stronger than the matched ESN at these sizes (all trained on the same target routes):

| Size | matched ESN | large ESN (3 recurrent steps) | random Fourier features |
|---:|---:|---:|---:|
| 8 | 5.6187 | 5.6198 | 5.6534 |
| 10 | 6.4167 | 6.4091 | 6.4386 |
| 12 | 7.0642 | 7.0757 | 7.0615 |

All three are still far behind the sweep and distance baselines in the main table (for example, N=12 sweep is 6.2668). The shot-noise result therefore does not justify QPU time: the edge that survives is an edge over random assignment (and, at high shot counts, over these particular random-feature baselines), not over the best classical methods. The limiting factor is the feature/readout quality, not sampling noise.

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

| Qubits | Shallow-local expressivity | Ring expressivity | Classical ESN expressivity |
|---|---|---|---|
| 4 | 0.313 | 0.714 | 0.141 |
| 8 | 0.144 | 0.976 | 0.830 |
| 12 | 0.049 | 0.364 | 1.615 |
| 16 | 0.021 | 0.178 | 4.321 |
| 20 | 0.011 | 0.118 | 5.216 |
