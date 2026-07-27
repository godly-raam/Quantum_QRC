# Q-Fleet: Quantum Reservoir VRP Backend

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
