"""
Hardware Noise Resilience Benchmark for QRC Logistics Engine
Compares standard QAOA highly-entangled penalty circuits vs. Ancilla-Free LCU.

Author: Rambabu Singh (Entangle Minds Team)
Target: Proof of NISQ viability via IBM FakeKyiv Simulation
"""

import os
import sys
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the local LCU module.
from modules.lcu_constraint_flattener import (  # pylint: disable=wrong-import-position,import-error
    build_lcu_constraint_layer,
    sample_lcu_branch,
)


def build_standard_qaoa_penalty(
    num_qubits: int, target_k: int, gamma: float = 1.0
) -> QuantumCircuit:
    """
    Builds a standard QAOA cardinality penalty circuit: f(x) = (1^T x - k)^2.
    Requires dense O(N^2) ZZ-entangling gates.
    """
    qc = QuantumCircuit(num_qubits)

    # Encode the quadratic (ZZ) interaction terms
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qc.rzz(2 * gamma, i, j)

    # Encode the linear (Z) bias terms
    for i in range(num_qubits):
        qc.rz(gamma * (1 - 2 * target_k), i)

    return qc


def calculate_tvd(ideal_dist: dict[str, float], noisy_dist: dict[str, float]) -> float:
    """Calculates Total Variation Distance between two probability distributions."""
    tvd = 0.0
    all_keys = set(ideal_dist.keys()).union(set(noisy_dist.keys()))
    for key in all_keys:
        p_ideal = ideal_dist.get(key, 0.0)
        p_noisy = noisy_dist.get(key, 0.0)
        tvd += 0.5 * abs(p_ideal - p_noisy)
    return tvd


def create_fake_kyiv_backend():
    """Load the optional IBM Runtime fake backend with an actionable error."""
    try:
        fake_provider = import_module("qiskit_ibm_runtime.fake_provider")
        return fake_provider.FakeKyiv()
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FakeKyiv requires qiskit-ibm-runtime. Install the project dependencies "
            "with `pip install -r requirements.txt`."
        ) from error


def run_hardware_experiment():  # pylint: disable=too-many-locals,too-many-statements
    """Execute NISQ hardware noise resilience benchmark comparing QAOA and LCU."""
    print("Initializing IBM FakeKyiv Hardware Simulation...")
    num_qubits = 8
    target_k = 4
    rng = np.random.default_rng(42)

    # 1. Initialize Backend and Transpiler
    backend = create_fake_kyiv_backend()
    simulator = AerSimulator.from_backend(backend)
    ideal_simulator = AerSimulator()  # Noiseless baseline

    # Optimization Level 3 aggressively optimizes gates for the hardware topology
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)

    # 2. Build Circuits
    print(f"Building constraint layers for {num_qubits} qubits...")

    # Setup initial superposition (standard for QAOA/QRC inputs)
    init_qc = QuantumCircuit(num_qubits)
    init_qc.h(range(num_qubits))

    # A. Standard QAOA
    qaoa_layer = build_standard_qaoa_penalty(num_qubits, target_k)
    qaoa_full = init_qc.copy()
    qaoa_full.compose(qaoa_layer, inplace=True)
    qaoa_full.measure_all()

    # B. Proposed LCU (Single Branch)
    sampled_theta = sample_lcu_branch(num_qubits, target_k, rng=rng)
    lcu_layer = build_lcu_constraint_layer(num_qubits, sampled_theta)
    lcu_full = init_qc.copy()
    lcu_full.compose(lcu_layer, inplace=True)
    lcu_full.measure_all()

    # 3. Transpile to Hardware
    print("Transpiling to heavy-hex hardware topology...")
    qaoa_transpiled = pass_manager.run(qaoa_full)
    lcu_transpiled = pass_manager.run(lcu_full)

    # Log Physical Hardware Metrics (Crucial for the paper)
    metrics = {
        "QAOA": {
            "depth": qaoa_transpiled.depth(),
            "cnots": qaoa_transpiled.count_ops().get("cx", 0),
        },
        "LCU": {
            "depth": lcu_transpiled.depth(),
            "cnots": lcu_transpiled.count_ops().get("cx", 0),
        },
    }

    print("\n--- Physical Hardware Metrics ---")
    print(
        f"Standard QAOA Penalty -> Depth: {metrics['QAOA']['depth']}, "
        f"CNOTs: {metrics['QAOA']['cnots']}"
    )
    print(
        f"Proposed LCU Penalty  -> Depth: {metrics['LCU']['depth']}, "
        f"CNOTs: {metrics['LCU']['cnots']}"
    )

    # 4. Execute Ideal vs Noisy
    print("\nExecuting Shots (Shots = 10,000)...")
    shots = 10000

    # Ideal Executions
    qaoa_ideal_counts = (
        ideal_simulator.run(qaoa_full, shots=shots).result().get_counts()
    )
    lcu_ideal_counts = ideal_simulator.run(lcu_full, shots=shots).result().get_counts()

    # Noisy Executions
    qaoa_noisy_counts = (
        simulator.run(qaoa_transpiled, shots=shots).result().get_counts()
    )
    lcu_noisy_counts = simulator.run(lcu_transpiled, shots=shots).result().get_counts()

    # Normalize to probabilities
    qaoa_ideal_prob = {k: v / shots for k, v in qaoa_ideal_counts.items()}
    qaoa_noisy_prob = {k: v / shots for k, v in qaoa_noisy_counts.items()}
    lcu_ideal_prob = {k: v / shots for k, v in lcu_ideal_counts.items()}
    lcu_noisy_prob = {k: v / shots for k, v in lcu_noisy_counts.items()}

    # 5. Calculate Degradation (TVD)
    qaoa_tvd = calculate_tvd(qaoa_ideal_prob, qaoa_noisy_prob)
    lcu_tvd = calculate_tvd(lcu_ideal_prob, lcu_noisy_prob)

    print("\n--- Signal Degradation (Total Variation Distance) ---")
    print(f"QAOA Degradation: {qaoa_tvd:.4f} (Closer to 1.0 is worse - pure noise)")
    print(f"LCU Degradation:  {lcu_tvd:.4f} (Closer to 0.0 is better - pure signal)")

    # 6. Plotting the "Money Graph"
    labels = ["Standard QAOA Penalty", "Proposed LCU (Ancilla-Free)"]
    tvd_values = [qaoa_tvd, lcu_tvd]
    cnot_values = [metrics["QAOA"]["cnots"], metrics["LCU"]["cnots"]]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_ylabel("Signal Degradation (TVD)", color=color, fontweight="bold")
    ax1.bar(labels, tvd_values, color=["darkred", "forestgreen"], alpha=0.8, width=0.4)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Hardware CNOT Count (Heavy-Hex)", color=color, fontweight="bold")
    ax2.plot(
        labels,
        cnot_values,
        color=color,
        marker="o",
        markersize=10,
        linewidth=3,
        linestyle="dashed",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(
        "Hardware Resilience: Constraint Flattening on IBM Kyiv Simulator",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig("hardware_resilience_benchmark.png", dpi=300)
    print("\nBenchmark complete! Graph saved as 'hardware_resilience_benchmark.png'")


if __name__ == "__main__":
    # Requires the dependencies declared in requirements.txt.
    run_hardware_experiment()
