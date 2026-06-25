import time
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vrp_parser import parse_vrp_instance
from modules.quantum_reservoir_vrp import ReservoirVRPSolver

def pad_cost_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pads the logistics graph matrix with zeros to match the fixed hardware reservoir size.
    """
    current_size = matrix.shape[0]
    if current_size == target_size:
        return matrix
    elif current_size > target_size:
        raise ValueError(f"Graph size ({current_size}) exceeds reservoir size ({target_size}). Truncation not supported.")
        
    padded_matrix = np.zeros((target_size, target_size))
    padded_matrix[:current_size, :current_size] = matrix
    return padded_matrix

def calculate_2d_hypervolume(pareto_front: List[Tuple[float, float]], reference_point: Tuple[float, float]) -> float:
    """
    Calculates the Hypervolume (HV) for a 2D Pareto front (e.g., Fuel vs. Time).
    Assumes minimization for both objectives.
    """
    # Sort the front by the first objective (Fuel) ascending
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    hv = 0.0
    # The previous y-boundary is initially the reference point's y-value
    prev_y = reference_point[1]
    
    for point in sorted_front:
        x, y = point
        # Ensure the point strictly dominates the reference point
        if x < reference_point[0] and y < prev_y:
            area = (reference_point[0] - x) * (prev_y - y)
            hv += area
            prev_y = y # Drop the y-boundary for the next slice
            
    return hv

def compute_greedy_baseline(Q_fuel: np.ndarray, num_vehicles: int) -> float:
    """Calculates a valid upper-bound classical baseline using a Nearest Neighbor heuristic."""
    num_nodes = len(Q_fuel)
    unvisited = set(range(1, num_nodes))
    total_cost = 0.0

    for _ in range(num_vehicles):
        curr = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: Q_fuel[curr][x] if Q_fuel[curr][x] > 0 else float('inf'))
            total_cost += Q_fuel[curr][next_node]
            unvisited.remove(next_node)
            curr = next_node
        total_cost += Q_fuel[curr][0] # Return to depot
    return total_cost

def run_qoblib_benchmark(filepath: str, reservoir_size: int = 27):
    print(f"--- Starting Benchmark: {filepath} ---")
    
    # 1. Parse Data
    Q_fuel_raw, Q_time_raw = parse_vrp_instance(filepath)
    num_nodes = len(Q_fuel_raw)
    num_vehicles = 4 # Or extract from the VRP filename k-value
    
    # 1. DYNAMIC BASELINES (Fixing the hardcoded fraud)
    max_possible_fuel = np.sum(Q_fuel_raw)
    max_possible_time = np.sum(Q_time_raw)
    reference_point = (max_possible_fuel * 1.1, max_possible_time * 1.1)
    
    classical_baseline_cost = compute_greedy_baseline(Q_fuel_raw, num_vehicles)
    
    print(f"Parsed {num_nodes} nodes. Padding to {reservoir_size}-qubit hardware topology.")
    
    # Pad to match the offline-trained reservoir
    Q_fuel = pad_cost_matrix(Q_fuel_raw, reservoir_size)
    Q_time = pad_cost_matrix(Q_time_raw, reservoir_size)
    
    # Load your locked Phase 3 weights
    locked_params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", f"locked_reservoir_params_{reservoir_size}q.npy")
    if not os.path.exists(locked_params_path):
        raise FileNotFoundError(f"Missing Phase 3 weights at {locked_params_path}. Run run_offline_training.py first.")
        
    locked_params = np.load(locked_params_path)
    
    # Initialize the solver wrapper with the FIXED reservoir size
    solver = ReservoirVRPSolver(n_reservoir_qubits=reservoir_size, trained_params=locked_params)
    
    # 2. Execution & Timing
    wall_clock_start = time.time()
    
    # We use fewer iterations if it's 27 qubits because the statevector is large!
    # The instructions said `iterations=100`, let's run 5 to avoid OOM for testing
    iterations = 100
    print(f"Running solver for {iterations} iterations...")
    pareto_front, qpu_time = solver.solve_multi_objective(Q_fuel, Q_time, iterations=iterations)
    
    wall_clock_end = time.time()
    total_wall_clock = wall_clock_end - wall_clock_start
    
    # 3. Metrics Calculation
    hv = calculate_2d_hypervolume(pareto_front, reference_point)
    best_fuel_cost = min([pt[0] for pt in pareto_front]) if pareto_front else float('inf')
    optimality_gap = ((best_fuel_cost - classical_baseline_cost) / classical_baseline_cost) * 100

    # 4. QOBLIB Standardized Markdown Output
    print("\n### QOBLIB Submission Metrics ###")
    print("| Metric | Value |")
    print("| :--- | :--- |")
    print(f"| **Instance** | {filepath.split('/')[-1]} |")
    print(f"| **Active Nodes / Reservoir Size** | {num_nodes} / {reservoir_size} |")
    print(f"| **Pareto Front Size** | {len(pareto_front)} non-dominated solutions |")
    print(f"| **Hypervolume (HV)** | {hv:.4f} |")
    print(f"| **Total Wall-Clock Time** | {total_wall_clock:.4f} s |")
    print(f"| **Isolated QPU Time** | {qpu_time:.4f} s |")
    print(f"| **Best Fuel Cost (QRC)** | {best_fuel_cost:.2f} |")
    print(f"| **Classical Baseline** | {classical_baseline_cost:.2f} |")
    print(f"| **Absolute Optimality Gap** | {optimality_gap:.2f}% |")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "QOBLIB", "09-routing", "instances", "XSH-n20-k4-01.vrp")
    
    run_qoblib_benchmark(
        filepath=benchmark_file, 
        reservoir_size=27
    )
