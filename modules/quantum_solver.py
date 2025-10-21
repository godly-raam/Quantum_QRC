# modules/quantum_solver.py - FIXED VERSION

from qiskit_aer.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import logging
import time
from typing import List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SolutionMetrics:
    """Data structure for solution quality and metadata."""
    is_valid_quantum_solution: bool
    total_distance: float
    execution_time: float
    notes: str

def _calculate_route_distances(routes: List[List[int]], distance_matrix: np.ndarray) -> Tuple[List[float], float]:
    """Calculate total and per-route distances with robust error handling."""
    distances = []
    total_distance = 0.0
    for route in routes:
        try:
            route_distance = 0.0
            if len(route) >= 2:
                for i in range(len(route) - 1):
                    dist_value = distance_matrix[route[i], route[i+1]]
                    if isinstance(dist_value, np.ndarray):
                        dist_value = float(dist_value.item())
                    else:
                        dist_value = float(dist_value)
                    route_distance += dist_value
            distances.append(float(route_distance))
            total_distance += route_distance
        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating route distance: {e}")
            distances.append(0.0)
    return distances, float(total_distance)

def _create_classical_fallback(distance_matrix: np.ndarray, num_vehicles: int, depot_node: int) -> List[List[int]]:
    """Generate a simple greedy classical solution as fallback."""
    logger.warning("Quantum solver failed. Using classical greedy fallback.")
    num_locations = distance_matrix.shape[0] - 1
    locations = list(range(1, num_locations + 1))
    
    # Simple round-robin assignment
    routes = [[] for _ in range(num_vehicles)]
    for loc in locations:
        vehicle_idx = loc % num_vehicles
        routes[vehicle_idx].append(loc)
    
    # Add depot at start and end of each route
    final_routes = [[depot_node] + route + [depot_node] for route in routes if route]
    return final_routes

def solve_quantum_vrp(
    distance_matrix: np.ndarray, 
    num_vehicles: int, 
    depot_node: int = 0, 
    reps: int = 5
) -> Tuple[List[List[int]], List[float], SolutionMetrics]:
    """
    Adaptive quantum-classical VRP solver.
    
    FIXED: Qiskit Sampler API compatibility
    """
    start_time = time.time()
    
    # Calculate problem complexity
    num_locations = distance_matrix.shape[0]
    estimated_qubits = (num_locations - 1) * num_vehicles
    
    logger.info(f"Problem size: {num_locations} locations, {num_vehicles} vehicles (~{estimated_qubits} qubits)")
    
    try:
        # Create VRP problem and convert to QUBO
        vrp_problem = VehicleRouting(
            distance_matrix, 
            num_vehicles=num_vehicles, 
            depot=depot_node
        )
        qp = vrp_problem.to_quadratic_program()
        
        # ============================================
        # ADAPTIVE METHOD SELECTION - FIXED SAMPLER API
        # ============================================
        
        if estimated_qubits <= 12:
            # SMALL PROBLEMS: Exact statevector simulation
            logger.info("Strategy: EXACT statevector simulation (best accuracy)")
            
            # FIX: Use run_options instead of constructor parameters
            sampler = Sampler(run_options={"shots": None, "seed": 42})
            optimizer = COBYLA(maxiter=150)
            adjusted_reps = min(reps, 5)
            method_note = "exact statevector"
            
        elif estimated_qubits <= 18:
            # MEDIUM PROBLEMS: Shot-based sampling
            logger.info("Strategy: SHOT-BASED sampling (balanced accuracy/memory)")
            
            # FIX: Use run_options
            sampler = Sampler(run_options={"shots": 2048, "seed": 42})
            optimizer = COBYLA(maxiter=100)
            adjusted_reps = min(reps, 4)
            method_note = "sampling (2048 shots)"
            
        else:
            # LARGE PROBLEMS: Reduced shots
            logger.warning("Strategy: Reduced sampling (memory-constrained)")
            
            # FIX: Use run_options
            sampler = Sampler(run_options={"shots": 1024, "seed": 42})
            optimizer = SPSA(maxiter=80)
            adjusted_reps = min(reps, 3)
            method_note = "reduced sampling (1024 shots)"
        
        # ============================================
        # QAOA EXECUTION
        # ============================================
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=adjusted_reps,
            initial_point=np.random.uniform(0, 2 * np.pi, 2 * adjusted_reps)
        )
        
        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
        
        logger.info(f"Executing QAOA: {adjusted_reps} layers, {method_note}")
        result = eigen_optimizer.solve(qp)
        
        # ============================================
        # RESULT INTERPRETATION
        # ============================================
        
        try:
            routes = vrp_problem.interpret(result)
            logger.info(f"Raw quantum result: {routes}")
            
            if not routes or not any(routes):
                raise ValueError("Empty routes returned from quantum solver")
            
            # Format and validate routes
            formatted_routes = []
            for route in routes:
                if isinstance(route, (list, tuple, np.ndarray)):
                    formatted_route = [int(x) for x in route]
                    if formatted_route:
                        formatted_routes.append(formatted_route)
            
            if not formatted_routes:
                raise ValueError("No valid routes after formatting")
            
            routes = formatted_routes
            is_valid_quantum = True
            notes = f"QAOA solution ({method_note}, depth={adjusted_reps})"
            
        except Exception as interpret_error:
            logger.error(f"Result interpretation failed: {interpret_error}")
            raise interpret_error
    
    except Exception as e:
        logger.error(f"Quantum solver error: {e}. Activating classical fallback.")
        routes = _create_classical_fallback(distance_matrix, num_vehicles, depot_node)
        is_valid_quantum = False
        notes = f"Classical fallback (quantum error: {str(e)[:60]})"
    
    # ============================================
    # DISTANCE CALCULATION
    # ============================================
    
    try:
        distances, total_distance = _calculate_route_distances(routes, distance_matrix)
        logger.info(f"Route distances: {[f'{d:.2f}' for d in distances]}, total: {total_distance:.2f}")
    except Exception as dist_error:
        logger.error(f"Distance calculation failed: {dist_error}")
        distances = [0.0] * len(routes)
        total_distance = 0.0
    
    execution_time = time.time() - start_time
    
    metrics = SolutionMetrics(
        is_valid_quantum_solution=is_valid_quantum,
        total_distance=float(total_distance),
        execution_time=float(execution_time),
        notes=notes
    )
    
    logger.info(f"✓ Solution completed in {execution_time:.2f}s | Quantum: {is_valid_quantum}")
    
    return routes, distances, metrics