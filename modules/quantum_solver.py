# modules/quantum_solver.py - FIXED VERSION

from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization.applications import VehicleRouting
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
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

def _create_classical_fallback(
    distance_matrix: np.ndarray, num_vehicles: int, depot_node: int
) -> List[List[int]]:
    """Solve the fallback VRP with OR-Tools' routing heuristic."""
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not 0 <= depot_node < distance_matrix.shape[0]:
        raise ValueError("depot_node is outside distance_matrix")
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be positive")
    if not np.isfinite(distance_matrix).all() or (distance_matrix < 0).any():
        raise ValueError("distance_matrix must contain finite, non-negative costs")

    logger.warning("Quantum solver failed. Using OR-Tools routing fallback.")
    # OR-Tools uses integer arc costs; millimetre-equivalent scaling preserves
    # the input's three decimal places while retaining the original matrix for reporting.
    integer_costs = np.rint(distance_matrix * 1_000).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(
        distance_matrix.shape[0], num_vehicles, depot_node
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(integer_costs[from_node, to_node])

    callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (  # pylint: disable=no-member
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC  # pylint: disable=no-member
    )
    search_parameters.local_search_metaheuristic = (  # pylint: disable=no-member
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH  # pylint: disable=no-member
    )
    search_parameters.time_limit.FromSeconds(1)

    assignment = routing.SolveWithParameters(search_parameters)
    if assignment is None:
        raise RuntimeError("OR-Tools could not construct a feasible routing solution")

    routes: List[List[int]] = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = assignment.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        if len(route) > 2:
            routes.append(route)

    return routes

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
            sampler=sampler,  # type: ignore
            optimizer=optimizer,
            reps=adjusted_reps,
            initial_point=np.random.uniform(0, 2 * np.pi, 2 * adjusted_reps)
        )
        
        eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)  # type: ignore
        
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
                    formatted_route = [int(x) for x in route]  # type: ignore
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