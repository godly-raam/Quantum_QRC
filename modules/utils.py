# modules/utils.py
"""
Utility functions for Q-Fleet QRC Backend
"""

import numpy as np
import requests
import time
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def get_osrm_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Fetch real-world road distances using the OSRM public API.
    
    Args:
        coords: Array of [lat, lon] coordinates
    
    Returns:
        Distance matrix in KILOMETERS (to match your synthetic scale)
    """
    # Note: coords are [lat, lon], so we flip them for OSRM
    coordinates_str = ";".join([f"{c[1]},{c[0]}" for c in coords])
    
    url = f"http://router.project-osrm.org/table/v1/driving/{coordinates_str}?annotations=distance"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # OSRM returns meters. Convert to KM to match your current scale.
            dist_matrix_meters = np.array(data['distances'])
            dist_matrix_km = dist_matrix_meters / 1000.0 
            return dist_matrix_km
        else:
            logger.warning(f"OSRM API failed: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"OSRM connection failed: {e}")
        return None


def generate_distance_matrix(
    num_locations: int, 
    center: list = [16.5, 80.5],
    spread: float = 0.05, # Reduced spread so points are closer (better for routing)
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate VRP instance with REAL road distances.
    """
    np.random.seed(seed)
    
    # Generate coordinates around center (Vijayawada)
    coords = np.random.randn(num_locations, 2) * spread + center
    
    # 1. Try to get REAL road distances
    logger.info("🌍 Fetching real road distances from OSRM...")
    distance_matrix = get_osrm_distance_matrix(coords)
    
    # 2. Fallback to Euclidean if OSRM fails
    if distance_matrix is None:
        logger.warning("⚠️ Using Euclidean fallback distances")
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(i + 1, num_locations):
                # Approximate KM: 1 deg lat ~= 111km
                d = np.linalg.norm(coords[i] - coords[j]) * 111.0
                distance_matrix[i, j] = distance_matrix[j, i] = d
                
    return coords, distance_matrix


def calculate_route_distance(
    route: List[int],
    distance_matrix: np.ndarray,
    traffic_multipliers: np.ndarray = None
) -> float:
    """
    Calculate total distance for a single route.
    
    Args:
        route: List of location indices
        distance_matrix: NxN distance matrix
        traffic_multipliers: Optional traffic adjustment factors
    
    Returns:
        Total route distance
    """
    if traffic_multipliers is None:
        traffic_multipliers = np.ones_like(distance_matrix)
    
    adjusted_matrix = distance_matrix * traffic_multipliers
    
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += adjusted_matrix[route[i], route[i+1]]
    
    return float(total_distance)


def calculate_all_routes_distance(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    traffic_multipliers: np.ndarray = None
) -> Tuple[List[float], float]:
    """
    Calculate distances for all routes.
    
    Returns:
        (list of per-route distances, total distance)
    """
    route_distances = []
    total = 0.0
    
    for route in routes:
        dist = calculate_route_distance(route, distance_matrix, traffic_multipliers)
        route_distances.append(dist)
        total += dist
    
    return route_distances, total


def validate_routes(
    routes: List[List[int]],
    num_locations: int,
    depot: int = 0
) -> Tuple[bool, str]:
    """
    Validate that routes are feasible.
    
    Returns:
        (is_valid, error_message)
    """
    # Check all routes start and end at depot
    for i, route in enumerate(routes):
        if len(route) < 2:
            return False, f"Route {i} too short"
        if route[0] != depot:
            return False, f"Route {i} doesn't start at depot"
        if route[-1] != depot:
            return False, f"Route {i} doesn't end at depot"
    
    # Check all locations covered exactly once
    visited = set()
    for route in routes:
        for loc in route[1:-1]:  # Exclude depot
            if loc in visited:
                return False, f"Location {loc} visited multiple times"
            visited.add(loc)
    
    # Check all non-depot locations covered
    expected_locations = set(range(1, num_locations))
    if visited != expected_locations:
        missing = expected_locations - visited
        return False, f"Missing locations: {missing}"
    
    return True, "Valid"


def format_solution_response(
    routes: List[List[int]],
    coordinates: np.ndarray,
    distance_matrix: np.ndarray,
    traffic_multipliers: np.ndarray,
    execution_time: float,
    method: str,
    is_quantum: bool,
    notes: str
) -> Dict:
    """
    Format solution into standardized API response.
    """
    route_distances, total_distance = calculate_all_routes_distance(
        routes, distance_matrix, traffic_multipliers
    )
    
    return {
        "routes": routes,
        "distances": route_distances,
        "coordinates": coordinates.tolist(),
        "total_distance": total_distance,
        "solution_method": method,
        "execution_time": execution_time,
        "is_quantum_solution": is_quantum,
        "notes": notes
    }


def generate_traffic_scenario(
    distance_matrix: np.ndarray,
    jam_probability: float = 0.2,
    severity_range: Tuple[float, float] = (1.5, 3.0)
) -> np.ndarray:
    """
    Generate random traffic scenario for testing.
    
    Args:
        distance_matrix: Base distance matrix
        jam_probability: Probability of traffic on each route
        severity_range: (min, max) traffic multiplier
    
    Returns:
        Traffic multiplier matrix
    """
    n = distance_matrix.shape[0]
    traffic = np.ones((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < jam_probability:
                severity = np.random.uniform(*severity_range)
                traffic[i, j] = traffic[j, i] = severity
    
    return traffic


def estimate_fuel_savings(
    original_distance: float,
    optimized_distance: float,
    fuel_efficiency_km_per_liter: float = 16.0,
    fuel_price_per_liter: float = 90.0,
    operating_days_per_year: int = 300
) -> Dict[str, float]:
    """
    Estimate economic impact of optimization.
    
    Returns:
        Dictionary with savings metrics
    """
    distance_saved_per_trip = original_distance - optimized_distance
    distance_saved_annual = distance_saved_per_trip * operating_days_per_year
    
    fuel_saved_liters = distance_saved_annual / fuel_efficiency_km_per_liter
    money_saved_annual = fuel_saved_liters * fuel_price_per_liter
    
    # CO2 emission factor: 2.68 kg CO2 per liter of diesel
    co2_saved_kg = fuel_saved_liters * 2.68
    
    return {
        "distance_saved_km_per_year": distance_saved_annual,
        "fuel_saved_liters_per_year": fuel_saved_liters,
        "money_saved_rupees_per_year": money_saved_annual,
        "co2_reduced_kg_per_year": co2_saved_kg,
        "co2_reduced_tonnes_per_year": co2_saved_kg / 1000
    }


def benchmark_timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


class PerformanceMonitor:
    """Track API performance metrics."""
    
    def __init__(self):
        self.request_times = []
        self.request_counts = {"optimize": 0, "traffic_jam": 0, "priority_delivery": 0}
    
    def record_request(self, endpoint: str, duration: float):
        """Record request timing."""
        self.request_times.append(duration)
        if endpoint in self.request_counts:
            self.request_counts[endpoint] += 1
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.request_times:
            return {"message": "No requests recorded yet"}
        
        return {
            "total_requests": len(self.request_times),
            "avg_response_time": np.mean(self.request_times),
            "median_response_time": np.median(self.request_times),
            "p95_response_time": np.percentile(self.request_times, 95),
            "request_breakdown": self.request_counts
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()