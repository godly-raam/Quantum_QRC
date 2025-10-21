# main.py - QRC-Enhanced Backend with Real-Time Adaptation

import os
import sys
import logging
import time
from typing import Optional, Dict, Any, List
sys.path.append('./modules')

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from modules import quantum_solver
from modules.quantum_reservoir_vrp import ReservoirVRPSolver, generate_synthetic_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Q-Fleet API with Quantum Reservoir Computing",
    description="Revolutionary quantum-classical hybrid VRP solver with real-time adaptation",
    version="2.0.0"
)

# CORS configuration - CRITICAL for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Additional CORS handler for edge cases
@app.middleware("http")
async def enhanced_cors_handler(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            content={"message": "OK"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "3600",
            }
        )
    
    try:
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )

# ============================================================
# GLOBAL STATE: Quantum Reservoir (trained once, persists)
# ============================================================
qrc_solver = None
current_problem_state = {
    'distance_matrix': None,
    'coordinates': None,
    'current_routes': None,
    'traffic_multipliers': None,
    'num_vehicles': 2
}

def initialize_qrc_solver():
    """Initialize and train QRC solver on startup."""
    global qrc_solver
    
    if qrc_solver is None:
        logger.info("🌌 Initializing Quantum Reservoir Computing solver...")
        
        qrc_solver = ReservoirVRPSolver(n_reservoir_qubits=8)
        
        logger.info("🎓 Training reservoir (this takes ~30 seconds)...")
        training_instances = int(os.getenv("QRC_TRAINING_INSTANCES", "15"))
        training_data = generate_synthetic_training_data(n_instances=training_instances)
        qrc_solver.train(training_data, max_epochs=30)
        
        logger.info("✅ Quantum Reservoir ready for real-time adaptation!")

@app.on_event("startup")
async def startup_event():
    """Initialize QRC solver when server starts."""
    logger.info("=" * 80)
    logger.info("Q-FLEET QRC BACKEND STARTING...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Training instances: {os.getenv('QRC_TRAINING_INSTANCES', '15')}")
    logger.info("=" * 80)
    initialize_qrc_solver()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class VrpProblem(BaseModel):
    num_locations: int = Field(..., ge=2, le=6, description="Number of locations (2-6)")
    num_vehicles: int = Field(..., ge=1, le=3, description="Number of vehicles (1-3)")
    reps: int = Field(4, ge=1, le=6, description="QAOA depth (1-6)")
    use_qrc: bool = Field(False, description="Use Quantum Reservoir Computing")

class VrpResponse(BaseModel):
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    solution_method: str
    execution_time: float
    is_quantum_solution: bool
    notes: str

class TrafficJamEvent(BaseModel):
    jam_locations: List[List[int]] = Field(..., description="List of [from, to] location pairs experiencing traffic")
    jam_severity: float = Field(2.5, ge=1.0, le=5.0, description="Traffic severity multiplier (1.0=normal, 5.0=severe)")

class PriorityDeliveryEvent(BaseModel):
    priority_location: int = Field(..., ge=1, description="New urgent delivery location index")
    priority_level: int = Field(1, ge=1, le=3, description="Priority level (1=urgent, 3=low)")

class AdaptationResponse(BaseModel):
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    adaptation_time: float
    method: str
    event_type: str
    notes: str

# ============================================================
# CORE ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Root endpoint - shows API is running."""
    return {
        "service": "Q-Fleet Quantum Reservoir Computing API",
        "version": "2.0.0",
        "status": "operational",
        "qrc_status": "trained" if (qrc_solver and qrc_solver.trained) else "initializing",
        "documentation": "/docs",
        "health_check": "/api/health",
        "main_endpoints": {
            "optimize": "POST /api/optimize",
            "traffic_jam": "POST /api/traffic-jam",
            "priority_delivery": "POST /api/priority-delivery",
            "compare_methods": "POST /api/compare-methods",
            "current_state": "GET /api/current-state",
            "reset": "POST /api/reset"
        }
    }

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests."""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/api/health")
def health_check():
    logger.info("Health check endpoint called.")
    qrc_status = "trained" if qrc_solver and qrc_solver.trained else "not trained"
    return {
        "status": "healthy",
        "message": "Q-Fleet API with QRC is running!",
        "qrc_status": qrc_status
    }

@app.get("/api/qrc-status")
def qrc_status():
    """Debug endpoint for QRC system status."""
    if not qrc_solver:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "trained": qrc_solver.trained,
        "n_locations": qrc_solver.n_locations,
        "n_vehicles": qrc_solver.n_vehicles,
        "reservoir_qubits": qrc_solver.reservoir.n_qubits,
        "reservoir_dim": qrc_solver.reservoir.dim,
        "has_readout": qrc_solver.readout_model is not None
    }

@app.post("/api/optimize", response_model=VrpResponse)
def optimize_routes(problem: VrpProblem):
    """Main optimization endpoint. Supports both QAOA and QRC methods."""
    logger.info(f"Received VRP request: {problem.dict()}")
    
    try:
        # Generate problem instance
        np.random.seed(123)
        depot_node = 0
        coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
        
        distance_matrix = np.zeros((problem.num_locations + 1, problem.num_locations + 1))
        for i in range(problem.num_locations + 1):
            for j in range(i + 1, problem.num_locations + 1):
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        # Store in global state
        current_problem_state['distance_matrix'] = distance_matrix
        current_problem_state['coordinates'] = coords
        current_problem_state['num_vehicles'] = problem.num_vehicles
        current_problem_state['traffic_multipliers'] = np.ones_like(distance_matrix)
        
        # Check QRC compatibility
        if problem.use_qrc and qrc_solver and qrc_solver.trained:
            logger.info("Using Quantum Reservoir Computing...")
            logger.info(f"Problem: {problem.num_locations + 1} locs, {problem.num_vehicles} vehicles")
            logger.info(f"QRC trained on: {qrc_solver.n_locations} locs, {qrc_solver.n_vehicles} vehicles")
            
            if problem.num_locations + 1 > qrc_solver.n_locations:
                logger.warning(f"Problem too large for QRC. Using QAOA instead.")
                problem.use_qrc = False
            elif problem.num_vehicles > qrc_solver.n_vehicles:
                logger.warning(f"Too many vehicles for QRC. Using QAOA instead.")
                problem.use_qrc = False
        
        # Try QRC if compatible
        if problem.use_qrc and qrc_solver and qrc_solver.trained:
            start_time = time.time()
            
            try:
                solution = qrc_solver.solve_realtime(
                    distance_matrix,
                    current_problem_state['traffic_multipliers'],
                    problem.num_vehicles
                )
                
                routes = solution.routes
                total_distance = solution.total_distance
                execution_time = time.time() - start_time
                is_quantum = True
                method = "Quantum Reservoir Computing"
                notes = solution.notes
                
                if not routes or len(routes) == 0:
                    raise ValueError("QRC returned empty routes")
                
                logger.info(f"QRC solution: {len(routes)} routes, {total_distance:.2f} total")
                
            except Exception as qrc_error:
                logger.error(f"QRC failed: {qrc_error}", exc_info=True)
                logger.warning("Falling back to QAOA...")
                problem.use_qrc = False
        
        # Use QAOA if QRC not used or failed
        if not problem.use_qrc or not qrc_solver or not qrc_solver.trained:
            logger.info("Using QAOA...")
            start_time = time.time()
            
            routes, distances_qaoa, metrics = quantum_solver.solve_quantum_vrp(
                distance_matrix,
                problem.num_vehicles,
                depot_node,
                reps=problem.reps
            )
            
            total_distance = metrics.total_distance
            execution_time = time.time() - start_time
            is_quantum = metrics.is_valid_quantum_solution
            method = "Quantum QAOA" if is_quantum else "Classical Fallback"
            notes = metrics.notes
        
        # Store routes
        current_problem_state['current_routes'] = routes
        
        # Calculate distances
        distances = []
        for route in routes:
            route_dist = 0
            for i in range(len(route) - 1):
                route_dist += distance_matrix[route[i], route[i+1]]
            distances.append(float(route_dist))
        
        logger.info(f"✓ Optimization complete: {len(routes)} routes, {total_distance:.2f} total")
        
        return {
            "routes": routes,
            "distances": distances,
            "coordinates": coords.tolist(),
            "total_distance": float(total_distance),
            "solution_method": method,
            "execution_time": float(execution_time),
            "is_quantum_solution": is_quantum,
            "notes": notes
        }
        
    except Exception as e:
        logger.error(f"Error in optimize_routes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# ============================================================
# REAL-TIME ADAPTATION ENDPOINTS
# ============================================================

@app.post("/api/traffic-jam", response_model=AdaptationResponse)
def handle_traffic_jam(event: TrafficJamEvent):
    """Real-time adaptation to traffic congestion."""
    logger.warning(f"🚨 TRAFFIC JAM EVENT: {len(event.jam_locations)} affected routes")
    
    if not qrc_solver or not qrc_solver.trained:
        raise HTTPException(status_code=400, detail="QRC solver not initialized")
    
    if current_problem_state['distance_matrix'] is None:
        raise HTTPException(status_code=400, detail="No active problem. Call /api/optimize first")
    
    try:
        start_time = time.time()
        
        adapted_solution = qrc_solver.adapt_to_traffic_jam(
            current_problem_state['current_routes'],
            current_problem_state['distance_matrix'],
            jam_location_pairs=event.jam_locations,
            jam_severity=event.jam_severity
        )
        
        current_problem_state['current_routes'] = adapted_solution.routes
        
        for (i, j) in event.jam_locations:
            current_problem_state['traffic_multipliers'][i, j] = event.jam_severity
            current_problem_state['traffic_multipliers'][j, i] = event.jam_severity
        
        distances = []
        adjusted_matrix = current_problem_state['distance_matrix'] * current_problem_state['traffic_multipliers']
        
        for route in adapted_solution.routes:
            route_dist = 0
            for i in range(len(route) - 1):
                route_dist += adjusted_matrix[route[i], route[i+1]]
            distances.append(float(route_dist))
        
        adaptation_time = time.time() - start_time
        
        logger.info(f"✅ Traffic jam adaptation complete in {adaptation_time:.3f}s")
        
        return {
            "routes": adapted_solution.routes,
            "distances": distances,
            "coordinates": current_problem_state['coordinates'].tolist(),
            "total_distance": adapted_solution.total_distance,
            "adaptation_time": adaptation_time,
            "method": "Quantum Reservoir Computing (Real-Time Adaptation)",
            "event_type": "traffic_jam",
            "notes": f"Adapted to traffic jam in {adaptation_time:.3f}s"
        }
        
    except Exception as e:
        logger.error(f"Traffic jam adaptation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")

@app.post("/api/priority-delivery", response_model=AdaptationResponse)
def handle_priority_delivery(event: PriorityDeliveryEvent):
    """Real-time adaptation to priority delivery request."""
    logger.warning(f"🚨 PRIORITY DELIVERY: Location {event.priority_location}")
    
    if not qrc_solver or not qrc_solver.trained:
        raise HTTPException(status_code=400, detail="QRC solver not initialized")
    
    if current_problem_state['distance_matrix'] is None:
        raise HTTPException(status_code=400, detail="No active problem. Call /api/optimize first")
    
    try:
        start_time = time.time()
        
        n_locations = current_problem_state['distance_matrix'].shape[0]
        if event.priority_location >= n_locations:
            raise ValueError(f"Invalid priority location {event.priority_location}")
        
        adapted_solution = qrc_solver.adapt_to_priority_delivery(
            current_problem_state['current_routes'],
            current_problem_state['distance_matrix'],
            event.priority_location,
            current_problem_state['traffic_multipliers']
        )
        
        current_problem_state['current_routes'] = adapted_solution.routes
        
        distances = []
        adjusted_matrix = current_problem_state['distance_matrix'] * current_problem_state['traffic_multipliers']
        
        for route in adapted_solution.routes:
            route_dist = 0
            for i in range(len(route) - 1):
                route_dist += adjusted_matrix[route[i], route[i+1]]
            distances.append(float(route_dist))
        
        adaptation_time = time.time() - start_time
        
        logger.info(f"✅ Priority delivery adaptation complete in {adaptation_time:.3f}s")
        
        return {
            "routes": adapted_solution.routes,
            "distances": distances,
            "coordinates": current_problem_state['coordinates'].tolist(),
            "total_distance": adapted_solution.total_distance,
            "adaptation_time": adaptation_time,
            "method": "Quantum Reservoir Computing (Real-Time Adaptation)",
            "event_type": "priority_delivery",
            "notes": f"Added priority delivery in {adaptation_time:.3f}s"
        }
        
    except Exception as e:
        logger.error(f"Priority delivery adaptation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")

@app.get("/api/current-state")
def get_current_state():
    """Get current problem state."""
    if current_problem_state['distance_matrix'] is None:
        return {"status": "no_problem", "message": "No active problem"}
    
    return {
        "status": "active",
        "num_locations": current_problem_state['distance_matrix'].shape[0],
        "num_vehicles": current_problem_state['num_vehicles'],
        "current_routes": current_problem_state['current_routes'],
        "coordinates": current_problem_state['coordinates'].tolist(),
        "traffic_multipliers": current_problem_state['traffic_multipliers'].tolist(),
        "has_traffic_jam": bool(np.any(current_problem_state['traffic_multipliers'] > 1.5))
    }

@app.post("/api/reset")
def reset_state():
    """Reset problem state."""
    current_problem_state['distance_matrix'] = None
    current_problem_state['coordinates'] = None
    current_problem_state['current_routes'] = None
    current_problem_state['traffic_multipliers'] = None
    
    logger.info("Problem state reset")
    return {"status": "reset", "message": "Problem state cleared"}

@app.post("/api/compare-methods")
def compare_methods(problem: VrpProblem):
    """Compare QAOA vs QRC."""
    logger.info("Running comparison: QAOA vs QRC")
    
    try:
        np.random.seed(123)
        coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
        
        distance_matrix = np.zeros((problem.num_locations + 1, problem.num_locations + 1))
        for i in range(problem.num_locations + 1):
            for j in range(i + 1, problem.num_locations + 1):
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        traffic_mult = np.ones_like(distance_matrix)
        results = {}
        
        # QAOA
        logger.info("Testing QAOA...")
        start = time.time()
        routes_qaoa, distances_qaoa, metrics_qaoa = quantum_solver.solve_quantum_vrp(
            distance_matrix, problem.num_vehicles, 0, reps=problem.reps
        )
        qaoa_time = time.time() - start
        
        results['qaoa'] = {
            'time': qaoa_time,
            'distance': metrics_qaoa.total_distance,
            'method': 'QAOA',
            'routes': routes_qaoa
        }
        
        # QRC
        if qrc_solver and qrc_solver.trained:
            logger.info("Testing QRC...")
            start = time.time()
            solution_qrc = qrc_solver.solve_realtime(
                distance_matrix, traffic_mult, problem.num_vehicles
            )
            qrc_time = time.time() - start
            
            results['qrc'] = {
                'time': qrc_time,
                'distance': solution_qrc.total_distance,
                'method': 'QRC',
                'routes': solution_qrc.routes
            }
            
            speedup = qaoa_time / qrc_time if qrc_time > 0 else 0
            
            results['comparison'] = {
                'speedup': speedup,
                'winner': 'QRC' if qrc_time < qaoa_time else 'QAOA'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Q-FLEET API v2.0: QUANTUM RESERVOIR COMPUTING EDITION")
    print("=" * 80)
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)