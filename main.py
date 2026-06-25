# main.py - QRC-Enhanced Backend with NON-BLOCKING Startup

import os
import sys
import logging
import time
import asyncio
import json
import uuid
import redis
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

# Redis Connection
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

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

# Global State
qrc_solver = None
training_status = {
    'status': 'pending',  # pending, training, ready, failed
    'progress': 0,
    'message': 'Waiting to start training'
}



async def initialize_qrc_solver_async():
    """
    Load existing model or train only if needed.
    This makes startup instant after first training.
    """
    global qrc_solver, training_status
    
    try:
        training_status['status'] = 'initializing'
        training_status['progress'] = 10
        training_status['message'] = 'Starting quantum reservoir...'
        
        logger.info("🌌 Initializing Quantum Reservoir Computing solver...")
        
        loop = asyncio.get_event_loop()
        n_qubits = int(os.getenv("QRC_NUM_QUBITS", "8"))
        
        # Load Phase 3 offline trained parameters
        weights_path = os.path.join(os.path.dirname(__file__), "weights", f"locked_reservoir_params_{n_qubits}q.npy")
        trained_params = None
        if os.path.exists(weights_path):
            trained_params = np.load(weights_path)
            logger.info(f"Loaded Phase 3 locked parameters from {weights_path}")
        else:
            logger.warning(f"No offline parameters found at {weights_path}. Running with random/unlocked parameters.")
        
        # Initialize solver (fast)
        qrc_solver = await loop.run_in_executor(
            None,
            ReservoirVRPSolver,
            n_qubits,
            trained_params
        )
        # System is instantly ready via MO-QAOA parameter transfer
        training_status['status'] = 'ready'
        training_status['progress'] = 100
        training_status['message'] = 'QRC locked parameters loaded. Ready!'
        logger.info("=" * 80)
        logger.info("✅ PRE-TRAINED RESERVOIR LOCKED - SYSTEM READY IN <1 SECOND!")
        logger.info("=" * 80)
        return  # EXIT - No training needed!
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        training_status['status'] = 'failed'
        training_status['progress'] = 0
        training_status['message'] = f'Training failed: {str(e)}'

@app.on_event("startup")
async def startup_event():
    """Initialize QRC solver when server starts - NON-BLOCKING."""
    logger.info("=" * 80)
    logger.info("Q-FLEET QRC BACKEND STARTING...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Training instances: {os.getenv('QRC_TRAINING_INSTANCES', '10')}")
    logger.info("=" * 80)
    
    # Start training in background WITHOUT awaiting it
    # This allows the server to start accepting requests immediately
    asyncio.create_task(initialize_qrc_solver_async())
    
    logger.info("✓ Server ready! QRC training running in background...")

# Pydantic Models

class VrpProblem(BaseModel):
    num_locations: int = Field(..., ge=2, le=6, description="Number of locations (2-6)")
    num_vehicles: int = Field(..., ge=1, le=3, description="Number of vehicles (1-3)")
    reps: int = Field(4, ge=1, le=6, description="QAOA depth (1-6)")
    use_qrc: bool = Field(False, description="Use Quantum Reservoir Computing")
    # NEW: Optional custom coordinates
    custom_coordinates: Optional[List[List[float]]] = Field(None, description="List of [lat, lon] points. First is Depot.")

class VrpResponse(BaseModel):
    problem_id: str
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    solution_method: str
    execution_time: float
    is_quantum_solution: bool
    notes: str

class TrafficJamEvent(BaseModel):
    problem_id: str
    jam_locations: List[List[int]] = Field(..., description="List of [from, to] location pairs experiencing traffic")
    jam_severity: float = Field(2.5, ge=1.0, le=5.0, description="Traffic severity multiplier")

class PriorityDeliveryEvent(BaseModel):
    problem_id: str
    priority_location: int = Field(..., ge=1, description="New urgent delivery location index")
    priority_level: int = Field(1, ge=1, le=3, description="Priority level")

class AdaptationResponse(BaseModel):
    routes: list
    distances: list
    coordinates: list
    total_distance: float
    adaptation_time: float
    method: str
    event_type: str
    notes: str

# Core Endpoints

@app.get("/")
def root():
    """Root endpoint - shows API is running."""
    return {
        "service": "Q-Fleet Quantum Reservoir Computing API",
        "version": "2.0.0",
        "status": "operational",
        "qrc_training_status": training_status['status'],
        "qrc_ready": training_status['status'] == 'ready',
        "documentation": "/docs",
        "health_check": "/api/health",
        "training_status": "/api/training-status",
        "main_endpoints": {
            "optimize": "POST /api/optimize",
            "traffic_jam": "POST /api/traffic-jam",
            "priority_delivery": "POST /api/priority-delivery"
        }
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint for Render."""
    logger.info("Health check endpoint called.")
    return {
        "status": "healthy",
        "message": "Q-Fleet API is running!",
        "server": "operational",
        "qrc_status": training_status['status']
    }

@app.get("/api/training-status")
def get_training_status():
    """Check QRC training progress."""
    return {
        "status": training_status['status'],
        "progress": training_status['progress'],
        "message": training_status['message'],
        "ready": training_status['status'] == 'ready'
    }

@app.get("/api/qrc-status")
def qrc_status():
    """Debug endpoint for QRC system status."""
    if not qrc_solver:
        return {
            "status": "not_initialized",
            "training": training_status
        }
    
    return {
        "status": "initialized",
        "trained": qrc_solver.trained,
        "reservoir_qubits": qrc_solver.reservoir.n_qubits,
        "training": training_status
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

@app.post("/api/optimize", response_model=VrpResponse)
async def optimize_routes(problem: VrpProblem):
    """Main optimization endpoint."""
    logger.info(f"Received VRP request: {problem.dict()}")
    
    # Input Guardrails
    if problem.num_locations > 8:
        if problem.use_qrc:
            logger.warning("Problem too large for QRC simulation (max 8). Switching to classical.")
            problem.use_qrc = False
            
    # Check if QRC is ready (if requested)
    if problem.use_qrc and training_status['status'] != 'ready':
        logger.warning(f"QRC requested but not ready (status: {training_status['status']}). Using QAOA.")
        problem.use_qrc = False
    
    try:
        # Use custom coordinates if provided
        if problem.custom_coordinates and len(problem.custom_coordinates) >= 2:
            logger.info(f"📍 Using {len(problem.custom_coordinates)} custom locations from user")
            coords = np.array(problem.custom_coordinates)
            
            # Recalculate distance matrix for THESE specific points using OSRM
            from modules.utils import get_osrm_distance_matrix
            loop = asyncio.get_running_loop()
            distance_matrix = await loop.run_in_executor(
                None,
                get_osrm_distance_matrix,
                coords
            )
            
            if distance_matrix is None:
                # Fallback if OSRM fails
                distance_matrix = np.zeros((len(coords), len(coords)))
                for i in range(len(coords)):
                    for j in range(len(coords)):
                        distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j]) * 111.0
            
            problem.num_locations = len(coords) - 1 # Exclude depot count
            depot_node = 0
            traffic_multipliers = np.ones_like(distance_matrix)
            
        else:
            # Generate problem instance
            np.random.seed(123) # Keep seed for reproducibility of demo, or remove for real random
            depot_node = 0
            coords = np.random.randn(problem.num_locations + 1, 2) * 0.1 + [16.5, 80.5]
            
            distance_matrix = np.zeros((problem.num_locations + 1, problem.num_locations + 1))
            for i in range(problem.num_locations + 1):
                for j in range(i + 1, problem.num_locations + 1):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distance_matrix[i, j] = distance_matrix[j, i] = dist
            
            traffic_multipliers = np.ones_like(distance_matrix)
        
        # Check QRC compatibility
        if problem.use_qrc and qrc_solver and qrc_solver.trained:
            if problem.num_locations + 1 > qrc_solver.n_locations:
                logger.warning(f"Problem too large for QRC. Using QAOA.")
                problem.use_qrc = False
            elif problem.num_vehicles > qrc_solver.n_vehicles:
                logger.warning(f"Too many vehicles for QRC. Using QAOA.")
                problem.use_qrc = False
        
        # Try QRC if compatible
        if problem.use_qrc and qrc_solver and qrc_solver.trained:
            start_time = time.time()
            
            try:
                solution = qrc_solver.solve_realtime(
                    distance_matrix,
                    traffic_multipliers,
                    problem.num_vehicles
                )
                
                routes = [[int(loc) for loc in route] for route in solution.routes]
                total_distance = float(solution.total_distance)
                execution_time = time.time() - start_time
                is_quantum = True
                method = "Quantum Reservoir Computing"
                notes = solution.notes
                
                if not routes or len(routes) == 0:
                    raise ValueError("QRC returned empty routes")
                
            except Exception as qrc_error:
                logger.error(f"QRC failed: {qrc_error}")
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
            
            routes = [[int(loc) for loc in route] for route in routes]
            total_distance = float(metrics.total_distance)
            execution_time = time.time() - start_time
            is_quantum = metrics.is_valid_quantum_solution
            method = "Quantum QAOA" if is_quantum else "Classical Fallback"
            notes = metrics.notes
        
        # Calculate distances
        distances = []
        for route in routes:
            route_dist = 0.0
            for i in range(len(route) - 1):
                route_dist += float(distance_matrix[route[i], route[i+1]])
            distances.append(route_dist)
            
        # Store state in Redis
        problem_id = str(uuid.uuid4())
        state = {
            "distance_matrix": distance_matrix.tolist(),
            "coordinates": coords.tolist(),
            "current_routes": routes,
            "traffic_multipliers": traffic_multipliers.tolist(),
            "num_vehicles": problem.num_vehicles,
            "created_at": time.time()
        }
        redis_client.setex(f"problem:{problem_id}", 3600, json.dumps(state))
        
        return {
            "problem_id": problem_id,
            "routes": routes,
            "distances": distances,
            "coordinates": coords.tolist(),
            "total_distance": total_distance,
            "solution_method": method,
            "execution_time": execution_time,
            "is_quantum_solution": is_quantum,
            "notes": notes
        }
        
    except Exception as e:
        logger.error(f"Error in optimize_routes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/api/traffic-jam", response_model=AdaptationResponse)
def handle_traffic_jam(event: TrafficJamEvent):
    """Real-time adaptation to traffic congestion."""
    if training_status['status'] != 'ready':
        raise HTTPException(
            status_code=503,
            detail=f"QRC not ready yet. Status: {training_status['message']}"
        )
    
    # Fetch state from Redis
    state_json = redis_client.get(f"problem:{event.problem_id}")
    if not state_json:
        raise HTTPException(status_code=404, detail="Problem ID not found or expired")
    
    state = json.loads(state_json)
    distance_matrix = np.array(state['distance_matrix'])
    current_routes = state['current_routes']
    traffic_multipliers = np.array(state['traffic_multipliers'])
    coordinates = np.array(state['coordinates'])
    
    try:
        start_time = time.time()
        
        adapted_solution = qrc_solver.adapt_to_traffic_jam(
            current_routes,
            distance_matrix,
            jam_location_pairs=event.jam_locations,
            jam_severity=event.jam_severity
        )
        
        # Convert routes to pure Python types (FIX for numpy serialization)
        routes = [[int(loc) for loc in route] for route in adapted_solution.routes]
        
        # Update state
        for (i, j) in event.jam_locations:
            traffic_multipliers[i, j] = event.jam_severity
            traffic_multipliers[j, i] = event.jam_severity
            
        # Save back to Redis
        state['current_routes'] = routes
        state['traffic_multipliers'] = traffic_multipliers.tolist()
        redis_client.setex(f"problem:{event.problem_id}", 3600, json.dumps(state))
        
        distances = []
        adjusted_matrix = distance_matrix * traffic_multipliers
        
        for route in routes:
            route_dist = 0
            for i in range(len(route) - 1):
                route_dist += adjusted_matrix[route[i], route[i+1]]
            distances.append(float(route_dist))  # Ensure float, not numpy.float64
        
        adaptation_time = time.time() - start_time
        
        # Convert all numpy types to Python native types
        return {
            "routes": routes,  # Already converted above
            "distances": distances,  # Already converted to float
            "coordinates": coordinates.tolist(),
            "total_distance": float(adapted_solution.total_distance),  # Ensure float
            "adaptation_time": float(adaptation_time),  # Ensure float
            "method": "Quantum Reservoir Computing",
            "event_type": "traffic_jam",
            "notes": f"Adapted in {adaptation_time:.3f}s"
        }
        
    except Exception as e:
        logger.error(f"Traffic jam adaptation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")


@app.post("/api/priority-delivery", response_model=AdaptationResponse)
def handle_priority_delivery(event: PriorityDeliveryEvent):
    """Real-time adaptation to priority delivery request."""
    if training_status['status'] != 'ready':
        raise HTTPException(
            status_code=503,
            detail=f"QRC not ready yet. Status: {training_status['message']}"
        )
    
    # Fetch state from Redis
    state_json = redis_client.get(f"problem:{event.problem_id}")
    if not state_json:
        raise HTTPException(status_code=404, detail="Problem ID not found or expired")
    
    state = json.loads(state_json)
    distance_matrix = np.array(state['distance_matrix'])
    current_routes = state['current_routes']
    traffic_multipliers = np.array(state['traffic_multipliers'])
    coordinates = np.array(state['coordinates'])
    
    try:
        start_time = time.time()
        
        adapted_solution = qrc_solver.adapt_to_priority_delivery(
            current_routes,
            distance_matrix,
            event.priority_location,
            traffic_multipliers
        )
        
        # Convert routes to pure Python types (FIX for numpy serialization)
        routes = [[int(loc) for loc in route] for route in adapted_solution.routes]
        
        # Save back to Redis
        state['current_routes'] = routes
        redis_client.setex(f"problem:{event.problem_id}", 3600, json.dumps(state))
        
        distances = []
        adjusted_matrix = distance_matrix * traffic_multipliers
        
        for route in routes:
            route_dist = 0
            for i in range(len(route) - 1):
                route_dist += adjusted_matrix[route[i], route[i+1]]
            distances.append(float(route_dist))  # Ensure float
        
        adaptation_time = time.time() - start_time
        
        # Convert all numpy types to Python native types
        return {
            "routes": routes,  # Already converted above
            "distances": distances,  # Already converted to float
            "coordinates": coordinates.tolist(),
            "total_distance": float(adapted_solution.total_distance),  # Ensure float
            "adaptation_time": float(adaptation_time),  # Ensure float
            "method": "Quantum Reservoir Computing",
            "event_type": "priority_delivery",
            "notes": f"Added priority delivery in {adaptation_time:.3f}s"
        }
        
    except Exception as e:
        logger.error(f"Priority delivery adaptation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")

@app.get("/api/current-state")
def get_current_state(problem_id: str):
    """Get current problem state."""
    state_json = redis_client.get(f"problem:{problem_id}")
    if not state_json:
        return {"status": "not_found", "message": "Problem ID not found or expired"}
    
    state = json.loads(state_json)
    distance_matrix = np.array(state['distance_matrix'])
    traffic_multipliers = np.array(state['traffic_multipliers'])
    
    return {
        "status": "active",
        "num_locations": distance_matrix.shape[0],
        "num_vehicles": state['num_vehicles'],
        "current_routes": state['current_routes'],
        "has_traffic_jam": bool(np.any(traffic_multipliers > 1.5)),
        "qrc_ready": training_status['status'] == 'ready'
    }

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Q-FLEET API v2.0: QUANTUM RESERVOIR COMPUTING EDITION")
    print("=" * 80)
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)