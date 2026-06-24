import numpy as np
import math

def parse_vrp_instance(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a VRP dataset file and returns Q_fuel (Euclidean distances) 
    and Q_time (Euclidean distances + traffic noise).
    
    Returns:
        Q_fuel: np.ndarray matrix of distances
        Q_time: np.ndarray matrix of time (with simulated noise)
    """
    coords = {}
    reading_coords = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("DEMAND_SECTION") or line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
                reading_coords = False
                continue
                
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node_id] = (x, y)
                    
    num_nodes = len(coords)
    if num_nodes == 0:
        raise ValueError(f"No coordinates parsed from {filepath}")
        
    Q_fuel = np.zeros((num_nodes, num_nodes))
    Q_time = np.zeros((num_nodes, num_nodes))
    
    # Map node IDs to matrix indices 0..num_nodes-1
    node_ids = sorted(list(coords.keys()))
    
    for i, id1 in enumerate(node_ids):
        for j, id2 in enumerate(node_ids):
            if i != j:
                x1, y1 = coords[id1]
                x2, y2 = coords[id2]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                Q_fuel[i, j] = dist
                
                # Model Q_time as distance + some positive traffic noise
                # using a random uniform noise factor between 1.0 and 1.5
                noise_factor = np.random.uniform(1.0, 1.5)
                Q_time[i, j] = dist * noise_factor
                
    return Q_fuel, Q_time
