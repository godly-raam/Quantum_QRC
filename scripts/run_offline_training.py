import os
import sys
import numpy as np

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.reservoir_trainer import train_reservoir_offline

def main():
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Train for 8 qubits (local testing)
    print("Training 8-qubit reservoir...")
    params_8q = train_reservoir_offline(num_qubits=8, layers=2)
    path_8q = os.path.join(weights_dir, "locked_reservoir_params_8q.npy")
    np.save(path_8q, params_8q)
    print(f"Saved 8-qubit parameters to {path_8q}\n")
    
    # Train for 27 qubits (MO-QAOA paper spec)
    print("Training 27-qubit reservoir...")
    params_27q = train_reservoir_offline(num_qubits=27, layers=2)
    path_27q = os.path.join(weights_dir, "locked_reservoir_params_27q.npy")
    np.save(path_27q, params_27q)
    print(f"Saved 27-qubit parameters to {path_27q}\n")

if __name__ == "__main__":
    main()
