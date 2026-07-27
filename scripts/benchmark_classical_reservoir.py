"""Compare a matched classical Echo State Network with the quantum reservoir.

The comparison intentionally uses the existing offline quantum-parameter trainer and
ReservoirVRPSolver route decoder.  Both reservoirs therefore expose 3 * N features
and are evaluated by the same direct feature-to-route readout; lower route cost is
better.  This is an ablation, not a claim of quantum advantage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit.quantum_info import Statevector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from modules.quantum_reservoir_vrp import (  # noqa: E402 # pylint: disable=wrong-import-position,import-error
    QuantumReservoir,
    ReservoirVRPSolver,
)
from modules.reservoir_trainer import (  # noqa: E402 # pylint: disable=wrong-import-position,import-error
    train_reservoir_offline,
)


@dataclass
class EchoStateNetwork:  # pylint: disable=too-many-instance-attributes
    """Fixed ESN with N recurrent units and a 3N matched output feature vector."""

    size: int
    input_dim: int
    spectral_radius: float = 0.9
    input_scale: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.input_weights = rng.uniform(
            -self.input_scale, self.input_scale, size=(self.size, self.input_dim)
        )
        recurrent = rng.uniform(-1.0, 1.0, size=(self.size, self.size))
        radius = np.max(np.abs(np.linalg.eigvals(recurrent)))
        self.recurrent_weights = recurrent * (self.spectral_radius / radius)
        self.state = np.zeros(self.size)

    def reset(self) -> None:
        """Reset reservoir hidden state to zero."""
        self.state.fill(0.0)

    def features(self, matrix: np.ndarray) -> np.ndarray:
        """Compute feature vector from input cost matrix."""
        normalized = _normalise_matrix(matrix).ravel()
        self.state = np.tanh(self.input_weights @ normalized + self.recurrent_weights @ self.state)
        # Match QuantumReservoir.measure_observables()'s X/Y/Z feature count: 3N.
        return np.concatenate((self.state, self.state**2, np.tanh(2.0 * self.state)))


def _normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(matrix)))
    return matrix if scale == 0.0 else matrix / scale


def _random_distance_matrix(num_locations: int, rng: np.random.Generator) -> np.ndarray:
    coordinates = rng.uniform(0.0, 1.0, size=(num_locations, 2))
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    matrix = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _route_cost(routes: list[list[int]], distance_matrix: np.ndarray) -> float:
    return float(
        sum(
            distance_matrix[start, end]
            for route in routes
            for start, end in zip(route, route[1:])
        )
    )


def _quantum_features(
    reservoir: QuantumReservoir, distance_matrix: np.ndarray, num_vehicles: int
) -> np.ndarray:
    circuit = reservoir.build_full_architecture(distance_matrix, distance_matrix, num_vehicles)
    state = Statevector.from_instruction(circuit).data
    reservoir.current_state = state / np.linalg.norm(state)
    return reservoir.measure_observables()[: 3 * reservoir.n_qubits]


def _evaluate(
    feature_fn: Callable[[np.ndarray], np.ndarray],
    decoder: ReservoirVRPSolver,
    instances: list[np.ndarray],
    num_vehicles: int,
) -> tuple[float, float, float]:
    feature_rows = []
    costs = []
    for distance_matrix in instances:
        features = feature_fn(distance_matrix)
        feature_rows.append(features)
        encoding_size = num_vehicles * distance_matrix.shape[0]
        padded = np.pad(features, (0, max(0, encoding_size - len(features))))
        route_encoding = padded[:encoding_size]
        routes = decoder._decode_routes(  # pylint: disable=protected-access
            route_encoding, num_vehicles, distance_matrix.shape[0]
        )
        costs.append(_route_cost(routes, distance_matrix))

    feature_matrix = np.vstack(feature_rows)
    # Sum per-feature variance to match reservoir_trainer.train_reservoir_offline().
    expressivity_variance = float(np.sum(np.var(feature_matrix, axis=0)))
    return expressivity_variance, float(np.mean(costs)), float(np.std(costs))


def main() -> None:
    """Run classical ESN vs quantum reservoir comparison benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits", type=int, default=8, help="Reservoir/ESN unit count (default: 8)"
    )
    parser.add_argument(
        "--instances", type=int, default=20, help="Number of synthetic routing instances"
    )
    parser.add_argument(
        "--vehicles", type=int, default=4, help="Number of vehicles for the common decoder"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to locked quantum parameters; trains with reservoir_trainer if omitted",
    )
    args = parser.parse_args()

    if args.qubits < 2:
        parser.error("--qubits must be at least 2")
    if args.instances < 2:
        parser.error("--instances must be at least 2 to measure expressivity variance")

    weights_path = args.weights or os.path.join(
        PROJECT_ROOT, "weights", f"locked_reservoir_params_{args.qubits}q.npy"
    )
    if os.path.exists(weights_path):
        quantum_params = np.load(weights_path)
    else:
        print(
            "No locked quantum weights found; "
            "training with reservoir_trainer.train_reservoir_offline()."
        )
        quantum_params = train_reservoir_offline(args.qubits)

    quantum_reservoir = QuantumReservoir(
        n_reservoir_qubits=args.qubits,
        trained_params=quantum_params,
        random_seed=args.seed,
    )
    esn = EchoStateNetwork(size=args.qubits, input_dim=args.qubits * args.qubits, seed=args.seed)
    decoder = ReservoirVRPSolver(n_reservoir_qubits=args.qubits, trained_params=quantum_params)

    rng = np.random.default_rng(args.seed)
    instances = [_random_distance_matrix(args.qubits, rng) for _ in range(args.instances)]

    quantum_results = _evaluate(
        lambda matrix: _quantum_features(quantum_reservoir, matrix, args.vehicles),
        decoder,
        instances,
        args.vehicles,
    )
    classical_results = _evaluate(
        lambda matrix: (esn.reset(), esn.features(matrix))[1],
        decoder,
        instances,
        args.vehicles,
    )

    report = {
        "configuration": {
            "qubits_or_esn_units": args.qubits,
            "matched_feature_dimension": 3 * args.qubits,
            "instances": args.instances,
            "vehicles": args.vehicles,
            "seed": args.seed,
        },
        "quantum_reservoir": {
            "expressivity_variance": quantum_results[0],
            "mean_route_cost": quantum_results[1],
            "route_cost_std": quantum_results[2],
        },
        "classical_echo_state_network": {
            "expressivity_variance": classical_results[0],
            "mean_route_cost": classical_results[1],
            "route_cost_std": classical_results[2],
        },
        "interpretation": (
            "Both reservoirs use the same direct decoder; lower route cost is better."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
