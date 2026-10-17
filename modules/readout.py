"""Trained classical readouts for quantum/classical reservoir features.

The readout predicts a vehicle assignment for every customer.  Vehicle labels
are canonicalised before training: routes are ordered by the polar angle of
their customer centroid around the depot.  This removes the label permutation
symmetry that otherwise makes the same routing solution look different from
run to run.

Two feature layouts are supported by the benchmark:

``flat``
    One global feature vector per instance -> ``n_vehicles * (n_locations - 1)``
    logits (customer-major).

``node``
    One feature vector per customer -> ``n_vehicles`` logits.  Weights are
    shared across customers.

Both use the same ridge objective and cross-validated regularisation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


def _route_stops(route: Iterable[int], depot: int = 0) -> tuple[int, ...]:
    """Return customer stops of a route, excluding the depot."""
    return tuple(int(node) for node in route if int(node) != depot)


def route_angle(
    stops: Sequence[int], coordinates: np.ndarray, depot: int = 0
) -> tuple[float, tuple[int, ...]]:
    """Polar angle of a route centroid around the depot, with a stable tie-break."""
    if len(stops) == 0:
        return float("inf"), ()
    points = np.asarray(coordinates, dtype=float)[list(stops)]
    centroid = points.mean(axis=0)
    delta = centroid - np.asarray(coordinates, dtype=float)[depot]
    angle = float(np.arctan2(delta[1], delta[0]))
    return angle, tuple(sorted(stops))


def canonicalize_routes(
    routes: Sequence[Sequence[int]], coordinates: np.ndarray, n_vehicles: int
) -> list[list[int]]:
    """Canonicalise vehicle labels by route centroid polar angle.

    The returned list always has exactly ``n_vehicles`` entries.  Non-empty
    routes come first, sorted by angle; empty routes are padded at the end.
    The ordering rule is invariant to the input vehicle permutation (ties are
    broken by the sorted list of stops, never by input position).
    """
    nonempty = [_route_stops(route) for route in routes]
    nonempty = [stops for stops in nonempty if stops]
    keyed = [(route_angle(stops, coordinates), stops) for stops in nonempty]
    keyed.sort(key=lambda item: item[0])

    ordered: list[list[int]] = [[0, *stops, 0] for _, stops in keyed]
    ordered.extend([[0, 0] for _ in range(max(0, n_vehicles - len(ordered)))])
    return ordered[:n_vehicles]


def routes_to_assignment(
    routes: Sequence[Sequence[int]], n_locations: int, n_vehicles: int
) -> np.ndarray:
    """Convert routes to a customer-major one-hot assignment matrix."""
    assignment = np.zeros((n_locations - 1, n_vehicles), dtype=float)
    for vehicle, route in enumerate(routes):
        for stop in _route_stops(route):
            if 1 <= stop < n_locations:
                assignment[stop - 1, vehicle] = 1.0
    return assignment


def assignments_from_logits(
    logits: np.ndarray, n_locations: int, n_vehicles: int
) -> np.ndarray:
    """Argmax vehicle label per customer from customer-major logits."""
    matrix = np.asarray(logits, dtype=float).reshape(n_locations - 1, n_vehicles)
    return np.argmax(matrix, axis=1)


def readout_assignment(
    readout: "RidgeReadout",
    features: np.ndarray,
    n_locations: int,
    n_vehicles: int,
) -> np.ndarray:
    """Predict a customer vehicle assignment with a flat or node readout."""
    if readout.mode == "node":
        rows = node_features(features, n_locations)
        return np.argmax(readout.predict(rows), axis=1)
    logits = readout.predict(np.asarray(features, dtype=float).reshape(1, -1))
    return assignments_from_logits(logits.reshape(-1), n_locations, n_vehicles)


def node_features(features: np.ndarray, n_qubits: int, n_zz: int = 3) -> np.ndarray:
    """Build per-customer feature rows from a reservoir feature vector.

    Row ``i`` (customer ``i + 1``) contains the three single-qubit features
    for qubit ``i`` plus the adjacent ZZ features that involve qubit ``i``.
    Missing adjacency slots are zero-padded so every row has the same width.
    """
    features = np.asarray(features, dtype=float)
    rows = []
    for node in range(1, n_qubits):
        base = features[3 * node : 3 * node + 3]
        zz_left = (
            features[3 * n_qubits + (node - 1)]
            if node - 1 < n_zz
            else 0.0
        )
        zz_right = (
            features[3 * n_qubits + node] if node < n_zz else 0.0
        )
        rows.append(np.concatenate([base, [zz_left, zz_right]]))
    return np.asarray(rows, dtype=float)


@dataclass
class RidgeReadout:
    """Ridge readout trained with k-fold CV on the training split only."""

    n_features: Optional[int] = None
    n_outputs: Optional[int] = None
    alphas: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0)
    cv_folds: int = 5
    seed: int = 0
    mode: str = "flat"
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    target_mean: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    best_alpha: Optional[float] = None
    cv_scores: dict[float, float] = field(default_factory=dict)

    def _standardise_fit(self, features: np.ndarray) -> np.ndarray:
        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.std = scale
        return (features - self.mean) / self.std

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("RidgeReadout must be fitted before prediction")
        return (features - self.mean) / self.std

    @staticmethod
    def _ridge_weights(
        features: np.ndarray, targets: np.ndarray, alpha: float
    ) -> np.ndarray:
        gram = features.T @ features
        regulariser = alpha * np.eye(gram.shape[0])
        return np.linalg.solve(gram + regulariser, features.T @ targets)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeReadout":
        """Fit with k-fold CV over ``alphas`` on the supplied training data."""
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be 2-D")
        if len(features) != len(targets):
            raise ValueError("features and targets must have matching rows")

        self.n_features = features.shape[1]
        self.n_outputs = targets.shape[1]
        self.target_mean = targets.mean(axis=0)
        centered_targets = targets - self.target_mean
        standardised = self._standardise_fit(features)

        n_samples = len(standardised)
        folds = max(2, min(self.cv_folds, n_samples))
        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        splits = np.array_split(indices, folds)

        best_alpha = float(self.alphas[0])
        best_score = float("inf")
        for alpha in self.alphas:
            scores = []
            for fold in splits:
                validation = np.zeros(n_samples, dtype=bool)
                validation[fold] = True
                train_mask = ~validation
                if not np.any(train_mask):
                    continue
                weights = self._ridge_weights(
                    standardised[train_mask],
                    centered_targets[train_mask],
                    float(alpha),
                )
                prediction = standardised[validation] @ weights
                scores.append(
                    float(
                        np.mean(
                            (prediction - centered_targets[validation]) ** 2
                        )
                    )
                )
            mean_score = float(np.mean(scores)) if scores else float("inf")
            self.cv_scores[float(alpha)] = mean_score
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = float(alpha)

        self.best_alpha = best_alpha
        self.weights = self._ridge_weights(
            standardised, centered_targets, best_alpha
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict raw logits for standardised features."""
        if self.weights is None or self.target_mean is None:
            raise RuntimeError("RidgeReadout must be fitted before prediction")
        return (
            self._standardise(np.asarray(features, dtype=float)) @ self.weights
            + self.target_mean
        )

    def predict_assignment(
        self, features: np.ndarray, n_locations: int, n_vehicles: int
    ) -> np.ndarray:
        """Predict a customer vehicle label array."""
        return assignments_from_logits(
            self.predict(features), n_locations, n_vehicles
        )

    def save(self, path: str) -> None:
        """Persist weights and standardisation statistics."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            mean=self.mean,
            std=self.std,
            target_mean=self.target_mean,
            best_alpha=np.array(self.best_alpha),
            n_features=np.array(self.n_features),
            n_outputs=np.array(self.n_outputs),
        )
        metadata = {
            "mode": self.mode,
            "best_alpha": self.best_alpha,
            "cv_scores": {str(key): value for key, value in self.cv_scores.items()},
            "seed": self.seed,
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RidgeReadout":
        """Load a readout persisted by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        metadata_path = os.path.splitext(path)[0] + ".json"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        readout = cls(
            n_features=int(data["n_features"]),
            n_outputs=int(data["n_outputs"]),
            mode=metadata.get("mode", "flat"),
            seed=int(metadata.get("seed", 0)),
        )
        readout.weights = data["weights"]
        readout.mean = data["mean"]
        readout.std = data["std"]
        readout.target_mean = data["target_mean"]
        readout.best_alpha = float(data["best_alpha"])
        readout.cv_scores = {
            float(key): float(value)
            for key, value in metadata.get("cv_scores", {}).items()
        }
        return readout
