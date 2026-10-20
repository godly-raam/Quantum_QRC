"""Fast statevector observables and a resumable on-disk feature cache.

The original Qiskit measurement loop builds one ``SparsePauliOp`` per
observable.  That is fine for a handful of features but expensive when the
pairwise readout needs every ``ZZ_ij`` term.  This module computes all
single-qubit ``X/Y/Z`` expectations and every pairwise ``ZZ`` expectation
directly from the statevector amplitudes.

Feature layout (little-endian qubit indexing, matching Qiskit):

* ``3 * n_qubits`` single-qubit values, ordered ``X_0, Y_0, Z_0, X_1, ...``.
* ``n_qubits * (n_qubits - 1) / 2`` pairwise ``ZZ_ij`` values.  The first
  ``min(3, n_qubits - 1)`` entries are nearest-neighbour pairs
  ``(0, 1), (1, 2), (2, 3)`` so the legacy node readout keeps working.  The
  remaining pairs follow row-major order over ``i < j``.

The cache stores features in shards keyed by ``(size, angle_scale,
theta_hash)`` and indexed by instance id.  Features do not depend on the
readout, so changing the readout never invalidates the cache.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional

import numpy as np

FEATURE_CACHE_VERSION = "v2-allzz"


def zz_pairs(n_qubits: int) -> list[tuple[int, int]]:
    """Return all qubit pairs in measurement order (adjacent pairs first)."""
    adjacent = [(qubit, qubit + 1) for qubit in range(min(3, n_qubits - 1))]
    seen = set(adjacent)
    remaining = [
        (left, right)
        for left in range(n_qubits)
        for right in range(left + 1, n_qubits)
        if (left, right) not in seen
    ]
    return adjacent + remaining


def zz_pair_index(n_qubits: int) -> dict[tuple[int, int], int]:
    """Map ``(i, j)`` to its offset in the ZZ block of the feature vector."""
    offset = 3 * n_qubits
    return {
        pair: offset + index for index, pair in enumerate(zz_pairs(n_qubits))
    }


def measure_features(statevector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Measure all single-qubit Paulis and all pairwise ``ZZ`` expectations."""
    state = np.asarray(statevector, dtype=complex)
    expected = 1 << n_qubits
    if state.shape != (expected,):
        raise ValueError(
            f"statevector must have shape ({expected},), got {state.shape}"
        )

    indices = np.arange(expected, dtype=np.int64)
    probabilities = np.abs(state) ** 2
    features: list[float] = []

    for qubit in range(n_qubits):
        flipped = indices ^ (1 << qubit)
        sign = 1.0 - 2.0 * ((indices >> qubit) & 1)
        x_value = np.vdot(state, state[flipped])
        y_value = -1j * np.sum(sign * np.conj(state) * state[flipped])
        z_value = np.sum(probabilities * sign)
        features.extend(
            [float(x_value.real), float(y_value.real), float(z_value.real)]
        )

    bits = ((indices[:, None] >> np.arange(n_qubits)) & 1).astype(float)
    signs = 2.0 * bits - 1.0
    correlations = (signs * probabilities[:, None]).T @ signs
    pair_index = zz_pair_index(n_qubits)
    offset = 3 * n_qubits
    zz_values = np.zeros(len(pair_index), dtype=float)
    for (left, right), position in pair_index.items():
        zz_values[position - offset] = correlations[left, right]
    features.extend(zz_values.tolist())

    return np.asarray(features, dtype=float)


def feature_config_hash(payload: str) -> str:
    """Stable short hash for a feature-generation configuration."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def array_hash(values: Optional[np.ndarray]) -> str:
    """Stable hash of an optional parameter array."""
    if values is None:
        return "none"
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:32]


class FeatureCache:
    """Resumable feature shards keyed by size, scale, theta, and instance id."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._shards: dict[tuple[int, float, str], dict[int, np.ndarray]] = {}
        self._dirty: set[tuple[int, float, str]] = set()

    def _path(self, size: int, angle_scale: float, theta_hash: str) -> str:
        key = f"{FEATURE_CACHE_VERSION}|{size}|{angle_scale:.12g}|{theta_hash}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, f"n{size}", f"{digest}.npz")

    def _load(self, size: int, angle_scale: float, theta_hash: str) -> dict[int, np.ndarray]:
        key = (size, float(angle_scale), theta_hash)
        if key in self._shards:
            return self._shards[key]

        path = self._path(size, angle_scale, theta_hash)
        shard: dict[int, np.ndarray] = {}
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as data:
                    instance_ids = np.asarray(data["instance_ids"], dtype=np.int64)
                    features = np.asarray(data["features"], dtype=float)
                if features.ndim == 2 and len(instance_ids) == len(features):
                    shard = {
                        int(instance_id): features[row].copy()
                        for row, instance_id in enumerate(instance_ids)
                    }
            except (OSError, ValueError, KeyError):
                shard = {}
        self._shards[key] = shard
        return shard

    def get(
        self,
        size: int,
        instance_id: int,
        angle_scale: float,
        theta_hash: str,
    ) -> Optional[np.ndarray]:
        """Return cached features, or ``None`` on a miss."""
        shard = self._load(size, angle_scale, theta_hash)
        value = shard.get(int(instance_id))
        return None if value is None else value.copy()

    def put(
        self,
        size: int,
        instance_id: int,
        angle_scale: float,
        theta_hash: str,
        features: np.ndarray,
    ) -> None:
        """Store features for one instance and mark the shard dirty."""
        key = (size, float(angle_scale), theta_hash)
        shard = self._load(size, angle_scale, theta_hash)
        shard[int(instance_id)] = np.asarray(features, dtype=float).copy()
        self._dirty.add(key)

    def get_or_compute(
        self,
        size: int,
        instance_id: int,
        angle_scale: float,
        theta_hash: str,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Return cached features or compute and cache them."""
        cached = self.get(size, instance_id, angle_scale, theta_hash)
        if cached is not None:
            return cached
        value = np.asarray(compute(), dtype=float)
        self.put(size, instance_id, angle_scale, theta_hash, value)
        return value

    def flush(self) -> None:
        """Atomically persist all dirty shards."""
        for size, angle_scale, theta_hash in sorted(self._dirty):
            shard = self._shards[(size, angle_scale, theta_hash)]
            if not shard:
                continue
            instance_ids = np.asarray(sorted(shard), dtype=np.int64)
            features = np.vstack([shard[instance_id] for instance_id in instance_ids])
            path = self._path(size, angle_scale, theta_hash)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.tmp.{os.getpid()}.npz"
            np.savez_compressed(
                temporary,
                instance_ids=instance_ids,
                features=features,
            )
            os.replace(temporary, path)
        self._dirty.clear()


def statevector_from_circuit(circuit, prefer_aer: bool = True) -> np.ndarray:
    """Return a circuit statevector, using Aer when available.

    Aer's statevector simulator is several times faster than
    ``Statevector.from_instruction`` for 16-qubit circuits.  Any Aer failure
    falls back to the Qiskit reference path so the benchmark never depends on
    the optional accelerator for correctness.
    """
    if prefer_aer:
        try:
            from qiskit_aer import AerSimulator

            saved = circuit.copy()
            saved.save_statevector()
            result = AerSimulator(method="statevector").run(saved).result()
            return np.asarray(result.get_statevector(saved), dtype=complex)
        except Exception:  # pragma: no cover - exercised by fallback tests
            pass

    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)
