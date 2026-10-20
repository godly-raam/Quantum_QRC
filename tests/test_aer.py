import numpy as np
from qiskit import QuantumCircuit

from modules.fast_sim import statevector_from_circuit


def test_aer_matches_qiskit_statevector():
    circuit = QuantumCircuit(6)
    circuit.h(0)
    for qubit in range(5):
        circuit.cx(qubit, qubit + 1)
    circuit.ry(0.37, 2)
    circuit.rz(-0.81, 4)

    reference = statevector_from_circuit(circuit, prefer_aer=False)
    accelerated = statevector_from_circuit(circuit, prefer_aer=True)

    assert accelerated.shape == reference.shape
    assert np.allclose(accelerated, reference, atol=1e-10)


def test_aer_failure_falls_back_to_qiskit(monkeypatch):
    from unittest.mock import patch

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)

    reference = statevector_from_circuit(circuit, prefer_aer=False)
    with patch("qiskit_aer.AerSimulator", side_effect=RuntimeError("no aer")):
        fallback = statevector_from_circuit(circuit, prefer_aer=True)

    assert np.allclose(fallback, reference, atol=1e-12)
