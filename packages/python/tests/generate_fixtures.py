# Generate test fixtures for randomized tests against Qiskit statevector simulator.
#   uv run packages/python/tests/generate_fixtures.py

import json
import random
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qlit import (
    Circuit,
    Gate,
)

FIXTURES_PATH = Path(__file__).parent / "randomized_fixtures.json"


def setup(
    qubits,
    gates, 
    t_gates, 
    w_samples,
    seed,
) -> tuple[QuantumCircuit, list[str]]:
    circuit = Circuit.random(qubits, gates, t_gates, seed)
    random.seed(seed)

    ws = []
    for _ in range(w_samples):
        ws.append(format(random.getrandbits(qubits), f"00{qubits}b"))

    qiskit_circuit = QuantumCircuit(qubits)
    for gate in circuit.gates:
        match gate:
            case Gate.X(a):
                qiskit_circuit.x(a)
            case Gate.Y(a):
                qiskit_circuit.y(a)
            case Gate.Z(a):
                qiskit_circuit.z(a)
            case Gate.H(a):
                qiskit_circuit.h(a)
            case Gate.S(a):
                qiskit_circuit.s(a)
            case Gate.Sdg(a):
                qiskit_circuit.sdg(a)
            case Gate.Cnot(a, b):
                qiskit_circuit.cx(a, b)
            case Gate.Cz(a, b):
                qiskit_circuit.cz(a, b)
            case Gate.T(a):
                qiskit_circuit.t(a)
            case Gate.Tdg(a):
                qiskit_circuit.tdg(a)
            case _:
                raise ValueError(f"Unknown gate: {gate}")

    return qiskit_circuit, ws


def generate_fixtures():
    print("Generating fixtures for randomized tests...")

    data = []
    iterations = 10
    w_samples = 50
    for i in range(iterations):
        qubits = 10
        t_gates = 5
        gates = 100
        seed = 1234 + i
        qiskit_circuit, ws = setup(qubits, gates, t_gates, w_samples, seed)
        
        state = Statevector.from_instruction(qiskit_circuit)
        w_data = []
        for w in ws:
            # Remember that Qiskit uses reversed basis states
            w_be = w[::-1]
            w_coeff = state.data[int(w_be, 2)]
            w_data.append({"w": w, "w_coeff_re": w_coeff.real, "w_coeff_im": w_coeff.imag})
        data.append({"qubits": qubits, "t_gates": t_gates, "gates": gates, "seed": seed, "w_data": w_data})

    with open(FIXTURES_PATH, "w") as f:
        f.write(json.dumps(data, indent=2))


if __name__ == "__main__":
    generate_fixtures()
