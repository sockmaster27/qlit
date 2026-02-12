import random
import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qlit import (
    CliffordTCircuit,
    CliffordTGate,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)


class QlitRandomizedTests(unittest.TestCase):
    def setup(
        self, qubits, gates, t_gates, seed
    ) -> tuple[CliffordTCircuit, QuantumCircuit, str]:
        circuit = CliffordTCircuit.random(qubits, gates, t_gates, seed)
        random.seed(seed)
        w = format(random.getrandbits(qubits), f"00{qubits}b")

        qiskit_circuit = QuantumCircuit(qubits)
        for gate in circuit.gates:
            match gate:
                case CliffordTGate.X(a):
                    qiskit_circuit.x(a)
                case CliffordTGate.Y(a):
                    qiskit_circuit.y(a)
                case CliffordTGate.Z(a):
                    qiskit_circuit.z(a)
                case CliffordTGate.H(a):
                    qiskit_circuit.h(a)
                case CliffordTGate.S(a):
                    qiskit_circuit.s(a)
                case CliffordTGate.Sdg(a):
                    qiskit_circuit.sdg(a)
                case CliffordTGate.Cnot(a, b):
                    qiskit_circuit.cx(a, b)
                case CliffordTGate.Cz(a, b):
                    qiskit_circuit.cz(a, b)
                case CliffordTGate.T(a):
                    qiskit_circuit.t(a)
                case CliffordTGate.Tdg(a):
                    qiskit_circuit.tdg(a)
                case _:
                    raise ValueError(f"Unknown gate: {gate}")

        return circuit, qiskit_circuit, w

    def test_against_statevector(self):
        print("\nTesting against Qiskit statevector simulator...")
        iterations = 500
        for i in range(iterations):
            qubits = 10
            t_gates = 5
            gates = 100
            seed = 1234 + i
            circuit, qiskit_circuit, w = self.setup(qubits, gates, t_gates, seed)

            state = Statevector.from_instruction(qiskit_circuit)
            # Remember that Qiskit uses reversed basis states
            w_be = w[::-1]
            w_coeff_qiskit = state.data[int(w_be, 2)]

            self.assertAlmostEqual(
                simulate_circuit(w, circuit),
                w_coeff_qiskit,
                msg=f"\nCPU: {w=} \n{qiskit_circuit} \n{circuit.gates=}",
            )
            self.assertAlmostEqual(
                simulate_circuit_gpu(w, circuit),
                w_coeff_qiskit,
                msg=f"\nGPU: {w=} \n{qiskit_circuit} \n{circuit.gates=}",
            )
            self.assertAlmostEqual(
                simulate_circuit_hybrid(w, circuit),
                w_coeff_qiskit,
                msg=f"\nHybrid: {w=} \n{qiskit_circuit} \n{circuit.gates=}",
            )
            print("✅", end="", flush=True)

    def test_implementations_against_eachother(self):
        print("\nTesting that all implementations give same result...")
        iterations = 10
        for i in range(iterations):
            qubits = 130
            t_gates = 3
            gates = 1000
            seed = 1234 + i
            circuit, qiskit_circuit, w = self.setup(qubits, gates, t_gates, seed)

            simulate_circuit_res = simulate_circuit(w, circuit)
            simulate_circuit_gpu_res = simulate_circuit_gpu(w, circuit)
            simulate_circuit_hybrid_res = simulate_circuit_hybrid(w, circuit)

            self.assertAlmostEqual(
                simulate_circuit_res,
                simulate_circuit_gpu_res,
                msg=f"\nCPU/GPU: {w=} \n{qiskit_circuit} \n{circuit.gates=}",
            )
            self.assertAlmostEqual(
                simulate_circuit_gpu_res,
                simulate_circuit_hybrid_res,
                msg=f"\nCPU/Hybrid: {w=} \n{qiskit_circuit} \n{circuit.gates=}",
            )
            print("✅", end="", flush=True)


if __name__ == "__main__":
    unittest.main(exit=False)
