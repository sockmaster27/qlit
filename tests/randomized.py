# This module is NOT called from test.rs
import random
import unittest
from qlit import CliffordTCircuit, CliffordTGate, simulate_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QlitRandomizedTests(unittest.TestCase):
    def test_random(self):
        iterations = 1000
        for i in range(iterations):
            qubits = 10
            t_gates = 5
            gates = 100
            seed = 1234 + i
            circuit = CliffordTCircuit.random(qubits, gates, t_gates, seed)
            random.seed(seed)
            w = format(random.getrandbits(qubits), f"00{qubits}b")

            # qiskit
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

            state = Statevector.from_instruction(qiskit_circuit)
            # Remember that Qiskit uses reversed basis states
            w_be = w[::-1]
            w_coeff_qiskit = state.data[int(w_be, 2)]

            # qlit
            w_coeff_qlit = simulate_circuit(w, circuit)

            self.assertAlmostEqual(
                w_coeff_qlit,
                w_coeff_qiskit,
                msg=f"\n{w=} \n{state.data=} \n{qiskit_circuit.draw()}",
            )
            print("✅", end="", flush=True)


if __name__ == "__main__":
    unittest.main(exit=False)
