# This module is NOT called from test.rs
import random
import unittest
from qlit import CliffordCircuit, CliffordGate, clifford_phase
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QlitRandomizedTests(unittest.TestCase):
    def test_random(self):
        iterations = 10000
        for i in range(iterations):
            n = 2
            m = n * n
            seed = 1234 + i
            circuit = CliffordCircuit.random(n, m, seed)
            random.seed(seed)
            w = format(random.getrandbits(n), f"00{n}b")

            # qiskit
            qiskit_circuit = QuantumCircuit(n)
            for gate in circuit.gates:
                match gate:
                    case CliffordGate.H(a):
                        qiskit_circuit.h(a)
                    case CliffordGate.S(a):
                        qiskit_circuit.s(a)
                    case CliffordGate.Cnot(a, b):
                        qiskit_circuit.cx(a, b)

            state = Statevector.from_instruction(qiskit_circuit)
            # Remember that Qiskit uses reversed basis states
            w_be = w[::-1]
            w_coeff_qiskit = state.data[int(w_be, 2)]

            # qlit
            w_coeff_qlit = clifford_phase(w, circuit)

            self.assertAlmostEqual(
                w_coeff_qlit,
                w_coeff_qiskit,
                msg=f"\n{w=} \n{state.data=} \n{qiskit_circuit.draw()}",
            )


if __name__ == "__main__":
    unittest.main(exit=False)
