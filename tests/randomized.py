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
            n = 10
            m = 10
            seed = 1234 + i
            circuit = CliffordTCircuit.random(n, m, seed)
            random.seed(seed)
            w = format(random.getrandbits(n), f"00{n}b")

            # qiskit
            qiskit_circuit = QuantumCircuit(n)
            for gate in circuit.gates:
                match gate:
                    case CliffordTGate.H(a):
                        qiskit_circuit.h(a)
                    case CliffordTGate.S(a):
                        qiskit_circuit.s(a)
                    case CliffordTGate.Cnot(a, b):
                        qiskit_circuit.cx(a, b)
                    case CliffordTGate.T(a):
                        qiskit_circuit.t(a)

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
            print("done")


if __name__ == "__main__":
    unittest.main(exit=False)
