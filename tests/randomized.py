# This module is NOT called from test.rs
import random
import unittest
from qlit import CliffordCircuit, CliffordGate, coeff_ratio
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class RustStabilizerRandomCiruits(unittest.TestCase):
    def test_random(self):
        iterations = 10000
        actual_circuits_run = 0
        for i in range(iterations):
            n = 10
            m = n * n
            seed = 1234 + i
            circuit = CliffordCircuit.random(n, m, seed)
            random.seed(seed)
            w1 = format(random.getrandbits(n), f"00{n}b")
            w2 = format(random.getrandbits(n), f"00{n}b")

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
            w1_be = w1[::-1]
            w2_be = w2[::-1]
            w1_coeff = state.data[int(w1_be, 2)]
            w2_coeff = state.data[int(w2_be, 2)]
            # Coefficient of w1 must be non-zero
            if w1_coeff < 1e-16:
                continue

            actual_circuits_run += 1

            # qlit
            r = coeff_ratio(w1, w2, circuit)

            self.assertAlmostEqual(
                r,
                w2_coeff / w1_coeff,
                msg=f"\n{w1=}, {w2=} \n{state.data=} \n{qiskit_circuit.draw()}",
            )
        print("Actual amount of circuits tested:", actual_circuits_run)


if __name__ == "__main__":
    unittest.main(exit=False)
