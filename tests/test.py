# This module is called from test.rs
import unittest
import qlit
from qlit import CliffordGate

class RustStabilizerGpu(unittest.TestCase):
    def test_zero(self):
        circuit = qlit.CliffordCircuit(8, [])
        for i in range(0b0000_0000, 0b1111_1111):
            w = format(i, "008b")
            result = qlit.simulate_clifford_circuit_gpu(w, circuit)
            if i == 0b0000_0000:
                self.assertEqual(result, qlit.BasisStateProbability.One)
            else:
                self.assertEqual(result, qlit.BasisStateProbability.Zero)

    def test_flipped(self):
        circuit = qlit.CliffordCircuit(8, [
            CliffordGate.H(0),
            CliffordGate.S(0),
            CliffordGate.S(0),
            CliffordGate.H(0),
        ])
        for i in range(0b0000_0000, 0b1111_1111):
            w = format(i, "008b")
            result = qlit.simulate_clifford_circuit_gpu(w, circuit)
            if i == 0b1000_0000:
                self.assertEqual(result, qlit.BasisStateProbability.One)
            else:
                self.assertEqual(result, qlit.BasisStateProbability.Zero)

    def test_bell_state(self):
        circuit = qlit.CliffordCircuit(8, [
            CliffordGate.H(0),
            CliffordGate.Cnot(0, 1),
        ])
        for i in range(0b0000_0000, 0b1111_1111):
            w = format(i, "008b")
            result = qlit.simulate_clifford_circuit_gpu(w, circuit)
            if i in [0b0000_0000, 0b1100_0000]:
                self.assertEqual(result, qlit.BasisStateProbability.InBetween)
            else:
                self.assertEqual(result, qlit.BasisStateProbability.Zero)

    def test_larger_circuit(self):
        circuit = qlit.CliffordCircuit(8, [
            CliffordGate.H(0),
            CliffordGate.H(1),
            CliffordGate.S(2),
            CliffordGate.H(3),
            CliffordGate.S(1),
            CliffordGate.S(0),
            CliffordGate.Cnot(2, 3),
            CliffordGate.S(1),
            CliffordGate.H(0),
            CliffordGate.S(3),
            CliffordGate.Cnot(1, 0),
            CliffordGate.S(3),
            CliffordGate.H(1),
            CliffordGate.S(3),
            CliffordGate.S(1),
            CliffordGate.S(3),
            CliffordGate.H(1),
            CliffordGate.Cnot(3, 2),
            CliffordGate.H(1),
            CliffordGate.Cnot(3, 1),
        ])
        for i in range(0b0000_0000, 0b1111_1111):
            w = format(i, "008b")
            result = qlit.simulate_clifford_circuit_gpu(w, circuit)
            if i in [
                0b0000_0000,
                0b0100_0000,
                0b1000_0000,
                0b1100_0000,
                0b0011_0000,
                0b0111_0000,
                0b1011_0000,
                0b1111_0000,
            ]:
                self.assertEqual(result, qlit.BasisStateProbability.InBetween)
            else:
                self.assertEqual(result, qlit.BasisStateProbability.Zero)

if __name__ == "__main__":
    unittest.main(exit=False)
