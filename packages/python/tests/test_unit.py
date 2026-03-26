import pytest
from math import sqrt

from qlit import (
    CliffordTCircuit,
    CliffordTGate,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)

DELTA = 1e-6


class TestCircuit:
    def test_invalid_qubit_index(self):
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.X(165)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.X(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Y(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Z(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.H(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.S(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Sdg(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.T(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Tdg(8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Cnot(8, 4)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Cnot(4, 8)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Cz(8, 4)])
        with pytest.raises(ValueError):
            CliffordTCircuit(8, [CliffordTGate.Cz(4, 8)])


class TestCpu:
    def test_mismatched_qubit_number(self):
        circuit = CliffordTCircuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit(w, circuit)

    def test_invalid_basis_state(self):
        circuit = CliffordTCircuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit(w, circuit)

    def test_zero(self):
        circuit = CliffordTCircuit(8, [])
        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_flipped(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(0),
                CliffordTGate.H(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_clifford_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(1),
                CliffordTGate.H(0),
                CliffordTGate.S(2),
                CliffordTGate.H(3),
                CliffordTGate.S(1),
                CliffordTGate.S(0),
                CliffordTGate.Cnot(2, 3),
                CliffordTGate.S(1),
                CliffordTGate.H(0),
                CliffordTGate.S(3),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.S(3),
                CliffordTGate.S(1),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 2),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case (
                    0b0000_0000
                    | 0b0100_0000
                    | 0b1100_0000
                    | 0b0011_0000
                    | 0b0111_0000
                    | 0b1011_0000
                ):
                    expected = 1j / sqrt(8)
                case 0b1000_0000 | 0b1111_0000:
                    expected = -1j / sqrt(8)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.T(0),
                CliffordTGate.H(1),
                CliffordTGate.S(1),
                CliffordTGate.H(3),
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(1),
                CliffordTGate.S(2),
                CliffordTGate.T(1),
                CliffordTGate.H(0),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.S(3),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b1101_0000:
                    expected = 0.25 + 0.25j
                case 0b1000_0000:
                    expected = sqrt(0.125)
                case 0b0100_0000:
                    expected = -sqrt(0.125)
                case 0b1100_0000:
                    expected = 0.25 - 0.25j
                case 0b0001_0000:
                    expected = -0.25 + 0.25j
                case 0b1001_0000:
                    expected = sqrt(0.125) * 1j
                case 0b0101_0000:
                    expected = -sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)


class TestGpu:
    def test_mismatched_qubit_number(self):
        circuit = CliffordTCircuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit_gpu(w, circuit)

    def test_invalid_basis_state(self):
        circuit = CliffordTCircuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit_gpu(w, circuit)

    def test_zero(self):
        circuit = CliffordTCircuit(8, [])
        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_flipped(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(0),
                CliffordTGate.H(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_clifford_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(1),
                CliffordTGate.H(0),
                CliffordTGate.S(2),
                CliffordTGate.H(3),
                CliffordTGate.S(1),
                CliffordTGate.S(0),
                CliffordTGate.Cnot(2, 3),
                CliffordTGate.S(1),
                CliffordTGate.H(0),
                CliffordTGate.S(3),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.S(3),
                CliffordTGate.S(1),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 2),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case (
                    0b0000_0000
                    | 0b0100_0000
                    | 0b1100_0000
                    | 0b0011_0000
                    | 0b0111_0000
                    | 0b1011_0000
                ):
                    expected = 1j / sqrt(8)
                case 0b1000_0000 | 0b1111_0000:
                    expected = -1j / sqrt(8)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.T(0),
                CliffordTGate.H(1),
                CliffordTGate.S(1),
                CliffordTGate.H(3),
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(1),
                CliffordTGate.S(2),
                CliffordTGate.T(1),
                CliffordTGate.H(0),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.S(3),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b1101_0000:
                    expected = 0.25 + 0.25j
                case 0b1000_0000:
                    expected = sqrt(0.125)
                case 0b0100_0000:
                    expected = -sqrt(0.125)
                case 0b1100_0000:
                    expected = 0.25 - 0.25j
                case 0b0001_0000:
                    expected = -0.25 + 0.25j
                case 0b1001_0000:
                    expected = sqrt(0.125) * 1j
                case 0b0101_0000:
                    expected = -sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)


class TestHybrid:
    def test_mismatched_qubit_number(self):
        circuit = CliffordTCircuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit_hybrid(w, circuit)

    def test_invalid_basis_state(self):
        circuit = CliffordTCircuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit_hybrid(w, circuit)

    def test_zero(self):
        circuit = CliffordTCircuit(8, [])
        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_flipped(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(0),
                CliffordTGate.H(0),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_clifford_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(1),
                CliffordTGate.H(0),
                CliffordTGate.S(2),
                CliffordTGate.H(3),
                CliffordTGate.S(1),
                CliffordTGate.S(0),
                CliffordTGate.Cnot(2, 3),
                CliffordTGate.S(1),
                CliffordTGate.H(0),
                CliffordTGate.S(3),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.S(3),
                CliffordTGate.S(1),
                CliffordTGate.S(3),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 2),
                CliffordTGate.H(1),
                CliffordTGate.Cnot(3, 1),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case (
                    0b0000_0000
                    | 0b0100_0000
                    | 0b1100_0000
                    | 0b0011_0000
                    | 0b0111_0000
                    | 0b1011_0000
                ):
                    expected = 1j / sqrt(8)
                case 0b1000_0000 | 0b1111_0000:
                    expected = -1j / sqrt(8)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)

    def test_larger_circuit(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.T(0),
                CliffordTGate.H(1),
                CliffordTGate.S(1),
                CliffordTGate.H(3),
                CliffordTGate.H(0),
                CliffordTGate.S(0),
                CliffordTGate.S(1),
                CliffordTGate.S(2),
                CliffordTGate.T(1),
                CliffordTGate.H(0),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.S(3),
            ],
        )

        for i in range(0b0000_0000, 0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b1101_0000:
                    expected = 0.25 + 0.25j
                case 0b1000_0000:
                    expected = sqrt(0.125)
                case 0b0100_0000:
                    expected = -sqrt(0.125)
                case 0b1100_0000:
                    expected = 0.25 - 0.25j
                case 0b0001_0000:
                    expected = -0.25 + 0.25j
                case 0b1001_0000:
                    expected = sqrt(0.125) * 1j
                case 0b0101_0000:
                    expected = -sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, abs=DELTA)
