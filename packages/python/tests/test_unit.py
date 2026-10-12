from math import sqrt

import pytest
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
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_x_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.X(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_y_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.Y(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_t_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.T(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_tdg_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.T(0), CliffordTGate.H(1), CliffordTGate.H(2), CliffordTGate.Cnot(1, 0), CliffordTGate.Cnot(2, 1)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000:
                    expected = 0.5
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_2(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.H(0), CliffordTGate.H(1), CliffordTGate.Cnot(1, 0), CliffordTGate.T(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = 0.5
                case 0b1000_0000 | 0b1100_0000:
                    expected = sqrt(0.125) + sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_1(self):
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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_2(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.Tdg(0),
                CliffordTGate.H(2),
                CliffordTGate.H(1),
                CliffordTGate.Y(2),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.Tdg(0),
                CliffordTGate.Cnot(2, 1),
                CliffordTGate.Sdg(1),
                CliffordTGate.S(2),
                CliffordTGate.X(0),
                CliffordTGate.S(0),
                CliffordTGate.Y(0),
                CliffordTGate.T(1),
                CliffordTGate.Tdg(2),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = -0.5j
                case 0b1100_0000:
                    expected = sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b1010_0000:
                    expected = -sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b0110_0000:
                    expected = 0.5j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_linear_cluster_state(self):
        circuit = CliffordTCircuit(
            8,
            [CliffordTGate.H(q) for q in range(8)]
            + [CliffordTGate.Cz(q, q + 1) for q in range(7)],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            adjacent_pairs = sum(w[q] == "1" and w[q + 1] == "1" for q in range(7))
            match adjacent_pairs % 2:
                case 0:
                    expected = 1 / 16
                case _:
                    expected = -1 / 16
            assert result == pytest.approx(expected, rel=DELTA)

    def test_multi_block_phase(self):
        circuit = CliffordTCircuit(
            70,
            [CliffordTGate.H(q) for q in range(70)]
            + [CliffordTGate.Cz(0, 64), CliffordTGate.Cz(0, 69)],
        )

        amplitude = 2**-35
        cases = [
            ([], amplitude),
            ([0], amplitude),
            ([64], amplitude),
            ([0, 64], -amplitude),  # Edge 0-64
            ([0, 69], -amplitude),  # Edge 0-69
            ([0, 64, 69], amplitude),  # Edges 0-64 and 0-69
            ([1, 2, 3], amplitude),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(70))
            result = simulate_circuit(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)

    def test_large_tableau(self):
        circuit = CliffordTCircuit(
            300,
            [
                CliffordTGate.H(1),
                CliffordTGate.T(1),
                CliffordTGate.H(78),
                CliffordTGate.T(78),
                CliffordTGate.H(123),
                CliffordTGate.T(123),
            ],
        )

        cases = [
            ([], 1 / sqrt(8)),
            ([1], (1 + 1j) / 4),
            ([1, 78], 1j / sqrt(8)),
            ([1, 78, 123], (-1 + 1j) / 4),
            ([42], 0),
            ([1, 42], 0),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(300))
            result = simulate_circuit(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)


@pytest.mark.gpu
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
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_x_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.X(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_y_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.Y(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_t_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.T(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_tdg_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.T(0), CliffordTGate.H(1), CliffordTGate.H(2), CliffordTGate.Cnot(1, 0), CliffordTGate.Cnot(2, 1)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000:
                    expected = 0.5
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_2(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.H(0), CliffordTGate.H(1), CliffordTGate.Cnot(1, 0), CliffordTGate.T(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = 0.5
                case 0b1000_0000 | 0b1100_0000:
                    expected = sqrt(0.125) + sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_1(self):
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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_2(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.Tdg(0),
                CliffordTGate.H(2),
                CliffordTGate.H(1),
                CliffordTGate.Y(2),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.Tdg(0),
                CliffordTGate.Cnot(2, 1),
                CliffordTGate.Sdg(1),
                CliffordTGate.S(2),
                CliffordTGate.X(0),
                CliffordTGate.S(0),
                CliffordTGate.Y(0),
                CliffordTGate.T(1),
                CliffordTGate.Tdg(2),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = -0.5j
                case 0b1100_0000:
                    expected = sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b1010_0000:
                    expected = -sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b0110_0000:
                    expected = 0.5j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_linear_cluster_state(self):
        circuit = CliffordTCircuit(
            8,
            [CliffordTGate.H(q) for q in range(8)]
            + [CliffordTGate.Cz(q, q + 1) for q in range(7)],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            adjacent_pairs = sum(w[q] == "1" and w[q + 1] == "1" for q in range(7))
            match adjacent_pairs % 2:
                case 0:
                    expected = 1 / 16
                case _:
                    expected = -1 / 16
            assert result == pytest.approx(expected, rel=DELTA)

    def test_multi_block_phase(self):
        circuit = CliffordTCircuit(
            70,
            [CliffordTGate.H(q) for q in range(70)]
            + [CliffordTGate.Cz(0, 64), CliffordTGate.Cz(0, 69)],
        )

        amplitude = 2**-35
        cases = [
            ([], amplitude),
            ([0], amplitude),
            ([64], amplitude),
            ([0, 64], -amplitude),  # Edge 0-64
            ([0, 69], -amplitude),  # Edge 0-69
            ([0, 64, 69], amplitude),  # Edges 0-64 and 0-69
            ([1, 2, 3], amplitude),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(70))
            result = simulate_circuit_gpu(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)

    def test_large_tableau(self):
        circuit = CliffordTCircuit(
            300,
            [
                CliffordTGate.H(1),
                CliffordTGate.T(1),
                CliffordTGate.H(78),
                CliffordTGate.T(78),
                CliffordTGate.H(123),
                CliffordTGate.T(123),
            ],
        )

        cases = [
            ([], 1 / sqrt(8)),
            ([1], (1 + 1j) / 4),
            ([1, 78], 1j / sqrt(8)),
            ([1, 78, 123], (-1 + 1j) / 4),
            ([42], 0),
            ([1, 42], 0),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(300))
            result = simulate_circuit_gpu(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)


@pytest.mark.gpu
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
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_x_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.X(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_y_gate(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.Y(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_imaginary(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.S(0),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1 / sqrt(2)
                case 0b1000_0000:
                    expected = 1j / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b1000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_bell_state(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.H(0),
                CliffordTGate.Cnot(0, 1),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000:
                    expected = 1 / sqrt(2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_t_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.T(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_only_tdg_gates(self):
        circuit = CliffordTCircuit(8, [*[CliffordTGate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.T(0), CliffordTGate.H(1), CliffordTGate.H(2), CliffordTGate.Cnot(1, 0), CliffordTGate.Cnot(2, 1)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000:
                    expected = 0.5
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_2(self):
        circuit = CliffordTCircuit(8, [CliffordTGate.H(0), CliffordTGate.H(1), CliffordTGate.Cnot(1, 0), CliffordTGate.T(0)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = 0.5
                case 0b1000_0000 | 0b1100_0000:
                    expected = sqrt(0.125) + sqrt(0.125) * 1j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_1(self):
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

        for i in range(0b1111_1111 + 1):
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
            assert result == pytest.approx(expected, rel=DELTA)

    def test_larger_circuit_2(self):
        circuit = CliffordTCircuit(
            8,
            [
                CliffordTGate.Tdg(0),
                CliffordTGate.H(2),
                CliffordTGate.H(1),
                CliffordTGate.Y(2),
                CliffordTGate.Cnot(1, 0),
                CliffordTGate.T(0),
                CliffordTGate.Tdg(0),
                CliffordTGate.Cnot(2, 1),
                CliffordTGate.Sdg(1),
                CliffordTGate.S(2),
                CliffordTGate.X(0),
                CliffordTGate.S(0),
                CliffordTGate.Y(0),
                CliffordTGate.T(1),
                CliffordTGate.Tdg(2),
            ],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = -0.5j
                case 0b1100_0000:
                    expected = sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b1010_0000:
                    expected = -sqrt(2) / 4 - sqrt(2) / 4 * 1j
                case 0b0110_0000:
                    expected = 0.5j
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_linear_cluster_state(self):
        circuit = CliffordTCircuit(
            8,
            [CliffordTGate.H(q) for q in range(8)]
            + [CliffordTGate.Cz(q, q + 1) for q in range(7)],
        )

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            adjacent_pairs = sum(w[q] == "1" and w[q + 1] == "1" for q in range(7))
            match adjacent_pairs % 2:
                case 0:
                    expected = 1 / 16
                case _:
                    expected = -1 / 16
            assert result == pytest.approx(expected, rel=DELTA)

    def test_multi_block_phase(self):
        circuit = CliffordTCircuit(
            70,
            [CliffordTGate.H(q) for q in range(70)]
            + [CliffordTGate.Cz(0, 64), CliffordTGate.Cz(0, 69)],
        )

        amplitude = 2**-35
        cases = [
            ([], amplitude),
            ([0], amplitude),
            ([64], amplitude),
            ([0, 64], -amplitude),  # Edge 0-64
            ([0, 69], -amplitude),  # Edge 0-69
            ([0, 64, 69], amplitude),  # Edges 0-64 and 0-69
            ([1, 2, 3], amplitude),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(70))
            result = simulate_circuit_hybrid(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)

    def test_large_tableau(self):
        circuit = CliffordTCircuit(
            300,
            [
                CliffordTGate.H(1),
                CliffordTGate.T(1),
                CliffordTGate.H(78),
                CliffordTGate.T(78),
                CliffordTGate.H(123),
                CliffordTGate.T(123),
            ],
        )

        cases = [
            ([], 1 / sqrt(8)),
            ([1], (1 + 1j) / 4),
            ([1, 78], 1j / sqrt(8)),
            ([1, 78, 123], (-1 + 1j) / 4),
            ([42], 0),
            ([1, 42], 0),
        ]
        for set_bits, expected in cases:
            w = "".join("1" if q in set_bits else "0" for q in range(300))
            result = simulate_circuit_hybrid(w, circuit)
            assert result == pytest.approx(expected, rel=DELTA)
