from cmath import rect
from math import cos, pi, sin, sqrt

import pytest
from qlit import (
    Circuit,
    Gate,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)

DELTA = 1e-6


class TestCircuit:
    def test_invalid_qubit_index(self):
        with pytest.raises(ValueError):
            Circuit(8, [Gate.X(165)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.X(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Y(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Z(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.H(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.S(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Sdg(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.T(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Tdg(8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Rx(1.0, 8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Ry(1.0, 8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Rz(1.0, 8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Cnot(8, 4)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Cnot(4, 8)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Cz(8, 4)])
        with pytest.raises(ValueError):
            Circuit(8, [Gate.Cz(4, 8)])


class TestCpu:
    def test_mismatched_qubit_number(self):
        circuit = Circuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit(w, circuit)

    def test_invalid_basis_state(self):
        circuit = Circuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit(w, circuit)

    def test_zero(self):
        circuit = Circuit(8, [])
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
        circuit = Circuit(8, [Gate.X(0)])

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
        circuit = Circuit(8, [Gate.Y(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
                Gate.S(0),
                Gate.H(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
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
        circuit = Circuit(
            8,
            [
                Gate.H(1),
                Gate.H(0),
                Gate.S(2),
                Gate.H(3),
                Gate.S(1),
                Gate.S(0),
                Gate.Cnot(2, 3),
                Gate.S(1),
                Gate.H(0),
                Gate.S(3),
                Gate.Cnot(1, 0),
                Gate.S(3),
                Gate.H(1),
                Gate.S(3),
                Gate.S(1),
                Gate.S(3),
                Gate.H(1),
                Gate.Cnot(3, 2),
                Gate.H(1),
                Gate.Cnot(3, 1),
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
        circuit = Circuit(8, [*[Gate.T(0)] * 5])

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
        circuit = Circuit(8, [*[Gate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rx_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Rx(theta, 0), Gate.H(1), Gate.Rx(phi, 1)])

        # |+> is an eigenstate of Rx, so qubit 1 only picks up a phase
        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b1000_0000 | 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_ry_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Ry(theta, 0), Gate.H(1), Gate.Ry(phi, 1)])

        amplitude = sqrt(0.5)
        c0, s0 = cos(theta / 2), sin(theta / 2)
        c1, s1 = cos(phi / 2), sin(phi / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = amplitude * c0 * (c1 - s1)
                case 0b0100_0000:
                    expected = amplitude * c0 * (c1 + s1)
                case 0b1000_0000:
                    expected = amplitude * s0 * (c1 - s1)
                case 0b1100_0000:
                    expected = amplitude * s0 * (c1 + s1)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rz_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(
            8, [Gate.H(0), Gate.Rz(theta, 0), Gate.H(0), Gate.H(1), Gate.Rz(phi, 1)]
        )

        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), phi / 2)
                case 0b1000_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rotations_by_pi(self):
        # Rx(π) = -iX, Ry(π) = -iY and Rz(π) = -iZ
        circuit = Circuit(8, [Gate.Rx(pi, 0), Gate.Ry(pi, 1), Gate.H(2), Gate.Rz(pi, 2)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i:
                case 0b1100_0000:
                    expected = -sqrt(0.5)
                case 0b1110_0000:
                    expected = sqrt(0.5)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_mixed_rotations(self):
        theta, alpha, beta = 0.8, 0.5, 1.0
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
                Gate.Rz(theta, 0),
                Gate.Rz(theta, 1),
                Gate.Ry(alpha, 2),
                Gate.Ry(beta, 2),
                Gate.Rx(alpha, 3),
                Gate.Rx(beta, 3),
            ],
        )

        # Rotations about the same axis add up
        amplitude = sqrt(0.5)
        c, s = cos((alpha + beta) / 2), sin((alpha + beta) / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit(w, circuit)
            match i & 0b1100_0000:
                case 0b0000_0000:
                    bell = rect(amplitude, -theta)
                case 0b1100_0000:
                    bell = rect(amplitude, theta)
                case _:
                    bell = 0
            ry = c if i & 0b0010_0000 == 0 else s
            rx = c if i & 0b0001_0000 == 0 else -1j * s
            match i & 0b0000_1111:
                case 0b0000:
                    expected = bell * ry * rx
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = Circuit(8, [Gate.T(0), Gate.H(1), Gate.H(2), Gate.Cnot(1, 0), Gate.Cnot(2, 1)])

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
        circuit = Circuit(8, [Gate.H(0), Gate.H(1), Gate.Cnot(1, 0), Gate.T(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.T(0),
                Gate.H(1),
                Gate.S(1),
                Gate.H(3),
                Gate.H(0),
                Gate.S(0),
                Gate.S(1),
                Gate.S(2),
                Gate.T(1),
                Gate.H(0),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.S(3),
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
        circuit = Circuit(
            8,
            [
                Gate.Tdg(0),
                Gate.H(2),
                Gate.H(1),
                Gate.Y(2),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.Tdg(0),
                Gate.Cnot(2, 1),
                Gate.Sdg(1),
                Gate.S(2),
                Gate.X(0),
                Gate.S(0),
                Gate.Y(0),
                Gate.T(1),
                Gate.Tdg(2),
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
        circuit = Circuit(
            8,
            [Gate.H(q) for q in range(8)]
            + [Gate.Cz(q, q + 1) for q in range(7)],
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
        circuit = Circuit(
            70,
            [Gate.H(q) for q in range(70)]
            + [Gate.Cz(0, 64), Gate.Cz(0, 69)],
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
        circuit = Circuit(
            300,
            [
                Gate.H(1),
                Gate.T(1),
                Gate.H(78),
                Gate.T(78),
                Gate.H(123),
                Gate.T(123),
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
        circuit = Circuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit_gpu(w, circuit)

    def test_invalid_basis_state(self):
        circuit = Circuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit_gpu(w, circuit)

    def test_zero(self):
        circuit = Circuit(8, [])
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
        circuit = Circuit(8, [Gate.X(0)])

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
        circuit = Circuit(8, [Gate.Y(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
                Gate.S(0),
                Gate.H(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
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
        circuit = Circuit(
            8,
            [
                Gate.H(1),
                Gate.H(0),
                Gate.S(2),
                Gate.H(3),
                Gate.S(1),
                Gate.S(0),
                Gate.Cnot(2, 3),
                Gate.S(1),
                Gate.H(0),
                Gate.S(3),
                Gate.Cnot(1, 0),
                Gate.S(3),
                Gate.H(1),
                Gate.S(3),
                Gate.S(1),
                Gate.S(3),
                Gate.H(1),
                Gate.Cnot(3, 2),
                Gate.H(1),
                Gate.Cnot(3, 1),
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
        circuit = Circuit(8, [*[Gate.T(0)] * 5])

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
        circuit = Circuit(8, [*[Gate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rx_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Rx(theta, 0), Gate.H(1), Gate.Rx(phi, 1)])

        # |+> is an eigenstate of Rx, so qubit 1 only picks up a phase
        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b1000_0000 | 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_ry_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Ry(theta, 0), Gate.H(1), Gate.Ry(phi, 1)])

        amplitude = sqrt(0.5)
        c0, s0 = cos(theta / 2), sin(theta / 2)
        c1, s1 = cos(phi / 2), sin(phi / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = amplitude * c0 * (c1 - s1)
                case 0b0100_0000:
                    expected = amplitude * c0 * (c1 + s1)
                case 0b1000_0000:
                    expected = amplitude * s0 * (c1 - s1)
                case 0b1100_0000:
                    expected = amplitude * s0 * (c1 + s1)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rz_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(
            8, [Gate.H(0), Gate.Rz(theta, 0), Gate.H(0), Gate.H(1), Gate.Rz(phi, 1)]
        )

        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), phi / 2)
                case 0b1000_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rotations_by_pi(self):
        # Rx(π) = -iX, Ry(π) = -iY and Rz(π) = -iZ
        circuit = Circuit(8, [Gate.Rx(pi, 0), Gate.Ry(pi, 1), Gate.H(2), Gate.Rz(pi, 2)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i:
                case 0b1100_0000:
                    expected = -sqrt(0.5)
                case 0b1110_0000:
                    expected = sqrt(0.5)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_mixed_rotations(self):
        theta, alpha, beta = 0.8, 0.5, 1.0
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
                Gate.Rz(theta, 0),
                Gate.Rz(theta, 1),
                Gate.Ry(alpha, 2),
                Gate.Ry(beta, 2),
                Gate.Rx(alpha, 3),
                Gate.Rx(beta, 3),
            ],
        )

        # Rotations about the same axis add up
        amplitude = sqrt(0.5)
        c, s = cos((alpha + beta) / 2), sin((alpha + beta) / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_gpu(w, circuit)
            match i & 0b1100_0000:
                case 0b0000_0000:
                    bell = rect(amplitude, -theta)
                case 0b1100_0000:
                    bell = rect(amplitude, theta)
                case _:
                    bell = 0
            ry = c if i & 0b0010_0000 == 0 else s
            rx = c if i & 0b0001_0000 == 0 else -1j * s
            match i & 0b0000_1111:
                case 0b0000:
                    expected = bell * ry * rx
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = Circuit(8, [Gate.T(0), Gate.H(1), Gate.H(2), Gate.Cnot(1, 0), Gate.Cnot(2, 1)])

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
        circuit = Circuit(8, [Gate.H(0), Gate.H(1), Gate.Cnot(1, 0), Gate.T(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.T(0),
                Gate.H(1),
                Gate.S(1),
                Gate.H(3),
                Gate.H(0),
                Gate.S(0),
                Gate.S(1),
                Gate.S(2),
                Gate.T(1),
                Gate.H(0),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.S(3),
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
        circuit = Circuit(
            8,
            [
                Gate.Tdg(0),
                Gate.H(2),
                Gate.H(1),
                Gate.Y(2),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.Tdg(0),
                Gate.Cnot(2, 1),
                Gate.Sdg(1),
                Gate.S(2),
                Gate.X(0),
                Gate.S(0),
                Gate.Y(0),
                Gate.T(1),
                Gate.Tdg(2),
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
        circuit = Circuit(
            8,
            [Gate.H(q) for q in range(8)]
            + [Gate.Cz(q, q + 1) for q in range(7)],
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
        circuit = Circuit(
            70,
            [Gate.H(q) for q in range(70)]
            + [Gate.Cz(0, 64), Gate.Cz(0, 69)],
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
        circuit = Circuit(
            300,
            [
                Gate.H(1),
                Gate.T(1),
                Gate.H(78),
                Gate.T(78),
                Gate.H(123),
                Gate.T(123),
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
        circuit = Circuit(8, [])
        w = "000000000"
        with pytest.raises(ValueError):
            simulate_circuit_hybrid(w, circuit)

    def test_invalid_basis_state(self):
        circuit = Circuit(8, [])
        w = "010a0100"
        with pytest.raises(ValueError):
            simulate_circuit_hybrid(w, circuit)

    def test_zero(self):
        circuit = Circuit(8, [])
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
        circuit = Circuit(8, [Gate.X(0)])

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
        circuit = Circuit(8, [Gate.Y(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.S(0),
                Gate.S(0),
                Gate.H(0),
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
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
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
        circuit = Circuit(
            8,
            [
                Gate.H(1),
                Gate.H(0),
                Gate.S(2),
                Gate.H(3),
                Gate.S(1),
                Gate.S(0),
                Gate.Cnot(2, 3),
                Gate.S(1),
                Gate.H(0),
                Gate.S(3),
                Gate.Cnot(1, 0),
                Gate.S(3),
                Gate.H(1),
                Gate.S(3),
                Gate.S(1),
                Gate.S(3),
                Gate.H(1),
                Gate.Cnot(3, 2),
                Gate.H(1),
                Gate.Cnot(3, 1),
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
        circuit = Circuit(8, [*[Gate.T(0)] * 5])

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
        circuit = Circuit(8, [*[Gate.Tdg(0)] * 5])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = 1
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rx_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Rx(theta, 0), Gate.H(1), Gate.Rx(phi, 1)])

        # |+> is an eigenstate of Rx, so qubit 1 only picks up a phase
        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000 | 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b1000_0000 | 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_ry_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(8, [Gate.Ry(theta, 0), Gate.H(1), Gate.Ry(phi, 1)])

        amplitude = sqrt(0.5)
        c0, s0 = cos(theta / 2), sin(theta / 2)
        c1, s1 = cos(phi / 2), sin(phi / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = amplitude * c0 * (c1 - s1)
                case 0b0100_0000:
                    expected = amplitude * c0 * (c1 + s1)
                case 0b1000_0000:
                    expected = amplitude * s0 * (c1 - s1)
                case 0b1100_0000:
                    expected = amplitude * s0 * (c1 + s1)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rz_gates(self):
        theta, phi = 1.0, 2.5
        circuit = Circuit(
            8, [Gate.H(0), Gate.Rz(theta, 0), Gate.H(0), Gate.H(1), Gate.Rz(phi, 1)]
        )

        amplitude = sqrt(0.5)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b0000_0000:
                    expected = rect(amplitude * cos(theta / 2), -phi / 2)
                case 0b0100_0000:
                    expected = rect(amplitude * cos(theta / 2), phi / 2)
                case 0b1000_0000:
                    expected = rect(amplitude * sin(theta / 2), -phi / 2 - pi / 2)
                case 0b1100_0000:
                    expected = rect(amplitude * sin(theta / 2), phi / 2 - pi / 2)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_rotations_by_pi(self):
        # Rx(π) = -iX, Ry(π) = -iY and Rz(π) = -iZ
        circuit = Circuit(8, [Gate.Rx(pi, 0), Gate.Ry(pi, 1), Gate.H(2), Gate.Rz(pi, 2)])

        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i:
                case 0b1100_0000:
                    expected = -sqrt(0.5)
                case 0b1110_0000:
                    expected = sqrt(0.5)
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_mixed_rotations(self):
        theta, alpha, beta = 0.8, 0.5, 1.0
        circuit = Circuit(
            8,
            [
                Gate.H(0),
                Gate.Cnot(0, 1),
                Gate.Rz(theta, 0),
                Gate.Rz(theta, 1),
                Gate.Ry(alpha, 2),
                Gate.Ry(beta, 2),
                Gate.Rx(alpha, 3),
                Gate.Rx(beta, 3),
            ],
        )

        # Rotations about the same axis add up
        amplitude = sqrt(0.5)
        c, s = cos((alpha + beta) / 2), sin((alpha + beta) / 2)
        for i in range(0b1111_1111 + 1):
            w = format(i, "08b")
            result = simulate_circuit_hybrid(w, circuit)
            match i & 0b1100_0000:
                case 0b0000_0000:
                    bell = rect(amplitude, -theta)
                case 0b1100_0000:
                    bell = rect(amplitude, theta)
                case _:
                    bell = 0
            ry = c if i & 0b0010_0000 == 0 else s
            rx = c if i & 0b0001_0000 == 0 else -1j * s
            match i & 0b0000_1111:
                case 0b0000:
                    expected = bell * ry * rx
                case _:
                    expected = 0
            assert result == pytest.approx(expected, rel=DELTA)

    def test_smaller_circuit_1(self):
        circuit = Circuit(8, [Gate.T(0), Gate.H(1), Gate.H(2), Gate.Cnot(1, 0), Gate.Cnot(2, 1)])

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
        circuit = Circuit(8, [Gate.H(0), Gate.H(1), Gate.Cnot(1, 0), Gate.T(0)])

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
        circuit = Circuit(
            8,
            [
                Gate.T(0),
                Gate.H(1),
                Gate.S(1),
                Gate.H(3),
                Gate.H(0),
                Gate.S(0),
                Gate.S(1),
                Gate.S(2),
                Gate.T(1),
                Gate.H(0),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.S(3),
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
        circuit = Circuit(
            8,
            [
                Gate.Tdg(0),
                Gate.H(2),
                Gate.H(1),
                Gate.Y(2),
                Gate.Cnot(1, 0),
                Gate.T(0),
                Gate.Tdg(0),
                Gate.Cnot(2, 1),
                Gate.Sdg(1),
                Gate.S(2),
                Gate.X(0),
                Gate.S(0),
                Gate.Y(0),
                Gate.T(1),
                Gate.Tdg(2),
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
        circuit = Circuit(
            8,
            [Gate.H(q) for q in range(8)]
            + [Gate.Cz(q, q + 1) for q in range(7)],
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
        circuit = Circuit(
            70,
            [Gate.H(q) for q in range(70)]
            + [Gate.Cz(0, 64), Gate.Cz(0, 69)],
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
        circuit = Circuit(
            300,
            [
                Gate.H(1),
                Gate.T(1),
                Gate.H(78),
                Gate.T(78),
                Gate.H(123),
                Gate.T(123),
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
