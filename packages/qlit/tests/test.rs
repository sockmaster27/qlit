use std::f64::consts::PI;

use num_complex::Complex;
use qlit::{CircuitCreationError, CliffordTCircuit, CliffordTGate::*};

const DELTA: f64 = 1e-6;
const ABS_DELTA: f64 = 1e-12;

#[test]
fn invalid_qubit_index() {
    assert_eq!(
        CliffordTCircuit::new(8, [X(165)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 165,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [X(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Y(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Z(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [H(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [S(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Sdg(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [T(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Tdg(8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Cnot(8, 4)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Cnot(4, 8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Cz(8, 4)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
    assert_eq!(
        CliffordTCircuit::new(8, [Cz(4, 8)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 8,
            qubits: 8
        })
    );
}

mod cpu {
    use qlit::{BitString, simulate_circuit};

    use super::*;

    fn assert_almost_eq(result: Complex<f64>, expected: Complex<f64>, w: &BitString) {
        super::assert_almost_eq(result, expected, &w.iter().collect::<Vec<_>>());
    }

    #[test]
    #[should_panic]
    fn mismatched_qubit_number() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();
        let w = BitString::zero(9);
        simulate_circuit(&w, &circuit);
    }

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn x_gate() {
        let circuit = CliffordTCircuit::new(8, [X(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn y_gate() {
        let circuit = CliffordTCircuit::new(8, [Y(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::I,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE / 2_f64.sqrt(),
                0b1000_0000 => Complex::I / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 => Complex::ONE / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_clifford_circuit() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 | 0b1100_0000 | 0b0011_0000 | 0b0111_0000
                | 0b1011_0000 => Complex::I / 8_f64.sqrt(),
                0b1000_0000 | 0b1111_0000 => -Complex::I / 8_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_t_gates() {
        let circuit = CliffordTCircuit::new(8, [T(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_tdg_gates() {
        let circuit = CliffordTCircuit::new(8, [Tdg(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_1() {
        let circuit = CliffordTCircuit::new(8, [T(0), H(1), H(2), Cnot(1, 0), Cnot(2, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000 => {
                    Complex { re: 0.5, im: 0.0 }
                }
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_2() {
        let circuit = CliffordTCircuit::new(8, [H(0), H(1), Cnot(1, 0), T(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 => Complex { re: 0.5, im: 0.0 },
                0b1000_0000 | 0b1100_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_1() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                T(0),
                H(1),
                S(1),
                H(3),
                H(0),
                S(0),
                S(1),
                S(2),
                T(1),
                H(0),
                Cnot(1, 0),
                T(0),
                S(3),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1101_0000 => Complex { re: 0.25, im: 0.25 },
                0b1000_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b0100_0000 => Complex {
                    re: -0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b1100_0000 => Complex {
                    re: 0.25,
                    im: -0.25,
                },
                0b0001_0000 => Complex {
                    re: -0.25,
                    im: 0.25,
                },
                0b1001_0000 => Complex {
                    re: 0.0,
                    im: 0.125_f64.sqrt(),
                },
                0b0101_0000 => Complex {
                    re: 0.0,
                    im: -0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_2() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                Tdg(0),
                H(2),
                H(1),
                Y(2),
                Cnot(1, 0),
                T(0),
                Tdg(0),
                Cnot(2, 1),
                Sdg(1),
                S(2),
                X(0),
                S(0),
                Y(0),
                T(1),
                Tdg(2),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex { re: 0.0, im: -0.5 },
                0b1100_0000 => Complex {
                    re: 2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b1010_0000 => Complex {
                    re: -2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b0110_0000 => Complex { re: 0.0, im: 0.5 },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn linear_cluster_state() {
        let circuit =
            CliffordTCircuit::new(8, (0..8).map(H).chain((0..7).map(|q| Cz(q, q + 1)))).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = BitString::from_u8_ltr(i);

            let result = simulate_circuit(&w, &circuit);

            let adjacent_pairs = (0..7).filter(|&q| w.get(q) && w.get(q + 1)).count();
            let expected = match adjacent_pairs % 2 {
                0 => Complex::ONE / 16.0,
                _ => -Complex::ONE / 16.0,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    /// Phase shifts from row multiplication between rows across different bit blocks.
    #[test]
    fn multi_block_phase() {
        let circuit =
            CliffordTCircuit::new(70, (0..70).map(H).chain([Cz(0, 64), Cz(0, 69)])).unwrap();

        let amplitude = 2_f64.powi(-35);
        let cases: [(&[usize], Complex<f64>); 7] = [
            (&[], Complex::ONE * amplitude),
            (&[0], Complex::ONE * amplitude),
            (&[64], Complex::ONE * amplitude),
            (&[0, 64], -Complex::ONE * amplitude), // Edge 0-64
            (&[0, 69], -Complex::ONE * amplitude), // Edge 0-69
            (&[0, 64, 69], Complex::ONE * amplitude), // Edges 0-64 and 0-69
            (&[1, 2, 3], Complex::ONE * amplitude),
        ];
        for (set_bits, expected) in cases {
            let mut w = BitString::zero(70);
            for &b in set_bits {
                w.set(b);
            }

            let result = simulate_circuit(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn large_tableau() {
        let circuit =
            CliffordTCircuit::new(300, [H(1), T(1), H(78), T(78), H(123), T(123)]).unwrap();

        let amplitude = 2_f64.powf(-1.5);
        let cases: [(&[usize], Complex<f64>); 6] = [
            (&[], Complex::from_polar(amplitude, 0.0)),
            (&[1], Complex::from_polar(amplitude, PI / 4.0)),
            (&[1, 78], Complex::from_polar(amplitude, 2.0 * PI / 4.0)),
            (
                &[1, 78, 123],
                Complex::from_polar(amplitude, 3.0 * PI / 4.0),
            ),
            (&[42], Complex::ZERO),
            (&[1, 42], Complex::ZERO),
        ];
        for (set_bits, expected) in cases {
            let mut w = BitString::zero(300);
            for &b in set_bits {
                w.set(b);
            }

            let result = simulate_circuit(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use qlit::simulate_circuit_gpu;

    use super::*;

    #[test]
    #[should_panic]
    fn mismatched_qubit_number() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();
        let w = [false; 9];
        simulate_circuit_gpu(&w, &circuit);
    }

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn x_gate() {
        let circuit = CliffordTCircuit::new(8, [X(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn y_gate() {
        let circuit = CliffordTCircuit::new(8, [Y(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::I,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE / 2_f64.sqrt(),
                0b1000_0000 => Complex::I / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 => Complex::ONE / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_clifford_circuit() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 | 0b1100_0000 | 0b0011_0000 | 0b0111_0000
                | 0b1011_0000 => Complex::I / 8_f64.sqrt(),
                0b1000_0000 | 0b1111_0000 => -Complex::I / 8_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_t_gates() {
        let circuit = CliffordTCircuit::new(8, [T(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_tdg_gates() {
        let circuit = CliffordTCircuit::new(8, [Tdg(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_1() {
        let circuit = CliffordTCircuit::new(8, [T(0), H(1), H(2), Cnot(1, 0), Cnot(2, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000 => {
                    Complex { re: 0.5, im: 0.0 }
                }
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_2() {
        let circuit = CliffordTCircuit::new(8, [H(0), H(1), Cnot(1, 0), T(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 => Complex { re: 0.5, im: 0.0 },
                0b1000_0000 | 0b1100_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_1() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                T(0),
                H(1),
                S(1),
                H(3),
                H(0),
                S(0),
                S(1),
                S(2),
                T(1),
                H(0),
                Cnot(1, 0),
                T(0),
                S(3),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1101_0000 => Complex { re: 0.25, im: 0.25 },
                0b1000_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b0100_0000 => Complex {
                    re: -0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b1100_0000 => Complex {
                    re: 0.25,
                    im: -0.25,
                },
                0b0001_0000 => Complex {
                    re: -0.25,
                    im: 0.25,
                },
                0b1001_0000 => Complex {
                    re: 0.0,
                    im: 0.125_f64.sqrt(),
                },
                0b0101_0000 => Complex {
                    re: 0.0,
                    im: -0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_2() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                Tdg(0),
                H(2),
                H(1),
                Y(2),
                Cnot(1, 0),
                T(0),
                Tdg(0),
                Cnot(2, 1),
                Sdg(1),
                S(2),
                X(0),
                S(0),
                Y(0),
                T(1),
                Tdg(2),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex { re: 0.0, im: -0.5 },
                0b1100_0000 => Complex {
                    re: 2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b1010_0000 => Complex {
                    re: -2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b0110_0000 => Complex { re: 0.0, im: 0.5 },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn linear_cluster_state() {
        let circuit =
            CliffordTCircuit::new(8, (0..8).map(H).chain((0..7).map(|q| Cz(q, q + 1)))).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_gpu(&w, &circuit);

            let adjacent_pairs = (0..7).filter(|&q| w[q] && w[q + 1]).count();
            let expected = match adjacent_pairs % 2 {
                0 => Complex::ONE / 16.0,
                _ => -Complex::ONE / 16.0,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    /// Phase shifts from row multiplication between rows across different bit blocks.
    #[test]
    fn multi_block_phase() {
        let circuit =
            CliffordTCircuit::new(70, (0..70).map(H).chain([Cz(0, 64), Cz(0, 69)])).unwrap();

        let amplitude = 2_f64.powi(-35);
        let cases: [(&[usize], Complex<f64>); 7] = [
            (&[], Complex::ONE * amplitude),
            (&[0], Complex::ONE * amplitude),
            (&[64], Complex::ONE * amplitude),
            (&[0, 64], -Complex::ONE * amplitude), // Edge 0-64
            (&[0, 69], -Complex::ONE * amplitude), // Edge 0-69
            (&[0, 64, 69], Complex::ONE * amplitude), // Edges 0-64 and 0-69
            (&[1, 2, 3], Complex::ONE * amplitude),
        ];
        for (set_bits, expected) in cases {
            let mut w = [false; 70];
            for &b in set_bits {
                w[b] = true;
            }

            let result = simulate_circuit_gpu(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn large_tableau() {
        let circuit =
            CliffordTCircuit::new(300, [H(1), T(1), H(78), T(78), H(123), T(123)]).unwrap();

        let amplitude = 2_f64.powf(-1.5);
        let cases: [(&[usize], Complex<f64>); 6] = [
            (&[], Complex::from_polar(amplitude, 0.0)),
            (&[1], Complex::from_polar(amplitude, PI / 4.0)),
            (&[1, 78], Complex::from_polar(amplitude, 2.0 * PI / 4.0)),
            (
                &[1, 78, 123],
                Complex::from_polar(amplitude, 3.0 * PI / 4.0),
            ),
            (&[42], Complex::ZERO),
            (&[1, 42], Complex::ZERO),
        ];
        for (set_bits, expected) in cases {
            let mut w = [false; 300];
            for &b in set_bits {
                w[b] = true;
            }

            let result = simulate_circuit_gpu(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }
}

#[cfg(feature = "gpu")]
mod hybrid {
    use qlit::simulate_circuit_hybrid;

    use super::*;

    #[test]
    #[should_panic]
    fn mismatched_qubit_number() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();
        let w = [false; 9];
        simulate_circuit_hybrid(&w, &circuit);
    }

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn x_gate() {
        let circuit = CliffordTCircuit::new(8, [X(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn y_gate() {
        let circuit = CliffordTCircuit::new(8, [Y(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::I,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE / 2_f64.sqrt(),
                0b1000_0000 => Complex::I / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 => Complex::ONE / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_clifford_circuit() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                H(0),
                H(1),
                S(2),
                H(3),
                S(1),
                S(0),
                Cnot(2, 3),
                S(1),
                H(0),
                S(3),
                Cnot(1, 0),
                S(3),
                H(1),
                S(3),
                S(1),
                S(3),
                H(1),
                Cnot(3, 2),
                H(1),
                Cnot(3, 1),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 | 0b1100_0000 | 0b0011_0000 | 0b0111_0000
                | 0b1011_0000 => Complex::I / 8_f64.sqrt(),
                0b1000_0000 | 0b1111_0000 => -Complex::I / 8_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_t_gates() {
        let circuit = CliffordTCircuit::new(8, [T(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn only_tdg_gates() {
        let circuit = CliffordTCircuit::new(8, [Tdg(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_1() {
        let circuit = CliffordTCircuit::new(8, [T(0), H(1), H(2), Cnot(1, 0), Cnot(2, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 | 0b1010_0000 | 0b0110_0000 => {
                    Complex { re: 0.5, im: 0.0 }
                }
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn smaller_circuit_2() {
        let circuit = CliffordTCircuit::new(8, [H(0), H(1), Cnot(1, 0), T(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 => Complex { re: 0.5, im: 0.0 },
                0b1000_0000 | 0b1100_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_1() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                T(0),
                H(1),
                S(1),
                H(3),
                H(0),
                S(0),
                S(1),
                S(2),
                T(1),
                H(0),
                Cnot(1, 0),
                T(0),
                S(3),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1101_0000 => Complex { re: 0.25, im: 0.25 },
                0b1000_0000 => Complex {
                    re: 0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b0100_0000 => Complex {
                    re: -0.125_f64.sqrt(),
                    im: 0.0,
                },
                0b1100_0000 => Complex {
                    re: 0.25,
                    im: -0.25,
                },
                0b0001_0000 => Complex {
                    re: -0.25,
                    im: 0.25,
                },
                0b1001_0000 => Complex {
                    re: 0.0,
                    im: 0.125_f64.sqrt(),
                },
                0b0101_0000 => Complex {
                    re: 0.0,
                    im: -0.125_f64.sqrt(),
                },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn larger_circuit_2() {
        let circuit = CliffordTCircuit::new(
            8,
            [
                Tdg(0),
                H(2),
                H(1),
                Y(2),
                Cnot(1, 0),
                T(0),
                Tdg(0),
                Cnot(2, 1),
                Sdg(1),
                S(2),
                X(0),
                S(0),
                Y(0),
                T(1),
                Tdg(2),
            ],
        )
        .unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex { re: 0.0, im: -0.5 },
                0b1100_0000 => Complex {
                    re: 2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b1010_0000 => Complex {
                    re: -2_f64.sqrt() / 4.0,
                    im: -2_f64.sqrt() / 4.0,
                },
                0b0110_0000 => Complex { re: 0.0, im: 0.5 },
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn linear_cluster_state() {
        let circuit =
            CliffordTCircuit::new(8, (0..8).map(H).chain((0..7).map(|q| Cz(q, q + 1)))).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit_hybrid(&w, &circuit);

            let adjacent_pairs = (0..7).filter(|&q| w[q] && w[q + 1]).count();
            let expected = match adjacent_pairs % 2 {
                0 => Complex::ONE / 16.0,
                _ => -Complex::ONE / 16.0,
            };
            assert_almost_eq(result, expected, &w);
        }
    }

    /// Phase shifts from row multiplication between rows across different bit blocks.
    #[test]
    fn multi_block_phase() {
        let circuit =
            CliffordTCircuit::new(70, (0..70).map(H).chain([Cz(0, 64), Cz(0, 69)])).unwrap();

        let amplitude = 2_f64.powi(-35);
        let cases: [(&[usize], Complex<f64>); 7] = [
            (&[], Complex::ONE * amplitude),
            (&[0], Complex::ONE * amplitude),
            (&[64], Complex::ONE * amplitude),
            (&[0, 64], -Complex::ONE * amplitude), // Edge 0-64
            (&[0, 69], -Complex::ONE * amplitude), // Edge 0-69
            (&[0, 64, 69], Complex::ONE * amplitude), // Edges 0-64 and 0-69
            (&[1, 2, 3], Complex::ONE * amplitude),
        ];
        for (set_bits, expected) in cases {
            let mut w = [false; 70];
            for &b in set_bits {
                w[b] = true;
            }

            let result = simulate_circuit_hybrid(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }

    #[test]
    fn large_tableau() {
        let circuit =
            CliffordTCircuit::new(300, [H(1), T(1), H(78), T(78), H(123), T(123)]).unwrap();

        let amplitude = 2_f64.powf(-1.5);
        let cases: [(&[usize], Complex<f64>); 6] = [
            (&[], Complex::from_polar(amplitude, 0.0)),
            (&[1], Complex::from_polar(amplitude, PI / 4.0)),
            (&[1, 78], Complex::from_polar(amplitude, 2.0 * PI / 4.0)),
            (
                &[1, 78, 123],
                Complex::from_polar(amplitude, 3.0 * PI / 4.0),
            ),
            (&[42], Complex::ZERO),
            (&[1, 42], Complex::ZERO),
        ];
        for (set_bits, expected) in cases {
            let mut w = [false; 300];
            for &b in set_bits {
                w[b] = true;
            }

            let result = simulate_circuit_hybrid(&w, &circuit);

            assert_almost_eq(result, expected, &w);
        }
    }
}

/// Relative tolerance of [`DELTA`], with an absolute floor for expected values near zero.
fn assert_almost_eq(result: Complex<f64>, expected: Complex<f64>, w: &[bool]) {
    let tolerance = (DELTA * expected.norm()).max(ABS_DELTA);
    let diff = (result - expected).norm();
    assert!(
        diff < tolerance,
        "w={}\nresult={result:?}\nexpected={expected:?}\ndifference={diff}",
        w.iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect::<String>(),
    );
}

/// Convert the 8 bits to a vector of 8 booleans.
///
/// # Example
/// ```ignore
/// bits_to_bools(0b1001_0110) -> [true, false, false, true, false, true, true, false]
/// ```
pub fn bits_to_bools(bits: u8) -> Vec<bool> {
    (0..8).map(|b| bits & (0b1000_0000 >> b) != 0).collect()
}
