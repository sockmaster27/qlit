use num_complex::Complex;
use qlit::{CircuitCreationError, CliffordTCircuit, CliffordTGate::*};

#[test]
fn invalid_qubit_index() {
    assert_eq!(
        CliffordTCircuit::new(8, [H(165)]),
        Err(CircuitCreationError::InvalidQubitIndex {
            index: 165,
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
        CliffordTCircuit::new(8, [T(8)]),
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
}

mod cpu {
    use qlit::simulate_circuit;

    use super::*;

    #[test]
    #[should_panic]
    fn mismatched_qubit_number() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();
        let w = [false; 9];
        simulate_circuit(&w, &circuit);
    }

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE / 2_f64.sqrt(),
                0b1000_0000 => Complex::I / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b1000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b1100_0000 => Complex::ONE / 2_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
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

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 | 0b0100_0000 | 0b1100_0000 | 0b0011_0000 | 0b0111_0000
                | 0b1011_0000 => Complex::I / 8_f64.sqrt(),
                0b1000_0000 | 0b1111_0000 => -Complex::I / 8_f64.sqrt(),
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
        }
    }

    #[test]
    fn only_t_gates() {
        let circuit = CliffordTCircuit::new(8, [T(0); 5]).unwrap();

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);

            let result = simulate_circuit(&w, &circuit);

            let expected = match i {
                0b0000_0000 => Complex::ONE,
                _ => Complex::ZERO,
            };
            assert_almost_eq(result, expected, i);
        }
    }

    #[test]
    fn larger_circuit() {
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
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
            assert_almost_eq(result, expected, i);
        }
    }
}

fn assert_almost_eq(result: Complex<f64>, expected: Complex<f64>, i: u8) {
    let diff = (result - expected).norm();
    assert!(
        diff < 1e-6,
        "w={i:008b}\nresult={result:?}\nexpected={expected:?}\ndifference={diff}",
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
