use num_complex::Complex;
use num_traits::{One, Zero};
use pyo3::{ffi::c_str, Python};
use qlit::coeff_ratio;
use qlit::{
    clifford_circuit::{CliffordCircuit, CliffordGate::*},
    python_module, simulate_clifford_circuit_gpu, BasisStateProbability,
};

mod cpu {
    use super::*;

    #[test]
    fn zero() {
        let circuit = CliffordCircuit::new(8, []);

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if i == 0b0000_0000 {
                Complex::one()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordCircuit::new(8, [H(0), S(0)]);

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if i == 0b0000_0000 {
                Complex::one()
            } else if i == 0b1000_0000 {
                Complex::i()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn negative_imaginary() {
        let circuit = CliffordCircuit::new(8, [H(0), S(0)]);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if i == 0b0000_0000 {
                -Complex::i()
            } else if i == 0b1000_0000 {
                Complex::one()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordCircuit::new(8, [H(0), S(0), S(0), H(0)]);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if i == 0b1000_0000 {
                Complex::one()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordCircuit::new(8, [H(0), Cnot(0, 1)]);

        let w1 = bits_to_bools(0b1100_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                Complex::one()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn larger_circuit() {
        let circuit = CliffordCircuit::new(
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
        );

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);
            let result = coeff_ratio(&w1, &w2, &circuit);
            let expected = if [
                0b0000_0000,
                0b0100_0000,
                0b1100_0000,
                0b0011_0000,
                0b0111_0000,
                0b1011_0000,
            ]
            .contains(&i)
            {
                -Complex::one()
            } else if [0b1000_0000, 0b1111_0000].contains(&i) {
                Complex::one()
            } else {
                Complex::zero()
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }
}

mod gpu {
    use super::*;

    #[test]
    fn zero() {
        let circuit = CliffordCircuit::new(8, []);

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);
            let result = simulate_clifford_circuit_gpu(&w, &circuit);
            let expected = if i == 0b0000_0000 {
                BasisStateProbability::One
            } else {
                BasisStateProbability::Zero
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordCircuit::new(8, [H(0), S(0), S(0), H(0)]);

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);
            let result = simulate_clifford_circuit_gpu(&w, &circuit);
            let expected = if i == 0b1000_0000 {
                BasisStateProbability::One
            } else {
                BasisStateProbability::Zero
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordCircuit::new(8, [H(0), Cnot(0, 1)]);

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);
            let result = simulate_clifford_circuit_gpu(&w, &circuit);
            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                BasisStateProbability::InBetween
            } else {
                BasisStateProbability::Zero
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn larger_circuit() {
        let circuit = CliffordCircuit::new(
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
        );

        for i in 0b0000_0000..=0b1111_1111 {
            let w = bits_to_bools(i);
            let result = simulate_clifford_circuit_gpu(&w, &circuit);
            let expected = if [
                0b0000_0000,
                0b0100_0000,
                0b1000_0000,
                0b1100_0000,
                0b0011_0000,
                0b0111_0000,
                0b1011_0000,
                0b1111_0000,
            ]
            .contains(&i)
            {
                BasisStateProbability::InBetween
            } else {
                BasisStateProbability::Zero
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }
}

#[test]
fn python() {
    pyo3::append_to_inittab!(python_module);
    Python::with_gil(|py| Python::run(py, c_str!(include_str!("test.py")), None, None).unwrap());
}

/// Convert the 8 bits to a vector of 8 booleans.
///
/// # Example
/// ```ignore
/// bits_to_bools(0b1001_0110) -> [true, false, false, true, false, true, true, false]
/// ```
fn bits_to_bools(bits: u8) -> Vec<bool> {
    (0..8).map(|b| bits & (0b1000_0000 >> b) != 0).collect()
}
