use num_complex::Complex;
use pyo3::{ffi::c_str, Python};
use qlit::{
    clifford_circuit::{CircuitCreationError, CliffordTCircuit, CliffordTGate::*},
    python_module, simulate_circuit,
};

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

        let expected = if i == 0b0000_0000 {
            Complex::ONE
        } else {
            Complex::ZERO
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

        let expected = if i == 0b0000_0000 {
            Complex::ONE / 2_f64.sqrt()
        } else if i == 0b1000_0000 {
            Complex::I / 2_f64.sqrt()
        } else {
            Complex::ZERO
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

        let expected = if i == 0b1000_0000 {
            Complex::ONE
        } else {
            Complex::ZERO
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

        let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
            Complex::ONE / 2_f64.sqrt()
        } else {
            Complex::ZERO
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
            Complex::I / 8_f64.sqrt()
        } else if [0b1000_0000, 0b1111_0000].contains(&i) {
            -Complex::I / 8_f64.sqrt()
        } else {
            Complex::ZERO
        };
        assert_almost_eq(result, expected, i);
    }
}

fn assert_almost_eq(result: Complex<f64>, expected: Complex<f64>, i: u8) {
    assert!(
        (result - expected).norm() < 1e-10,
        "w={i:008b}\nresult={result:?}\nexpected={expected:?}",
    );
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
pub fn bits_to_bools(bits: u8) -> Vec<bool> {
    (0..8).map(|b| bits & (0b1000_0000 >> b) != 0).collect()
}
