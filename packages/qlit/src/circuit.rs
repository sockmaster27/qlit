use std::{error::Error, fmt::Display, iter::Copied, slice, vec};

use rand::{RngExt, SeedableRng, rngs::Xoshiro128PlusPlus, seq::IndexedRandom};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    X(u32),
    Y(u32),
    Z(u32),

    S(u32),
    /// The inverse of the S gate.
    Sdg(u32),

    H(u32),

    Cnot(u32, u32),
    Cz(u32, u32),

    T(u32),
    /// The inverse of the T gate.
    Tdg(u32),

    Rx(f64, u32),
    Ry(f64, u32),
    Rz(f64, u32),
}
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    /// The number of qubits in the circuit.
    qubits: u32,
    /// The ordered list of gates in the circuit.
    gates: Vec<Gate>,
}
impl Circuit {
    pub fn new(
        qubits: u32,
        gates: impl IntoIterator<Item = Gate>,
    ) -> Result<Self, CircuitCreationError> {
        let gates: Vec<Gate> = gates.into_iter().collect();
        let check_index = |a| {
            if a >= qubits {
                return Err(CircuitCreationError::InvalidQubitIndex { index: a, qubits });
            }
            Ok(())
        };
        for &gate in &gates {
            use Gate::*;
            match gate {
                X(a) => check_index(a)?,
                Y(a) => check_index(a)?,
                Z(a) => check_index(a)?,
                H(a) => check_index(a)?,
                S(a) => check_index(a)?,
                Sdg(a) => check_index(a)?,
                Cnot(a, b) => {
                    check_index(a)?;
                    check_index(b)?;
                }
                Cz(a, b) => {
                    check_index(a)?;
                    check_index(b)?;
                }
                T(a) => {
                    check_index(a)?;
                }
                Tdg(a) => {
                    check_index(a)?;
                }
                Rx(_, a) => check_index(a)?,
                Ry(_, a) => check_index(a)?,
                Rz(_, a) => check_index(a)?,
            }
        }
        Ok(Self { qubits, gates })
    }

    /// Create a random circuit from the Clifford+T fragment, with the given number of `qubits` and `gates` of which `t_gates` are T gates.
    pub fn random_clifford_t(qubits: u32, gates: usize, t_gates: usize, seed: u64) -> Self {
        assert!(1 < qubits, "qubits must be greater than 1");
        assert!(
            t_gates <= gates,
            "t_gates must be less than or equal to gates"
        );

        // Xoshiro128++ (unlike SmallRng) is portable across platforms, so a given
        // seed always produces the same circuit regardless of pointer width.
        // Useful for CI testing and benchmarking.
        let mut rng = Xoshiro128PlusPlus::seed_from_u64(seed);

        let mut t_gate_positions = Vec::new();
        for _ in 0..t_gates {
            let mut pos;
            loop {
                pos = rng.random_range(0..gates);
                if !t_gate_positions.contains(&pos) {
                    t_gate_positions.push(pos);
                    break;
                }
            }
        }

        Self {
            qubits,
            gates: (0..gates)
                .map(|i| {
                    use Gate::*;
                    let a = rng.random_range(0..qubits);
                    let mut b = a;
                    while b == a {
                        b = rng.random_range(0..qubits);
                    }
                    if t_gate_positions.contains(&i) {
                        [T(a), Tdg(a)].choose(&mut rng).unwrap().clone()
                    } else {
                        [X(a), Y(a), Z(a), H(a), S(a), Sdg(a), Cnot(a, b), Cz(a, b)]
                            .choose(&mut rng)
                            .unwrap()
                            .clone()
                    }
                })
                .collect(),
        }
    }

    /// The number of qubits in the circuit.
    pub fn qubits(&self) -> u32 {
        self.qubits
    }

    /// The number of gates in the circuit.
    pub fn gates(&self) -> usize {
        self.gates.len()
    }

    /// An iterator over the gates in the circuit, in the order that they are applied.
    pub fn iter(&self) -> impl Iterator<Item = Gate> {
        self.into_iter()
    }
}
pub struct Iter<'a> {
    inner: Copied<slice::Iter<'a, Gate>>,
}
impl<'a> Iterator for Iter<'a> {
    type Item = Gate;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, f)
    }
    fn nth(&mut self, n: usize) -> Option<Gate> {
        self.inner.nth(n)
    }
    fn last(self) -> Option<Gate> {
        self.inner.last()
    }
    fn count(self) -> usize {
        self.inner.count()
    }
}
impl<'a> IntoIterator for &'a Circuit {
    type Item = Gate;

    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: self.gates.iter().copied(),
        }
    }
}
pub struct IntoIter {
    inner: vec::IntoIter<Gate>,
}
impl Iterator for IntoIter {
    type Item = Gate;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, f)
    }
    fn nth(&mut self, n: usize) -> Option<Gate> {
        self.inner.nth(n)
    }
    fn last(self) -> Option<Gate> {
        self.inner.last()
    }
    fn count(self) -> usize {
        self.inner.count()
    }
}
impl IntoIterator for Circuit {
    type Item = Gate;

    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.gates.into_iter(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitCreationError {
    InvalidQubitIndex { index: u32, qubits: u32 },
}
impl Display for CircuitCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitCreationError::InvalidQubitIndex { index, qubits } => {
                write!(
                    f,
                    "Invalid qubit index {index} for circuit of {qubits} qubits"
                )
            }
        }
    }
}
impl Error for CircuitCreationError {}
