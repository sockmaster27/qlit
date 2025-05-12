use std::borrow::Borrow;
use std::fmt::Debug;
use std::mem;

use num_complex::Complex;
use num_traits::{One, Zero};
use pyo3::prelude::*;

use crate::clifford_circuit::CliffordGate;

type BitBlock = u64;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

#[derive(Debug)]
enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[pyclass]
pub struct GeneratorRow {
    n: usize,
    /// The augmented stabilizer tableau,
    /// ```text
    /// P1 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// P2 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// ...
    /// Pn -> x1 x2 ... xn | z1 z2 ... zn | r
    /// ```
    /// is layed out row-wise in the following way:
    /// ```text
    /// P1 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// P2 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// ...
    /// Pn -> x1 x2 ... xn | z1 z2 ... zn | r
    /// (E -> x1 x2 ... xn | z1 z2 ... zn | r)
    /// ```
    /// Note that the x, z and r parts of each row is right-padded with zeros, to take up a whole number of blocks.
    tableau: Vec<BitBlock>,
}
impl GeneratorRow {
    /// Initialize a new generator with `n` qubits in the initial zero state.
    pub fn zero(n: usize) -> Self {
        let mut tableau = vec![0; tableau_block_length(n)];
        for i in 0..n {
            let block_index = z_row_block_index(n, i, i / BLOCK_SIZE);
            tableau[block_index] = bitmask(i % BLOCK_SIZE);
        }
        GeneratorRow { tableau, n }
    }

    /// Apply a Clifford gate to the generator.
    pub fn apply_gate(&mut self, gate: CliffordGate) {
        let n = self.n;
        match gate {
            CliffordGate::S(a) => {
                for i in 0..n {
                    let r_block = r_row_block_index(n, i);
                    let x_block = x_row_block_index(n, i, a / BLOCK_SIZE);
                    let z_block = z_row_block_index(n, i, a / BLOCK_SIZE);
                    let a_bit = a % BLOCK_SIZE;
                    let a_bitmask = bitmask(a_bit);

                    self.tableau[r_block] ^= align_bit_to(
                        self.tableau[x_block] & self.tableau[z_block] & a_bitmask,
                        a_bit,
                        0,
                    );
                    self.tableau[z_block] ^= self.tableau[x_block] & a_bitmask;
                }
            }
            CliffordGate::H(a) => {
                for i in 0..n {
                    let r_block = r_row_block_index(n, i);
                    let x_block = x_row_block_index(n, i, a / BLOCK_SIZE);
                    let z_block = z_row_block_index(n, i, a / BLOCK_SIZE);
                    let a_bit = a % BLOCK_SIZE;
                    let a_bitmask = bitmask(a_bit);

                    self.tableau[r_block] ^= align_bit_to(
                        self.tableau[x_block] & self.tableau[z_block] & a_bitmask,
                        a_bit,
                        0,
                    );
                    // Swap x and z.
                    self.tableau[z_block] ^= self.tableau[x_block] & a_bitmask;
                    self.tableau[x_block] ^= self.tableau[z_block] & a_bitmask;
                    self.tableau[z_block] ^= self.tableau[x_block] & a_bitmask;
                }
            }
            CliffordGate::Cnot(a, b) => {
                for i in 0..n {
                    let r_block = r_row_block_index(n, i);
                    let xa_block = x_row_block_index(n, i, a / BLOCK_SIZE);
                    let za_block = z_row_block_index(n, i, a / BLOCK_SIZE);
                    let xb_block = x_row_block_index(n, i, b / BLOCK_SIZE);
                    let zb_block = z_row_block_index(n, i, b / BLOCK_SIZE);
                    let a_bit = a % BLOCK_SIZE;
                    let b_bit = b % BLOCK_SIZE;
                    let a_bitmask = bitmask(a_bit);
                    let b_bitmask = bitmask(b_bit);

                    self.tableau[r_block] ^=
                        align_bit_to(self.tableau[xa_block] & a_bitmask, a_bit, 0)
                            & align_bit_to(self.tableau[zb_block] & b_bitmask, b_bit, 0)
                            & !(align_bit_to(self.tableau[xb_block] & b_bitmask, b_bit, 0)
                                ^ align_bit_to(self.tableau[za_block] & a_bitmask, a_bit, 0));
                    self.tableau[za_block] ^=
                        align_bit_to(self.tableau[zb_block] & b_bitmask, b_bit, a_bit);
                    self.tableau[xb_block] ^=
                        align_bit_to(self.tableau[xa_block] & a_bitmask, a_bit, b_bit);
                }
            }
        }
    }

    /// Compute the ratio of the coefficients of `w1` and `w2`, such that
    /// ```text
    /// coefficient_ratio(w1, w2) * coeff(w1) = coeff(w2)
    /// ```
    pub fn coeff_ratio(&mut self, w1: &[bool], w2: &[bool]) -> Complex<i8> {
        let n = self.n;
        assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");
        assert_eq!(w2.len(), n, "Basis state 2 must have length {n}");

        let aux_row = n;

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Reset the auxiliary row.
        for i in 0..row_block_length(n) {
            self.tableau[row_block_index(n, aux_row, i)] = 0;
        }
        // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
        let mut row = 0;
        for j in 0..n {
            if self.x_bit(row, j) == true {
                if w1[j] != w2[j] {
                    self.multiply_rows_into(row, aux_row);
                }
                row += 1;
            }
        }

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        self.stabilizer_matrix_entry(aux_row, w1, w2)
    }

    /// Same as [`coeff_ratio`], but for the special case where `w2` is equal to `w1` except for a single flipped bit.
    pub fn coeff_ratio_flipped_bit(&mut self, w1: &[bool], flipped_bit: usize) -> Complex<i8> {
        let n = self.n;
        assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Identify the row with a set bit in the given position.
        let mut row = 0;
        for j in 0..n {
            if self.x_bit(j, flipped_bit) == true {
                row = j;
                break;
            }
        }

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        self.stabilizer_matrix_entry(
            row,
            w1,
            w1.iter()
                .enumerate()
                .map(|(i, &b)| if i == flipped_bit { !b } else { b }),
        )
    }

    /// Bring tableau's x part into reduced row echelon form.
    ///
    /// This should take O(n^2) time, plus an additional O(n^2) time for each gate that has been applied since the last call to this function.
    fn bring_into_rref(&mut self) {
        let n = self.n;
        let mut a = 0;
        for col in 0..n {
            let mut pivot = None;
            for row in (a..n).rev() {
                if self.x_bit(row, col) == true {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(pivot) = pivot {
                for row in 0..pivot {
                    if self.x_bit(row, col) == true {
                        self.multiply_rows_into(pivot, row);
                    }
                }
                self.swap_rows(a, pivot);
                a += 1;
            }
        }
    }

    /// Compute the entry of the `row`th stabilizer matrix, `P[w2, w1]`, for the given basis state pair.
    fn stabilizer_matrix_entry<W1, W2>(&self, row: usize, w1: W1, w2: W2) -> Complex<i8>
    where
        W1: IntoIterator<Item: Borrow<bool>>,
        W2: IntoIterator<Item: Borrow<bool>>,
    {
        let n = self.n;
        let mut w1 = w1.into_iter();
        let mut w2 = w2.into_iter();

        let mut res = if self.row_negative(row) {
            -Complex::one()
        } else {
            Complex::one()
        };
        for q in 0..n {
            // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
            res *= match (
                self.tensor_element(row, q),
                w1.next().unwrap().borrow(),
                w2.next().unwrap().borrow(),
            ) {
                (Pauli::I, false, false) => Complex::one(),
                (Pauli::I, true, true) => Complex::one(),

                (Pauli::X, false, true) => Complex::one(),
                (Pauli::X, true, false) => Complex::one(),

                (Pauli::Y, false, true) => Complex::i(),
                (Pauli::Y, true, false) => -Complex::i(),

                (Pauli::Z, false, false) => Complex::one(),
                (Pauli::Z, true, true) => -Complex::one(),

                _ => return Complex::zero(),
            };
        }
        res
    }

    /// Set row with index `target` to be the product of the `source` and `target` rows.
    ///
    /// NOTE: Since all stabilizers must commute, multiplication order is irrelevant.
    fn multiply_rows_into(&mut self, source: usize, target: usize) {
        let n = self.n;

        // Compute the sign bit.
        let mut phase: i8 = 0;
        for q in 0..n {
            match (
                self.tensor_element(source, q),
                self.tensor_element(target, q),
            ) {
                (Pauli::X, Pauli::Y) => phase += 1,
                (Pauli::X, Pauli::Z) => phase -= 1,

                (Pauli::Y, Pauli::Z) => phase += 1,
                (Pauli::Y, Pauli::X) => phase -= 1,

                (Pauli::Z, Pauli::X) => phase += 1,
                (Pauli::Z, Pauli::Y) => phase -= 1,

                _ => {}
            }
            phase = phase.rem_euclid(4);
        }
        match phase {
            0 => {
                // Do nothing.
            }
            2 => {
                // Negate the sign bit.
                self.tableau[r_row_block_index(n, target)] ^= bitmask(0);
            }
            _ => unreachable!("No valid stabilizer can have imaginary phase: {phase}"),
        };

        // XOR the whole thing.
        for i in 0..row_block_length(n) {
            self.tableau[row_block_index(n, target, i)] ^=
                self.tableau[row_block_index(n, source, i)];
        }
    }

    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        let n = self.n;

        for i in 0..row_block_length(n) {
            self.tableau
                .swap(row_block_index(n, row1, i), row_block_index(n, row2, i));
        }
    }

    /// Get whether the given row is negative or not, i.e. the contents of the sign bit.
    fn row_negative(&self, row: usize) -> bool {
        let n = self.n;
        self.tableau[r_row_block_index(n, row)] != 0
    }

    /// Get the Pauli matrix corresponding to the `q`th tensor element in the `p`th row,
    fn tensor_element(&self, row: usize, q: usize) -> Pauli {
        let n = self.n;
        let q_block_index = q / BLOCK_SIZE;
        let q_bit_index = q % BLOCK_SIZE;
        let q_bitmask = bitmask(q_bit_index);

        let x = self.tableau[x_row_block_index(n, row, q_block_index)] & q_bitmask != 0;
        let z = self.tableau[z_row_block_index(n, row, q_block_index)] & q_bitmask != 0;

        match (x, z) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (true, true) => Pauli::Y,
            (false, true) => Pauli::Z,
        }
    }

    /// Get the value of the x bit corresponding to the `q`th tensor element in the `row`th row.
    fn x_bit(&self, row: usize, q: usize) -> bool {
        let n = self.n;
        let block_index = q / BLOCK_SIZE;
        let bit_index = q % BLOCK_SIZE;
        self.tableau[x_row_block_index(n, row, block_index)] & bitmask(bit_index) != 0
    }
}

/// Get the bitmask for the i'th bit, e.g.
///
/// ```text
/// bitmask(0) -> 10000000
/// bitmask(1) -> 01000000
/// bitmask(6) -> 00000010
/// ```
///
/// # Panics
/// If `i` is greater than or equal to `BLOCK_SIZE` in debug mode.
fn bitmask(i: usize) -> BitBlock {
    debug_assert!(i < BLOCK_SIZE);
    1 << (BLOCK_SIZE - 1 - i)
}

/// Get the index of the i'th block of the `row`th row.
fn row_block_index(n: usize, row: usize, i: usize) -> usize {
    debug_assert!(row < n + 1);
    debug_assert!(i < row_block_length(n));
    row * row_block_length(n) + i
}
/// Get the index of the i'th block of the x part of the `row`th row.
fn x_row_block_index(n: usize, row: usize, i: usize) -> usize {
    debug_assert!(row < n + 1);
    debug_assert!(i < n.div_ceil(BLOCK_SIZE));
    row_block_index(n, row, i)
}
/// Get the index of the i'th block of the z part of the `row`th row.
fn z_row_block_index(n: usize, row: usize, i: usize) -> usize {
    debug_assert!(row < n + 1);
    debug_assert!(i < n.div_ceil(BLOCK_SIZE));
    row_block_index(n, row, n.div_ceil(BLOCK_SIZE) + i)
}
/// Get the index of the block containing the sign bit of the `row`th row.
fn r_row_block_index(n: usize, row: usize) -> usize {
    debug_assert!(row < n + 1);
    row_block_index(n, row, 2 * n.div_ceil(BLOCK_SIZE))
}

/// Get the block-length of the rows in the tableau.
fn row_block_length(n: usize) -> usize {
    2 * n.div_ceil(BLOCK_SIZE) + BLOCK_SIZE
}
/// Get the block-length of the tableau.
fn tableau_block_length(n: usize) -> usize {
    // Make room for the auxiliary row.
    row_block_length(n) * (n + 1)
}

/// Bit-shift the given block such that the `from`th bit is moved to the `to`th position.
pub fn align_bit_to(block: BitBlock, from: usize, to: usize) -> BitBlock {
    if to < from {
        block << (from - to)
    } else {
        block >> (to - from)
    }
}
