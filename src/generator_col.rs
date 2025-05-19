use std::borrow::Borrow;
use std::fmt::Debug;
use std::mem;

use num_complex::Complex;

type BitBlock = u64;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

#[derive(Debug)]
enum Pauli {
    I,
    X,
    Y,
    Z,
}

pub struct GeneratorCol {
    n: usize,
    /// The augmented stabilizer tableau,
    /// ```text
    /// P1 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// P2 -> x1 x2 ... xn | z1 z2 ... zn | r
    /// ...
    /// Pn -> x1 x2 ... xn | z1 z2 ... zn | r
    /// ```
    /// is layed out column-wise in the following way:
    /// ```text
    /// P1 -> x1 z1 x2 z2 ... xn zn | r
    /// P2 -> x1 z1 x2 z2 ... xn zn | r
    /// ...
    /// Pn -> x1 z1 x2 z2 ... xn zn | r
    /// (E -> x1 z1 x2 z2 ... xn zn | r)
    /// ```
    /// Note that the x and z columns are interleaved, and that an auxiliary row, E, is added at the end.
    tableau: Vec<BitBlock>,
}
impl GeneratorCol {
    /// Initialize a new generator with `n` qubits in the initial zero state.
    pub fn zero(n: usize) -> Self {
        let mut tableau = vec![0; tableau_block_length(n)];
        for i in 0..n {
            let block_index = z_column_block_index(n, i / BLOCK_SIZE, i);
            tableau[block_index] = bitmask(i % BLOCK_SIZE);
        }
        GeneratorCol { tableau, n }
    }

    pub fn apply_s_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let r = r_column_block_index(n, i);
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            self.tableau[z] ^= self.tableau[x];
        }
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let r = r_column_block_index(n, i);
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            self.tableau.swap(z, x);
        }
    }
    pub fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let r = r_column_block_index(n, i);
            let xa = x_column_block_index(n, i, a);
            let za = z_column_block_index(n, i, a);
            let xb = x_column_block_index(n, i, b);
            let zb = z_column_block_index(n, i, b);
            self.tableau[r] ^=
                self.tableau[xa] & self.tableau[zb] & !(self.tableau[xb] ^ self.tableau[za]);
            self.tableau[za] ^= self.tableau[zb];
            self.tableau[xb] ^= self.tableau[xa];
        }
    }
    pub fn apply_z_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let r = r_column_block_index(n, i);
            let x = x_column_block_index(n, i, a);
            self.tableau[r] ^= self.tableau[x];
        }
    }

    /// Compute the ratio of the coefficients of `w1` and `w2`, such that
    /// ```text
    /// coefficient_ratio(w1, w2) * coeff(w1) = coeff(w2)
    /// ```
    pub fn coeff_ratio(&mut self, w1: &[bool], w2: &[bool]) -> Complex<f64> {
        let n = self.n;
        debug_assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");
        debug_assert_eq!(w2.len(), n, "Basis state 2 must have length {n}");

        let aux_row = n;
        let aux_block_index = n / BLOCK_SIZE;
        let aux_bit_index = n % BLOCK_SIZE;
        let aux_bitmask = bitmask(aux_bit_index);

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Reset the auxiliary row.
        for q in 0..n {
            self.tableau[x_column_block_index(n, aux_block_index, q)] &= !aux_bitmask;
            self.tableau[z_column_block_index(n, aux_block_index, q)] &= !aux_bitmask;
        }
        self.tableau[r_column_block_index(n, aux_block_index)] &= !aux_bitmask;
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
    pub fn coeff_ratio_flipped_bit(&mut self, w1: &[bool], flipped_bit: usize) -> Complex<f64> {
        let n = self.n;
        debug_assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");

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
        let aux_row = n;

        let mut a = 0;
        for col in 0..n {
            // Find pivot row.
            let mut pivot = None;
            let a_block_index = a / BLOCK_SIZE;
            for i in (a_block_index..column_block_length(n)).rev() {
                // Mask that is set to all-ones except for the bit corresponding to the auxiliary row.
                let aux_mask = if aux_row / BLOCK_SIZE == i {
                    !bitmask(aux_row % BLOCK_SIZE)
                } else {
                    !0
                };
                let block = self.tableau[x_column_block_index(n, i, col)] & aux_mask;
                if block != 0 {
                    let row = BLOCK_SIZE * i + lsb_index(block);
                    if row >= a {
                        pivot = Some(row);
                    }
                }
            }

            if let Some(pivot) = pivot {
                // Determine phase shift.
                for row in 0..pivot {
                    if self.x_bit(row, col) == true {
                        self.multiply_phase_shift(pivot, row);
                    }
                }

                // XOR
                for row_i in 0..column_block_length(n) {
                    if self.tableau[x_column_block_index(n, row_i, col)] == 0 {
                        continue;
                    }

                    // Mask that is set to all-ones except for the bit corresponding to the pivot row.
                    let pivot_mask = if pivot / BLOCK_SIZE == row_i {
                        !bitmask(pivot % BLOCK_SIZE)
                    } else {
                        !0
                    };

                    if self.r_bit(pivot) == true {
                        self.tableau[r_column_block_index(n, row_i)] ^=
                            self.tableau[x_column_block_index(n, row_i, col)] & pivot_mask;
                    }
                    for col2 in (0..n).rev() {
                        if self.z_bit(pivot, col2) == true {
                            self.tableau[z_column_block_index(n, row_i, col2)] ^=
                                self.tableau[x_column_block_index(n, row_i, col)] & pivot_mask;
                        }
                    }
                    for col2 in (0..n).rev() {
                        if self.x_bit(pivot, col2) == true {
                            self.tableau[x_column_block_index(n, row_i, col2)] ^=
                                self.tableau[x_column_block_index(n, row_i, col)] & pivot_mask;
                        }
                    }
                }

                // Swap rows.
                self.swap_rows(a, pivot);
                a += 1;
            }
        }
    }

    /// Compute the entry of the `row`th stabilizer matrix, `P[w2, w1]`, for the given basis state pair.
    fn stabilizer_matrix_entry<W1, W2>(&self, row: usize, w1: W1, w2: W2) -> Complex<f64>
    where
        W1: IntoIterator<Item: Borrow<bool>>,
        W2: IntoIterator<Item: Borrow<bool>>,
    {
        let n = self.n;
        let mut w1 = w1.into_iter();
        let mut w2 = w2.into_iter();

        let mut res = if self.row_negative(row) {
            -Complex::ONE
        } else {
            Complex::ONE
        };
        for q in 0..n {
            // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
            res *= match (
                self.tensor_element(row, q),
                w1.next().unwrap().borrow(),
                w2.next().unwrap().borrow(),
            ) {
                (Pauli::I, false, false) => Complex::ONE,
                (Pauli::I, true, true) => Complex::ONE,

                (Pauli::X, false, true) => Complex::ONE,
                (Pauli::X, true, false) => Complex::ONE,

                (Pauli::Y, false, true) => Complex::I,
                (Pauli::Y, true, false) => -Complex::I,

                (Pauli::Z, false, false) => Complex::ONE,
                (Pauli::Z, true, true) => -Complex::ONE,

                _ => return Complex::ZERO,
            };
        }
        res
    }

    /// Set row with index `target` to be the product of the `source` and `target` rows.
    ///
    /// NOTE: Since all stabilizers must commute, multiplication order is irrelevant.
    fn multiply_rows_into(&mut self, source: usize, target: usize) {
        let n = self.n;

        let source_block_index = source / BLOCK_SIZE;
        let target_block_index = target / BLOCK_SIZE;
        let source_bit_index = source % BLOCK_SIZE;
        let target_bit_index = target % BLOCK_SIZE;
        let source_bitmask = bitmask(source_bit_index);
        let target_bitmask = bitmask(target_bit_index);

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
                self.tableau[r_column_block_index(n, target_block_index)] ^= target_bitmask;
            }
            _ => unreachable!("No valid stabilizer can have imaginary phase: {phase}"),
        };
        self.tableau[r_column_block_index(n, target_block_index)] ^= align_bit_to(
            self.tableau[r_column_block_index(n, source_block_index)] & source_bitmask,
            source_bit_index,
            target_bit_index,
        );

        // XOR the x and z components.
        for q in 0..n {
            let x_source = x_column_block_index(n, source_block_index, q);
            let x_target = x_column_block_index(n, target_block_index, q);
            let z_source = z_column_block_index(n, source_block_index, q);
            let z_target = z_column_block_index(n, target_block_index, q);
            self.tableau[x_target] ^= align_bit_to(
                self.tableau[x_source] & source_bitmask,
                source_bit_index,
                target_bit_index,
            );
            self.tableau[z_target] ^= align_bit_to(
                self.tableau[z_source] & source_bitmask,
                source_bit_index,
                target_bit_index,
            );
        }
    }

    fn multiply_phase_shift(&mut self, source: usize, target: usize) {
        let n = self.n;

        let target_block_index = target / BLOCK_SIZE;
        let target_bit_index = target % BLOCK_SIZE;
        let target_bitmask = bitmask(target_bit_index);

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
                self.tableau[r_column_block_index(n, target_block_index)] ^= target_bitmask;
            }
            _ => unreachable!("No valid stabilizer can have imaginary phase: {phase}"),
        };
    }

    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        self.multiply_rows_into(row1, row2);
        self.multiply_rows_into(row2, row1);
        self.multiply_rows_into(row1, row2);
    }

    /// Get whether the given row is negative or not, i.e. the contents of the sign bit.
    fn row_negative(&self, row: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[r_column_block_index(n, row_block_index)] & row_bitmask != 0
    }

    /// Get the Pauli matrix corresponding to the `q`th tensor element in the `p`th row,
    fn tensor_element(&self, row: usize, q: usize) -> Pauli {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);

        let x = self.tableau[x_column_block_index(n, row_block_index, q)] & row_bitmask != 0;
        let z = self.tableau[z_column_block_index(n, row_block_index, q)] & row_bitmask != 0;

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
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[x_column_block_index(n, row_block_index, q)] & row_bitmask != 0
    }
    /// Get the value of the z bit corresponding to the `q`th tensor element in the `row`th row.
    fn z_bit(&self, row: usize, q: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[z_column_block_index(n, row_block_index, q)] & row_bitmask != 0
    }
    /// Get the value of the r bit of the `row`th row.
    fn r_bit(&self, row: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[r_column_block_index(n, row_block_index)] & row_bitmask != 0
    }
}

/// Get the index of the least significant (right-most) bit in the given block, e.g.
/// ```text
/// lsb_index(10000000)
///           ^0
/// lsb_index(01000000)
///            ^1
/// lsb_index(11010000)
///              ^3
/// ```
///
/// # Panics
/// If `block` is zero in debug mode.
fn lsb_index(block: BitBlock) -> usize {
    debug_assert!(block != 0);
    let trailing_zeros: usize = block.trailing_zeros().try_into().unwrap();
    BLOCK_SIZE - 1 - trailing_zeros
}

/// Get the bitmask for the i'th bit, e.g.
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

/// Set the i'th bit of the given block, i.e. set the bit to 1.
/// ```text
/// set_bit(00000000, 0) -> 10000000
/// set_bit(10001000, 1) -> 11001000
/// set_bit(10011000, 6) -> 10011010
/// ```
///
/// # Panics
/// If `i` is greater than or equal to `BLOCK_SIZE` in debug mode.
fn set_bit(block: &mut BitBlock, i: usize) {
    debug_assert!(i < BLOCK_SIZE);
    *block |= bitmask(i);
}

/// Unset the i'th bit of the given block, i.e. set the bit to 0.
/// ```text
/// set_bit(11111111, 0) -> 01111111
/// set_bit(01110111, 1) -> 00110111
/// set_bit(01100111, 6) -> 01100101
/// ```
///
/// # Panics
/// If `i` is greater than or equal to `BLOCK_SIZE` in debug mode.
fn unset_bit(block: &mut BitBlock, i: usize) {
    debug_assert!(i < BLOCK_SIZE);
    *block &= !bitmask(i);
}

/// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
/// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn x_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    debug_assert!(q < n);
    2 * q * column_block_length(n) + i
}
/// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
/// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn z_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    debug_assert!(q < n);
    (2 * q + 1) * column_block_length(n) + i
}
/// Get the index of the i'th block of the r column.
/// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn r_column_block_index(n: usize, i: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    2 * n * column_block_length(n) + i
}

/// Get the block-length of the columns in the tableau.
fn column_block_length(n: usize) -> usize {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
/// Get the block-length of the tableau.
fn tableau_block_length(n: usize) -> usize {
    column_block_length(n) * (n + n + 1)
}

/// Bit-shift the given block such that the `from`th bit is moved to the `to`th position.
pub fn align_bit_to(block: BitBlock, from: usize, to: usize) -> BitBlock {
    if to < from {
        block << (from - to)
    } else {
        block >> (to - from)
    }
}

impl Debug for GeneratorCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n;
        write!(f, "Generator {{ n: {n:?}\n")?;

        write!(f, "\t      ")?;
        for i in 0..n {
            write!(f, "x{i} ")?;
        }
        write!(f, "| ")?;
        for i in 0..n {
            write!(f, "z{i} ")?;
        }
        write!(f, "| r\n")?;

        for i in 0..n {
            let block_i = i / BLOCK_SIZE;
            write!(f, "\tP{i} -> ")?;
            for j in 0..n {
                let block_index = x_column_block_index(n, block_i, j);
                if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
                    write!(f, " 1 ")?;
                } else {
                    write!(f, " 0 ")?;
                }
            }
            write!(f, "| ")?;
            for j in 0..n {
                let block_index = z_column_block_index(n, block_i, j);
                if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
                    write!(f, " 1 ")?;
                } else {
                    write!(f, " 0 ")?;
                }
            }
            write!(f, "| ")?;
            let block_index = r_column_block_index(n, block_i);
            if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
                write!(f, "1 ")?;
            } else {
                write!(f, "0 ")?;
            }
            write!(f, "\n")?;
        }

        let i = n;
        let block_i = i / BLOCK_SIZE;
        write!(f, "\t(E -> ")?;
        for j in 0..n {
            let block_index = x_column_block_index(n, block_i, j);
            if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
                write!(f, " 1 ")?;
            } else {
                write!(f, " 0 ")?;
            }
        }
        write!(f, "| ")?;
        for j in 0..n {
            let block_index = z_column_block_index(n, block_i, j);
            if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
                write!(f, " 1 ")?;
            } else {
                write!(f, " 0 ")?;
            }
        }
        write!(f, "| ")?;
        let block_index = r_column_block_index(n, block_i);
        if self.tableau[block_index] & bitmask(i % BLOCK_SIZE) != 0 {
            write!(f, "1")?;
        } else {
            write!(f, "0")?;
        }
        write!(f, ")\n")?;

        write!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::clifford_circuit::{CliffordTCircuit, CliffordTGate::*};
    use crate::utils::bits_to_bools;

    use super::*;

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else if i == 0b1000_0000 {
                Complex::I
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn negative_imaginary() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                -Complex::I
            } else if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let circuit = CliffordTCircuit::new(8, [H(0), S(0), S(0), H(0)]).unwrap();

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let circuit = CliffordTCircuit::new(8, [H(0), Cnot(0, 1)]).unwrap();

        let w1 = bits_to_bools(0b1100_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn larger_circuit() {
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

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = GeneratorCol::zero(8);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(&w1, &w2);

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
                -Complex::ONE
            } else if [0b1000_0000, 0b1111_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn bitflip_ratio() {
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

        let w = bits_to_bools(0b1000_0000);
        let mut g = GeneratorCol::zero(8);
        apply_clifford_circuit(&mut g, &circuit);

        assert_eq!(g.coeff_ratio_flipped_bit(&w, 0), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(&w, 1), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(&w, 2), Complex::ZERO);
    }

    #[test]
    fn repeated_reading() {
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

        let mut g = GeneratorCol::zero(8);
        apply_clifford_circuit(&mut g, &circuit);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratio(&w1, &w2);

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
                -Complex::ONE
            } else if [0b1000_0000, 0b1111_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    fn apply_clifford_circuit(g: &mut GeneratorCol, circuit: &CliffordTCircuit) {
        for &gate in circuit.gates() {
            match gate {
                S(a) => g.apply_s_gate(a),
                H(a) => g.apply_h_gate(a),
                Cnot(a, b) => g.apply_cnot_gate(a, b),
                T(_) => unreachable!(),
            }
        }
    }
}
