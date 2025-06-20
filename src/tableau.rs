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

/// An extended stabilizer tableau.
#[derive(Clone)]
pub struct ExtendedTableau {
    /// The number of qubits in the tableau.
    n: usize,
    /// The current number of active r-columns in the tableau.
    r_cols: usize,
    /// The augmented stabilizer tableau,
    /// ```text
    /// P1 -> x1 x2 ... xn | z1 z2 ... zn | r1 r2 ... r2^t
    /// P2 -> x1 x2 ... xn | z1 z2 ... zn | r1 r2 ... r2^t
    /// ...
    /// Pn -> x1 x2 ... xn | z1 z2 ... zn | r1 r2 ... r2^t
    /// ```
    /// is layed out column-wise in the following way:
    /// ```text
    /// P1 -> x1 z1 x2 z2 ... xn zn | r1 r2 ... r2^t
    /// P2 -> x1 z1 x2 z2 ... xn zn | r1 r2 ... r2^t
    /// ...
    /// Pn -> x1 z1 x2 z2 ... xn zn | r1 r2 ... r2^t
    /// (E -> x1 z1 x2 z2 ... xn zn | r1 r2 ... r2^t)
    /// ```
    /// Note that the x and z columns are interleaved, and that an auxiliary row, E, is added at the end.
    tableau: Vec<BitBlock>,
}
impl ExtendedTableau {
    /// Initialize a new tableau with `n` qubits in the initial zero state.
    /// This allocates a tableau with capacity for `2^r_cols_log2` r-columns.
    pub fn zero(n: usize, r_cols_log2: usize) -> Self {
        let r_cols_capacity = 1 << r_cols_log2;
        let mut tableau = vec![0; tableau_block_length(n, r_cols_capacity)];
        for i in 0..n {
            let block_index = z_column_block_index(n, i / BLOCK_SIZE, i);
            tableau[block_index] = bitmask(i % BLOCK_SIZE);
        }
        ExtendedTableau {
            n,
            r_cols: 1,
            tableau,
        }
    }

    pub fn apply_s_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            }
            self.tableau[z] ^= self.tableau[x];
        }
    }
    pub fn apply_sdg_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            self.tableau[z] ^= self.tableau[x];
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            }
        }
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            }
            self.tableau.swap(z, x);
        }
    }
    pub fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let xa = x_column_block_index(n, i, a);
            let za = z_column_block_index(n, i, a);
            let xb = x_column_block_index(n, i, b);
            let zb = z_column_block_index(n, i, b);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^=
                    self.tableau[xa] & self.tableau[zb] & !(self.tableau[xb] ^ self.tableau[za]);
            }
            self.tableau[za] ^= self.tableau[zb];
            self.tableau[xb] ^= self.tableau[xa];
        }
    }
    pub fn apply_cz_gate(&mut self, a: usize, b: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let xa = x_column_block_index(n, i, a);
            let za = z_column_block_index(n, i, a);
            let xb = x_column_block_index(n, i, b);
            let zb = z_column_block_index(n, i, b);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                // TODO: Simplify expression?
                self.tableau[r] ^= (self.tableau[xb] & self.tableau[zb])
                    ^ (self.tableau[xa]
                        & self.tableau[xb]
                        & !(self.tableau[zb] ^ self.tableau[za]))
                    ^ (self.tableau[xb] & (self.tableau[zb] ^ self.tableau[xa]));
            }
            self.tableau[za] ^= self.tableau[xb];
            self.tableau[zb] ^= self.tableau[xa];
        }
    }
    pub fn apply_x_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let z = z_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[z];
            }
        }
    }
    pub fn apply_y_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[x] ^ self.tableau[z];
            }
        }
    }
    pub fn apply_z_gate(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r = r_column_block_index(n, i, j);
                self.tableau[r] ^= self.tableau[x];
            }
        }
    }

    /// Double the number of r-columns in the tableau to represent the current state of the tableau both with and without the Z(a) gate applied.
    ///
    /// The first half of the resulting r-columns will be unchanged,
    /// while the second half will be those where the Z(a) gate is applied.
    pub fn split_r_columns(&mut self, a: usize) {
        let n = self.n;
        let r_cols = self.r_cols;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            for j in 0..r_cols {
                let r1 = r_column_block_index(n, i, j);
                let r2 = r_column_block_index(n, i, j + r_cols);
                self.tableau[r2] = self.tableau[r1] ^ self.tableau[x];
            }
        }
        self.r_cols *= 2;
    }

    /// Compute the ratio of the coefficients of `w1` and `w2`, such that
    /// ```text
    /// coefficient_ratio(w1, w2) * coeff(w1) = coeff(w2)
    /// ```
    ///
    /// This will respect the sign of the `i`th column.
    pub fn coeff_ratio(&mut self, i: usize, w1: &[bool], w2: &[bool]) -> Complex<f64> {
        let n = self.n;
        let r_cols = self.r_cols;
        debug_assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");
        debug_assert_eq!(w2.len(), n, "Basis state 2 must have length {n}");

        let aux_row = n;
        let aux_block_index = aux_row / BLOCK_SIZE;
        let aux_bit_index = aux_row % BLOCK_SIZE;
        let aux_bitmask = bitmask(aux_bit_index);

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Reset the auxiliary row.
        for r in 0..(n + n + r_cols) {
            self.tableau[column_block_index(n, aux_block_index, r)] &= !aux_bitmask;
        }
        // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
        let mut row = 0;
        for q in 0..n {
            if self.x_bit(row, q) == true {
                if w1[q] != w2[q] {
                    self.multiply_rows_into(row, aux_row);
                }
                row += 1;
            }
        }

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        self.stabilizer_matrix_entry(i, aux_row, w1, w2)
    }
    /// Same as [`Self::coeff_ratio`], but for the special case where `w2` is equal to `w1` except for a single flipped bit.
    ///
    /// This will respect the sign of the `i`th column.
    pub fn coeff_ratio_flipped_bit(
        &mut self,
        i: usize,
        w1: &[bool],
        flipped_bit: usize,
    ) -> Complex<f64> {
        let n = self.n;
        debug_assert_eq!(w1.len(), n, "Basis state 1 must have length {n}");

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Identify the row with a set bit in the given position.
        let mut row = None;
        for r in 0..n {
            if self.x_bit(r, flipped_bit) == true {
                row = Some(r);
                break;
            }
        }

        // Compute the (w2, w1) entry in the stabilizer of the correct form.
        let w2 = w1
            .iter()
            .enumerate()
            .map(|(i, &b)| if i == flipped_bit { !b } else { b });
        match row {
            None => Complex::ZERO,
            Some(row) => self.stabilizer_matrix_entry(i, row, w1, w2),
        }
    }

    /// Bring tableau's x part into reduced row echelon form by performing a series of row multiplications.
    ///
    /// This should take O(n^2) time, plus an additional O(n^2) time for each gate that has been applied since the last call to this function.
    fn bring_into_rref(&mut self) {
        let n = self.n;
        let r_cols = self.r_cols;

        let aux_row = n;
        let aux_block_index = aux_row / BLOCK_SIZE;
        let aux_bit_index = aux_row % BLOCK_SIZE;

        let mut a = 0;
        for col in 0..n {
            // Find pivot row.
            let mut pivot = None;
            let a_block_index = a / BLOCK_SIZE;
            for i in (a_block_index..column_block_length(n)).rev() {
                // Bitmask blocking out the auxiliary row.
                let aux_mask = if i == aux_block_index {
                    !bitmask(aux_bit_index)
                } else {
                    !0
                };
                let block = self.tableau[x_column_block_index(n, i, col)] & aux_mask;
                if block != 0 {
                    let row = BLOCK_SIZE * i + lsb_index(block);
                    if row >= a {
                        pivot = Some(row);
                        break;
                    }
                }
            }

            if let Some(pivot) = pivot {
                let pivot_block_index = pivot / BLOCK_SIZE;
                let pivot_bit_index = pivot % BLOCK_SIZE;
                for i in 0..=pivot_block_index {
                    // Bitmask blocking out the pivot row.
                    let pivot_mask = if i == pivot_block_index {
                        !bitmask(pivot_bit_index)
                    } else {
                        !0
                    };
                    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
                    let mask = self.tableau[x_column_block_index(n, i, col)] & pivot_mask;
                    if mask == 0 {
                        continue;
                    }

                    // Determine phase change caused by multiplication of the individual Pauli matrices.
                    // We encode phase as `phase = 2*phase_bit2 + phase_bit1`,
                    // but in a bit block so we can operate on all rows in the block at once.
                    // Since i^phase works modulo 4, we can just use two bits and let additions/subtractions wrap around.
                    let mut phase_bit1: BitBlock = 0;
                    let mut phase_bit2: BitBlock = 0;
                    for col2 in 0..n {
                        fn x(x: BitBlock, z: BitBlock) -> BitBlock {
                            x & !z
                        }
                        fn z(x: BitBlock, z: BitBlock) -> BitBlock {
                            !x & z
                        }
                        fn y(x: BitBlock, z: BitBlock) -> BitBlock {
                            x & z
                        }

                        let x1 = self.tableau[x_column_block_index(n, i, col2)];
                        let z1 = self.tableau[z_column_block_index(n, i, col2)];
                        // Fill these blocks with the bits in the pivot row.
                        let x2 = if self.x_bit(pivot, col2) { !0 } else { 0 };
                        let z2 = if self.z_bit(pivot, col2) { !0 } else { 0 };

                        // XY = +iZ
                        // YZ = +iX
                        // ZX = +iY
                        let add = (x(x1, z1) & y(x2, z2))
                            | (y(x1, z1) & z(x2, z2))
                            | (z(x1, z1) & x(x2, z2));
                        phase_bit2 ^= add & phase_bit1;
                        phase_bit1 ^= add;

                        // YX = -iZ
                        // ZY = -iX
                        // XZ = -iY
                        let sub = (y(x1, z1) & x(x2, z2))
                            | (z(x1, z1) & y(x2, z2))
                            | (x(x1, z1) & z(x2, z2));
                        phase_bit2 ^= sub & !phase_bit1;
                        phase_bit1 ^= sub;
                    }
                    // A valid stabilizer row can only ever have a prefix of +1 or -1.
                    // phase_bit1 being 1 implies a phase of either 1 or 3, making the prefix i or -i respectively.
                    // This should never be able to happen, and we cannot represent it.
                    debug_assert!(phase_bit1 == 0, "Imaginary sign");
                    // phase_bit2 = 1  =>  phase = 2  =>  i^2 = -1    flip the sign bit.
                    // phase_bit2 = 0  =>  phase = 0  =>  i^0 = +1    do nothing.
                    for j in 0..r_cols {
                        self.tableau[r_column_block_index(n, i, j)] ^= phase_bit2 & mask;
                    }

                    // XOR
                    for j in 0..(n + n + r_cols) {
                        if self.bit(pivot, j) == true {
                            self.tableau[column_block_index(n, i, j)] ^= mask;
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
    ///
    /// This will respect the sign of the `i`th column.
    fn stabilizer_matrix_entry<W1, W2>(&self, i: usize, row: usize, w1: W1, w2: W2) -> Complex<f64>
    where
        W1: IntoIterator<Item: Borrow<bool>>,
        W2: IntoIterator<Item: Borrow<bool>>,
    {
        let n = self.n;
        let mut w1 = w1.into_iter();
        let mut w2 = w2.into_iter();

        let mut res = if self.row_negative(i, row) {
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
        let r_cols = self.r_cols;

        let source_block_index = source / BLOCK_SIZE;
        let target_block_index = target / BLOCK_SIZE;
        let source_bit_index = source % BLOCK_SIZE;
        let target_bit_index = target % BLOCK_SIZE;
        let source_bitmask = bitmask(source_bit_index);
        let target_bitmask = bitmask(target_bit_index);

        // Determine phase shift.
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
                for j in 0..r_cols {
                    self.tableau[r_column_block_index(n, target_block_index, j)] ^= target_bitmask;
                }
            }
            _ => unreachable!("No valid stabilizer can have imaginary phase: {phase}"),
        };

        // XOR
        for j in 0..(n + n + r_cols) {
            let source_block = column_block_index(n, source_block_index, j);
            let target_block = column_block_index(n, target_block_index, j);
            self.tableau[target_block] ^= align_bit_to(
                self.tableau[source_block] & source_bitmask,
                source_bit_index,
                target_bit_index,
            );
        }
    }

    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        let n = self.n;
        let r_cols = self.r_cols;

        let row1_block_index = row1 / BLOCK_SIZE;
        let row2_block_index = row2 / BLOCK_SIZE;
        let row1_bit_index = row1 % BLOCK_SIZE;
        let row2_bit_index = row2 % BLOCK_SIZE;
        let row1_bitmask = bitmask(row1_bit_index);
        let row2_bitmask = bitmask(row2_bit_index);
        for j in 0..(n + n + r_cols) {
            let block1 = column_block_index(n, row1_block_index, j);
            let block2 = column_block_index(n, row2_block_index, j);
            let bit1 = self.tableau[block1] & row1_bitmask != 0;
            let bit2 = self.tableau[block2] & row2_bitmask != 0;
            match (bit1, bit2) {
                (true, false) => {
                    unset_bit(&mut self.tableau[block1], row1_bit_index);
                    set_bit(&mut self.tableau[block2], row2_bit_index);
                }
                (false, true) => {
                    set_bit(&mut self.tableau[block1], row1_bit_index);
                    unset_bit(&mut self.tableau[block2], row2_bit_index);
                }
                _ => {}
            }
        }
    }

    /// Get whether the given row is negative or not, i.e. the contents of the sign bit.
    ///
    /// This will respect the sign of the `i`th column.
    fn row_negative(&self, i: usize, row: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[r_column_block_index(n, row_block_index, i)] & row_bitmask != 0
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

    /// Get the value of the bit corresponding to the `j`th column in the `row`th row.
    fn bit(&self, row: usize, j: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask = bitmask(row_bit_index);
        self.tableau[column_block_index(n, row_block_index, j)] & row_bitmask != 0
    }
    /// Get the value of the x bit corresponding to the `q`th tensor element in the `row`th row.
    fn x_bit(&self, row: usize, q: usize) -> bool {
        self.bit(row, 2 * q)
    }
    /// Get the value of the z bit corresponding to the `q`th tensor element in the `row`th row.
    fn z_bit(&self, row: usize, q: usize) -> bool {
        self.bit(row, 2 * q + 1)
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

/// Get the index of the i'th block of the `j`th column.
fn column_block_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    j * column_block_length(n) + i
}
/// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
fn x_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q)
}
/// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
fn z_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q + 1)
}
/// Get the index of the i'th block of the r column.
fn r_column_block_index(n: usize, i: usize, j: usize) -> usize {
    column_block_index(n, i, 2 * n + j)
}

/// Get the block-length of the columns in the tableau.
fn column_block_length(n: usize) -> usize {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
/// Get the block-length of the tableau.
fn tableau_block_length(n: usize, r_cols: usize) -> usize {
    column_block_length(n) * (n + n + r_cols)
}

/// Bit-shift the given block such that the `from`th bit is moved to the `to`th position.
pub fn align_bit_to(block: BitBlock, from: usize, to: usize) -> BitBlock {
    if to < from {
        block << (from - to)
    } else {
        block >> (to - from)
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::{CliffordTCircuit, CliffordTGate::*};
    use crate::utils::bits_to_bools;

    use super::*;

    #[test]
    fn zero() {
        let circuit = CliffordTCircuit::new(8, []).unwrap();

        let w1 = bits_to_bools(0b0000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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

            let mut g = ExtendedTableau::zero(8, 0);
            apply_clifford_circuit(&mut g, &circuit);
            let result = g.coeff_ratio(0, &w1, &w2);

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
        let mut g = ExtendedTableau::zero(8, 0);
        apply_clifford_circuit(&mut g, &circuit);

        assert_eq!(g.coeff_ratio_flipped_bit(0, &w, 0), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(0, &w, 1), -Complex::ONE);
        assert_eq!(g.coeff_ratio_flipped_bit(0, &w, 2), Complex::ZERO);
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

        let mut g = ExtendedTableau::zero(8, 0);
        apply_clifford_circuit(&mut g, &circuit);

        let w1 = bits_to_bools(0b1000_0000);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratio(0, &w1, &w2);

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

    fn apply_clifford_circuit(g: &mut ExtendedTableau, circuit: &CliffordTCircuit) {
        for &gate in circuit.gates() {
            match gate {
                S(a) => g.apply_s_gate(a),
                H(a) => g.apply_h_gate(a),
                Cnot(a, b) => g.apply_cnot_gate(a, b),
                _ => unreachable!(),
            }
        }
    }
}
