use std::fmt::Debug;
use std::mem;

use num_complex::Complex;

use crate::bitstring::BitStringArray;
use crate::utils::{bit_indices, bitmask, flip_bit, set_bit, unset_bit};

type BitBlock = u64;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

#[derive(Debug, PartialEq, Eq)]
enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// An extended stabilizer tableau.
///
/// Able to represent a sequence of stabilizer states, differing by single-qubit Pauli rotations.
#[derive(Clone)]
pub struct ExtendedTableau {
    /// The number of qubits in the tableau.
    n: usize,
    /// The current number of active c-columns in the tableau.
    c_cols: usize,
    /// The augmented stabilizer tableau,
    /// ```text
    /// P1 -> x1 x2 ... xn | z1 z2 ... zn | r | c1 c2 ... ct
    /// P2 -> x1 x2 ... xn | z1 z2 ... zn | r | c1 c2 ... ct
    /// ...
    /// Pn -> x1 x2 ... xn | z1 z2 ... zn | r | c1 c2 ... ct
    /// ```
    /// is laid out column-wise in the following way:
    /// ```text
    /// P1 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
    /// P2 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
    /// ...
    /// Pn -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
    /// (E -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct)
    /// ```
    /// Note that the x and z columns are interleaved, and that an auxiliary row, E, is added at the end.
    /// This auxiliary row is assumed to be kept zeroed.
    tableau: Vec<BitBlock>,
    row_pivots: Vec<Option<usize>>,
    /// Buffer used to store the output of [`Self::coeff_ratios`] and [`Self::coeff_ratios_flipped_bit`].
    /// Must have length of at least 2^`c_cols` at all times.
    output: Vec<Complex<f64>>,
}
impl ExtendedTableau {
    /// Initialize a new tableau with `n` qubits in the initial zero state.
    /// This allocates an extended tableau with capacity for representing a total of 2^`capacity_log2` states.
    pub fn zero(n: usize, capacity_log2: usize) -> Self {
        let mut tableau = vec![0; tableau_block_length(n, capacity_log2)];
        for i in 0..n {
            let block_index = z_column_block_index(n, i / BLOCK_SIZE, i);
            tableau[block_index] = bitmask(i % BLOCK_SIZE);
        }
        ExtendedTableau {
            n,
            c_cols: 0,
            tableau,
            row_pivots: vec![None; n],
            output: vec![Complex::ZERO; 1 << capacity_log2],
        }
    }
    #[cfg(test)]
    pub fn random(n: usize, capacity_log2: usize, seed: u64) -> Self {
        use rand::rngs::Xoshiro128PlusPlus;
        use rand::{RngExt, SeedableRng};

        let c_cols = capacity_log2;
        let rng = Xoshiro128PlusPlus::seed_from_u64(seed);
        let mut tableau: Vec<BitBlock> = rng
            .random_iter()
            .take(tableau_block_length(n, c_cols))
            .collect();
        for j in 0..(n + n + 1 + c_cols) {
            let block_index = column_block_index(n, column_block_length(n) - 1, j);
            tableau[block_index] &= !0 << (BLOCK_SIZE - (n % BLOCK_SIZE));
        }
        ExtendedTableau {
            n,
            c_cols,
            tableau,
            row_pivots: vec![None; n],
            output: vec![Complex::ZERO; 1 << capacity_log2],
        }
    }

    /// Return the number of states currently represented by the tableau.
    #[inline]
    pub fn contained_states(&self) -> usize {
        1 << self.c_cols
    }

    pub fn apply_s_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            self.tableau[z] ^= self.tableau[x];
        }
    }
    pub fn apply_sdg_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[z] ^= self.tableau[x];
            self.tableau[r] ^= self.tableau[x] & self.tableau[z];
        }
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^= self.tableau[x] & self.tableau[z];
            self.tableau.swap(z, x);
        }
    }
    pub fn apply_cnot_gate(&mut self, a: usize, b: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let xa = x_column_block_index(n, i, a);
            let za = z_column_block_index(n, i, a);
            let xb = x_column_block_index(n, i, b);
            let zb = z_column_block_index(n, i, b);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^=
                self.tableau[xa] & self.tableau[zb] & !(self.tableau[xb] ^ self.tableau[za]);
            self.tableau[za] ^= self.tableau[zb];
            self.tableau[xb] ^= self.tableau[xa];
        }
    }
    pub fn apply_cz_gate(&mut self, a: usize, b: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let xa = x_column_block_index(n, i, a);
            let za = z_column_block_index(n, i, a);
            let xb = x_column_block_index(n, i, b);
            let zb = z_column_block_index(n, i, b);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^=
                self.tableau[xa] & self.tableau[xb] & (self.tableau[za] ^ self.tableau[zb]);
            self.tableau[za] ^= self.tableau[xb];
            self.tableau[zb] ^= self.tableau[xa];
        }
    }
    pub fn apply_x_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let z = z_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^= self.tableau[z];
        }
    }
    pub fn apply_y_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^= self.tableau[x] ^ self.tableau[z];
        }
    }
    pub fn apply_z_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let r = r_column_block_index(n, i);
            self.tableau[r] ^= self.tableau[x];
        }
    }

    /// Fork the extended tableau such that it represents the current sequence of states,
    /// appended by the same sequence but with the Z-gate applied to qubit `a` of each state,
    /// doubling the number of states represented by the tableau.
    pub fn fork_apply_z_gate(&mut self, a: usize) {
        let n = self.n;
        let c_cols = self.c_cols;
        let x = x_column_block_index(n, 0, a);
        let c = c_column_block_index(n, 0, c_cols);
        self.tableau.copy_within(x..(x + column_block_length(n)), c);
        self.c_cols += 1;
    }

    /// The coeff. ratio describes the ratio of the coefficients of `w1` and `w2`, such that
    /// ```text
    /// coeff_ratio(w1, w2) * coeff(w1) = coeff(w2)
    /// ```
    ///
    /// This function takes an array of basis states `w1s` with length at least equal to [`Self::contained_states`],
    /// and returns a slice of the coeff. ratios between each `w1s[i]` and `w2`,
    /// each respecting the state of the i'th state in the sequence.
    /// The output will have length exactly equal to [`Self::contained_states`].
    pub fn coeff_ratios(&mut self, w1s: &BitStringArray, w2: &[bool]) -> &[Complex<f64>] {
        let n = self.n;
        let c_cols = self.c_cols;
        let contained_states = self.contained_states();
        debug_assert!(
            w1s.len() >= contained_states,
            "Basis state 1 must have length at least {contained_states}"
        );
        debug_assert_eq!(w2.len(), n, "Basis state 2 must have length {n}");

        let aux_row = n;
        let aux_block_index = aux_row / BLOCK_SIZE;
        let aux_bit_index = aux_row % BLOCK_SIZE;

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        for s in 0..contained_states {
            // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
            let mut mask: Vec<BitBlock> = vec![0; column_block_length(n)];
            for row in 0..n {
                if let Some(q) = self.row_pivots[row]
                    && w1s.get(s, q) != w2[q]
                {
                    let row_block_index = row / BLOCK_SIZE;
                    let row_bit_index = row % BLOCK_SIZE;
                    mask[row_block_index] = set_bit(mask[row_block_index], row_bit_index);
                }
            }
            // XOR
            for j in 0..(n + n + 1 + c_cols) {
                // Start by going block-wise, reducing to a single block
                let mut block: BitBlock = 0;
                for i in 0..column_block_length(n) {
                    block ^= self.tableau[column_block_index(n, i, j)] & mask[i];
                }
                // Reduce last block:
                // The XOR of all bits in a block is just the parity
                let block_index = column_block_index(n, aux_block_index, j);
                self.tableau[block_index] = if block.count_ones() % 2 != 0 {
                    set_bit(self.tableau[block_index], aux_bit_index)
                } else {
                    unset_bit(self.tableau[block_index], aux_bit_index)
                };
            }
            // Determine phase change caused by multiplication of the individual Pauli matrices.
            // These phases are encoded with phase = 2*phase_bit2 + phase_bit1.
            let mut phase_bit1: BitBlock = 0;
            let mut phase_bit2: BitBlock = 0;
            for col in 0..n {
                let mut x1 = 0;
                let mut z1 = 0;
                // Reduce block-wise
                for i in 0..column_block_length(n) {
                    let x2 = self.tableau[x_column_block_index(n, i, col)] & mask[i];
                    let z2 = self.tableau[z_column_block_index(n, i, col)] & mask[i];
                    apply_phase_shift(x1, z1, x2, z2, &mut phase_bit1, &mut phase_bit2);
                    x1 ^= x2;
                    z1 ^= z2;
                }
                // Reduce the final block by iteratively 'folding' it in half, e.g.
                // X
                // Y
                // Z    ZX = +iY
                // X -> XY = +iZ -> ZY = -iX
                for e in 1..=BLOCK_SIZE.ilog2() {
                    let shift = BLOCK_SIZE / 2usize.pow(e);
                    let x2 = x1 >> shift;
                    let z2 = z1 >> shift;
                    let low_mask = !0 >> (BLOCK_SIZE - shift);
                    x1 &= low_mask;
                    z1 &= low_mask;
                    apply_phase_shift(x1, z1, x2, z2, &mut phase_bit1, &mut phase_bit2);
                    x1 ^= x2;
                    z1 ^= z2;
                }
            }
            let phase = (2 * phase_bit2.count_ones() + phase_bit1.count_ones()) % 4;
            debug_assert!(phase % 2 == 0, "Imaginary sign");
            if phase == 2 {
                let block_index = r_column_block_index(n, aux_block_index);
                self.tableau[block_index] = flip_bit(self.tableau[block_index], aux_bit_index);
            }

            // Compute the (w2, w1) entry in the stabilizer of the correct form.
            self.output[s] =
                self.stabilizer_matrix_entry(s, aux_row, w1s.iter_string(s), w2.iter().copied());
        }
        // Reset the auxiliary row.
        for j in 0..(n + n + 1 + c_cols) {
            let block_index = column_block_index(n, aux_block_index, j);
            self.tableau[block_index] = unset_bit(self.tableau[block_index], aux_bit_index);
        }
        &self.output[..contained_states]
    }
    /// Same as [`Self::coeff_ratios`], but for the special case where `w2` is equal to `w1` except for a single flipped bit.
    pub fn coeff_ratios_flipped_bit(
        &mut self,
        w1s: &BitStringArray,
        flipped_bit: usize,
    ) -> (Complex<f64>, Vec<bool>) {
        let n = self.n;
        let contained_states = self.contained_states();

        // Bring tableau's x part into reduced row echelon form.
        self.bring_into_rref();

        // Identify the row with a set bit in the given position.
        let mut row = None;
        for r in 0..n {
            if self.row_pivots[r] == Some(flipped_bit) {
                row = Some(r);
                break;
            }
        }

        match row {
            None => (Complex::ZERO, vec![]),
            Some(row) => {
                if self.stabilizer_matrix_entry_is_zero(row, (0..n).map(|i| i == flipped_bit)) {
                    return (Complex::ZERO, vec![]);
                }
                let r = self.stabilizer_matrix_entry_phase_part(row);
                let mut signs = vec![];
                for s in 0..contained_states {
                    signs.push(self.stabilizer_matrix_entry_sign_part(s, row, w1s.iter_string(s)));
                }
                (r, signs)
            }
        }
    }

    /// Bring tableau's x part into reduced row echelon form by performing a series of row multiplications.
    ///
    /// This should take O(n^2) time, plus an additional O(n^2) time for each gate that has been applied since the last call to this function.
    fn bring_into_rref(&mut self) {
        let n = self.n;
        let c_cols = self.c_cols;

        // Bitmask with zeros in indices corresponding to rows where pivots have already been seen
        let mut pivot_mask: Vec<BitBlock> = vec![!0; column_block_length(n)];

        for col in 0..n {
            // Find pivot row.
            let mut pivot = None;
            let mut m = None;
            for block_index in 0..column_block_length(n) {
                let block = self.tableau[x_column_block_index(n, block_index, col)]
                    & pivot_mask[block_index];
                for bit_index in bit_indices(block) {
                    let row = BLOCK_SIZE * block_index + bit_index;
                    if m <= self.row_pivots[row] {
                        pivot = Some(row);
                        m = self.row_pivots[row]
                    }
                    self.row_pivots[row] = Some(col);
                }
            }

            if let Some(pivot) = pivot {
                let pivot_block_index = pivot / BLOCK_SIZE;
                let pivot_bit_index = pivot % BLOCK_SIZE;
                pivot_mask[pivot_block_index] =
                    unset_bit(pivot_mask[pivot_block_index], pivot_bit_index);
                self.row_pivots[pivot] = Some(col);

                for i in 0..column_block_length(n) {
                    // Bitmask blocking out the pivot row.
                    let pivot_mask = if i == pivot_block_index {
                        !bitmask::<BitBlock>(pivot_bit_index)
                    } else {
                        !0
                    };
                    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
                    let mask = self.tableau[x_column_block_index(n, i, col)] & pivot_mask;
                    if mask == 0 {
                        continue;
                    }

                    // Determine phase change caused by multiplication of the individual Pauli matrices.
                    // These phases are encoded with phase = 2*phase_bit2 + phase_bit1.
                    let mut phase_bit1: BitBlock = 0;
                    let mut phase_bit2: BitBlock = 0;
                    for col2 in 0..n {
                        let x1 = self.tableau[x_column_block_index(n, i, col2)];
                        let z1 = self.tableau[z_column_block_index(n, i, col2)];
                        // Fill these blocks with the bits in the pivot row.
                        let x2 = if self.x_bit(pivot, col2) { !0 } else { 0 };
                        let z2 = if self.z_bit(pivot, col2) { !0 } else { 0 };

                        apply_phase_shift(x1, z1, x2, z2, &mut phase_bit1, &mut phase_bit2);
                    }
                    // A valid stabilizer row can only ever have a prefix of +1 or -1.
                    // phase_bit1 being 1 implies a phase of either 1 or 3, making the prefix i or -i respectively.
                    // This should never be able to happen, and we cannot represent it.
                    debug_assert!(phase_bit1 == 0, "Imaginary sign");
                    // phase_bit2 = 1  =>  phase = 2  =>  i^2 = -1    flip the sign bit.
                    // phase_bit2 = 0  =>  phase = 0  =>  i^0 = +1    do nothing.
                    self.tableau[r_column_block_index(n, i)] ^= phase_bit2 & mask;

                    // XOR
                    for j in 0..(n + n + 1 + c_cols) {
                        if self.bit(pivot, j) {
                            self.tableau[column_block_index(n, i, j)] ^= mask;
                        }
                    }
                }
            }
        }

        // Reset the pivots of all-zero rows
        for (block_index, &block) in pivot_mask.iter().enumerate() {
            for bit_index in bit_indices(block) {
                let row = BLOCK_SIZE * block_index + bit_index;
                if row >= n {
                    return;
                }
                self.row_pivots[row] = None;
            }
        }
    }

    /// Returns true if and only if [`Self::stabilizer_matrix_entry`] returns [`Complex::ZERO`].
    ///
    /// `w1_xor_w2` should be equal to the bitwise XOR of `w1` and `w2`, i.e. for each bit, whether they differ or not.
    fn stabilizer_matrix_entry_is_zero<W>(&self, row: usize, mut w1_xor_w2: W) -> bool
    where
        W: Iterator<Item = bool>,
    {
        let n = self.n;

        for q in 0..n {
            let different = w1_xor_w2.next().unwrap();
            if self.x_bit(row, q) != different {
                return true;
            }
        }
        false
    }
    /// Computes the factor of [`Self::stabilizer_matrix_entry`] caused by the complex phase rotation
    /// contributed by each [`Pauli::Y`] element in the tensor product.
    fn stabilizer_matrix_entry_phase_part(&self, row: usize) -> Complex<f64> {
        let n = self.n;

        let mut res = Complex::ONE;
        for q in 0..n {
            if self.tensor_element(row, q) == Pauli::Y {
                res *= Complex::I
            }
        }
        res
    }
    /// Computes the factor of [`Self::stabilizer_matrix_entry`] caused by simple sign flips.
    ///
    /// This does *NOT* include the phase rotation caused by [`Pauli::Y`] elements,
    /// computed by [`Self::stabilizer_matrix_entry_phase_part`].
    fn stabilizer_matrix_entry_sign_part<W1>(&self, i: usize, row: usize, mut w1: W1) -> bool
    where
        W1: Iterator<Item = bool>,
    {
        let n = self.n;

        let mut res = self.row_negative(i, row);
        for q in 0..n {
            let b1 = w1.next().unwrap();
            if self.z_bit(row, q) && b1 {
                res = !res;
            }
        }
        res
    }
    /// Compute the entry of the row'th stabilizer matrix, `P[w2, w1]`, for the given basis state pair.
    ///
    /// This will respect the state of the i'th tableau in the sequence.
    fn stabilizer_matrix_entry<W1, W2>(&self, i: usize, row: usize, w1: W1, w2: W2) -> Complex<f64>
    where
        W1: Iterator<Item = bool> + Clone,
        W2: Iterator<Item = bool>,
    {
        if self.stabilizer_matrix_entry_is_zero(row, w1.clone().zip(w2).map(|(b1, b2)| b1 != b2)) {
            return Complex::ZERO;
        }
        let p = self.stabilizer_matrix_entry_phase_part(row);
        if self.stabilizer_matrix_entry_sign_part(i, row, w1) {
            -p
        } else {
            p
        }
    }
    /// This is the less clever version of [`Self::stabilizer_matrix_entry`],
    /// used only in testing to validate the more optimized version.
    #[cfg(test)]
    fn stabilizer_matrix_entry_reference<W1, W2>(
        &self,
        i: usize,
        row: usize,
        mut w1: W1,
        mut w2: W2,
    ) -> Complex<f64>
    where
        W1: Iterator<Item = bool>,
        W2: Iterator<Item = bool>,
    {
        let n = self.n;

        let mut res = if self.row_negative(i, row) {
            -Complex::ONE
        } else {
            Complex::ONE
        };
        for q in 0..n {
            // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
            res *= match (
                self.tensor_element(row, q),
                w1.next().unwrap(),
                w2.next().unwrap(),
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
    /// Get whether the given row is negative or not, i.e. the contents of the sign bit.
    ///
    /// This will respect the sign of the i'th state.
    fn row_negative(&self, mut i: usize, row: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask: BitBlock = bitmask(row_bit_index);
        let mut r = self.tableau[r_column_block_index(n, row_block_index)];
        let mut j = 0;
        while i != 0 {
            if i % 2 != 0 {
                r ^= self.tableau[c_column_block_index(n, row_block_index, j)];
            }
            i /= 2;
            j += 1;
        }
        r & row_bitmask != 0
    }

    /// Get the Pauli matrix corresponding to the q'th tensor element in the `row`'th row.
    fn tensor_element(&self, row: usize, q: usize) -> Pauli {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask: BitBlock = bitmask(row_bit_index);

        let x = self.tableau[x_column_block_index(n, row_block_index, q)] & row_bitmask != 0;
        let z = self.tableau[z_column_block_index(n, row_block_index, q)] & row_bitmask != 0;

        match (x, z) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (true, true) => Pauli::Y,
            (false, true) => Pauli::Z,
        }
    }

    /// Get the value of the bit corresponding to the j'th column in the `row`'th row.
    #[inline]
    fn bit(&self, row: usize, j: usize) -> bool {
        let n = self.n;
        let row_block_index = row / BLOCK_SIZE;
        let row_bit_index = row % BLOCK_SIZE;
        let row_bitmask: BitBlock = bitmask(row_bit_index);
        self.tableau[column_block_index(n, row_block_index, j)] & row_bitmask != 0
    }
    /// Get the value of the x bit corresponding to the q'th tensor element in the `row`'th row.
    #[inline]
    fn x_bit(&self, row: usize, q: usize) -> bool {
        self.bit(row, 2 * q)
    }
    /// Get the value of the z bit corresponding to the q'th tensor element in the `row`'th row.
    #[inline]
    fn z_bit(&self, row: usize, q: usize) -> bool {
        self.bit(row, 2 * q + 1)
    }
    /// Get the value of the r bit corresponding to the `row`'th row.
    #[inline]
    fn r_bit(&self, row: usize) -> bool {
        let n = self.n;
        self.bit(row, n + n)
    }
    /// Get the value of the c bit corresponding to the j'th column in the `row`'th row.
    #[inline]
    fn c_bit(&self, row: usize, j: usize) -> bool {
        let n = self.n;
        self.bit(row, n + n + 1 + j)
    }
}
impl Debug for ExtendedTableau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.n;
        let c_cols = self.c_cols;
        for row in 0..(n + 1) {
            write!(
                f,
                "\n\t{} -> ",
                if row == n {
                    "E".to_owned()
                } else {
                    self.row_pivots[row].map_or("-".to_owned(), |v| v.to_string())
                }
            )?;
            for q in 0..n {
                write!(f, "{} ", if self.x_bit(row, q) { "1" } else { "0" })?;
            }
            write!(f, "| ")?;
            for q in 0..n {
                write!(f, "{} ", if self.z_bit(row, q) { "1" } else { "0" })?;
            }
            write!(f, "| ")?;
            write!(f, "{} ", if self.r_bit(row) { "1" } else { "0" })?;
            write!(f, "| ")?;
            for j in 0..c_cols {
                write!(f, "{} ", if self.c_bit(row, j) { "1" } else { "0" })?;
            }
        }
        Ok(())
    }
}

/// Given two block-pairs encoding two vectors of Pauli operators, A and B,
/// and the block-pair encoding a vector of phases,
/// updates these phases entry-wise respective to the effect of multiplying A with B.
///
/// E.g. for some bit-entry in the blocks, if x1=1, z1=0 then the A=X and if x2=1 and z2=1 then B=Y.
/// So we have XY = +iZ, so the phase is updated by this +i part.
///
/// These phases are encoded with phase = 2*phase_bit2 + phase_bit1.
/// Since i^phase works modulo 4, we can just use two bits and let additions/subtractions wrap around.
#[inline]
fn apply_phase_shift(
    x1: BitBlock,
    z1: BitBlock,
    x2: BitBlock,
    z2: BitBlock,
    phase_bit1: &mut BitBlock,
    phase_bit2: &mut BitBlock,
) {
    let x1z2 = x1 & z2;
    let ac = x1z2 ^ (z1 & x2);
    let neg = (x1 ^ z1 ^ x2 ^ z2) ^ x1z2;
    *phase_bit2 ^= ac & (*phase_bit1 ^ neg);
    *phase_bit1 ^= ac;
}
/// This is the less clever version of [`apply_phase_shift`],
/// used only in testing to validate the more optimized version.
#[cfg(test)]
fn apply_phase_shift_reference(
    x1: BitBlock,
    z1: BitBlock,
    x2: BitBlock,
    z2: BitBlock,
    phase_bit1: &mut BitBlock,
    phase_bit2: &mut BitBlock,
) {
    fn x(x: BitBlock, z: BitBlock) -> BitBlock {
        x & !z
    }
    fn z(x: BitBlock, z: BitBlock) -> BitBlock {
        !x & z
    }
    fn y(x: BitBlock, z: BitBlock) -> BitBlock {
        x & z
    }

    // XY = +iZ
    // YZ = +iX
    // ZX = +iY
    let add = (x(x1, z1) & y(x2, z2)) | (y(x1, z1) & z(x2, z2)) | (z(x1, z1) & x(x2, z2));
    *phase_bit2 ^= add & *phase_bit1;
    *phase_bit1 ^= add;

    // YX = -iZ
    // ZY = -iX
    // XZ = -iY
    let sub = (y(x1, z1) & x(x2, z2)) | (z(x1, z1) & y(x2, z2)) | (x(x1, z1) & z(x2, z2));
    *phase_bit2 ^= sub & !*phase_bit1;
    *phase_bit1 ^= sub;
}

/// Get the index of the i'th block of the `j`th column.
#[inline]
fn column_block_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    j * column_block_length(n) + i
}
/// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
#[inline]
fn x_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q)
}
/// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
#[inline]
fn z_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q + 1)
}
/// Get the index of the i'th block of the r column.
#[inline]
fn r_column_block_index(n: usize, i: usize) -> usize {
    column_block_index(n, i, n + n)
}
/// Get the index of the i'th block of the j'th c column.
#[inline]
fn c_column_block_index(n: usize, i: usize, j: usize) -> usize {
    column_block_index(n, i, n + n + 1 + j)
}

/// Get the block-length of the columns in the tableau.
#[inline]
fn column_block_length(n: usize) -> usize {
    // Make room for the auxiliary row.
    (n + 1).div_ceil(BLOCK_SIZE)
}
/// Get the block-length of the tableau.
#[inline]
fn tableau_block_length(n: usize, c_cols: usize) -> usize {
    column_block_length(n) * (n + n + 1 + c_cols)
}

#[cfg(test)]
mod tests {
    use crate::utils::bits_to_bools;
    use rand::rngs::Xoshiro128PlusPlus;
    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn zero() {
        let w1 = BitStringArray::from_u8s(&[0b0000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            let result = g.coeff_ratios(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn imaginary() {
        let w1 = BitStringArray::from_u8s(&[0b0000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            g.apply_h_gate(0);
            g.apply_s_gate(0);
            let result = g.coeff_ratios(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                Complex::ONE
            } else if i == 0b1000_0000 {
                Complex::I
            } else {
                Complex::ZERO
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn negative_imaginary() {
        let w1 = BitStringArray::from_u8s(&[0b1000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            g.apply_h_gate(0);
            g.apply_s_gate(0);
            let result = g.coeff_ratios(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                -Complex::I
            } else if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn flipped() {
        let w1 = BitStringArray::from_u8s(&[0b1000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            g.apply_h_gate(0);
            g.apply_s_gate(0);
            g.apply_s_gate(0);
            g.apply_h_gate(0);
            let result = g.coeff_ratios(&w1, &w2);

            let expected = if i == 0b1000_0000 {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn bell_state() {
        let w1 = BitStringArray::from_u8s(&[0b1100_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            g.apply_h_gate(0);
            g.apply_cnot_gate(0, 1);
            let result = g.coeff_ratios(&w1, &w2);

            let expected = if [0b0000_0000, 0b1100_0000].contains(&i) {
                Complex::ONE
            } else {
                Complex::ZERO
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn larger_circuit() {
        let w1 = BitStringArray::from_u8s(&[0b1000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let mut g = ExtendedTableau::zero(8, 0);
            g.apply_h_gate(0);
            g.apply_h_gate(1);
            g.apply_s_gate(2);
            g.apply_h_gate(3);
            g.apply_s_gate(1);
            g.apply_s_gate(0);
            g.apply_cnot_gate(2, 3);
            g.apply_s_gate(1);
            g.apply_h_gate(0);
            g.apply_s_gate(3);
            g.apply_cnot_gate(1, 0);
            g.apply_s_gate(3);
            g.apply_h_gate(1);
            g.apply_s_gate(3);
            g.apply_s_gate(1);
            g.apply_s_gate(3);
            g.apply_h_gate(1);
            g.apply_cnot_gate(3, 2);
            g.apply_h_gate(1);
            g.apply_cnot_gate(3, 1);
            let result = g.coeff_ratios(&w1, &w2);

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
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn bitflip_ratio() {
        let w1 = BitStringArray::from_u8s(&[0b1000_0000]);
        let mut g = ExtendedTableau::zero(8, 0);
        g.apply_h_gate(0);
        g.apply_h_gate(1);
        g.apply_s_gate(2);
        g.apply_h_gate(3);
        g.apply_s_gate(1);
        g.apply_s_gate(0);
        g.apply_cnot_gate(2, 3);
        g.apply_s_gate(1);
        g.apply_h_gate(0);
        g.apply_s_gate(3);
        g.apply_cnot_gate(1, 0);
        g.apply_s_gate(3);
        g.apply_h_gate(1);
        g.apply_s_gate(3);
        g.apply_s_gate(1);
        g.apply_s_gate(3);
        g.apply_h_gate(1);
        g.apply_cnot_gate(3, 2);
        g.apply_h_gate(1);
        g.apply_cnot_gate(3, 1);

        assert_eq!(
            g.coeff_ratios_flipped_bit(&w1, 0),
            (Complex::ONE, vec![true])
        );
        assert_eq!(
            g.coeff_ratios_flipped_bit(&w1, 1),
            (Complex::ONE, vec![true])
        );
        assert_eq!(g.coeff_ratios_flipped_bit(&w1, 2), (Complex::ZERO, vec![]));
    }

    #[test]
    fn repeated_reading() {
        let mut g = ExtendedTableau::zero(8, 0);
        g.apply_h_gate(0);
        g.apply_h_gate(1);
        g.apply_s_gate(2);
        g.apply_h_gate(3);
        g.apply_s_gate(1);
        g.apply_s_gate(0);
        g.apply_cnot_gate(2, 3);
        g.apply_s_gate(1);
        g.apply_h_gate(0);
        g.apply_s_gate(3);
        g.apply_cnot_gate(1, 0);
        g.apply_s_gate(3);
        g.apply_h_gate(1);
        g.apply_s_gate(3);
        g.apply_s_gate(1);
        g.apply_s_gate(3);
        g.apply_h_gate(1);
        g.apply_cnot_gate(3, 2);
        g.apply_h_gate(1);
        g.apply_cnot_gate(3, 1);

        let w1 = BitStringArray::from_u8s(&[0b1000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratios(&w1, &w2);

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
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    #[test]
    fn fork_apply_z_gate() {
        let mut g = ExtendedTableau::zero(8, 1);
        g.apply_h_gate(0);
        g.fork_apply_z_gate(0);

        let w1 = BitStringArray::from_u8s(&[0b0000_0000, 0b1000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratios(&w1, &w2);

            let expected = if i == 0b0000_0000 {
                [Complex::ONE, -Complex::ONE]
            } else if i == 0b1000_0000 {
                [Complex::ONE, Complex::ONE]
            } else {
                [Complex::ZERO, Complex::ZERO]
            };
            assert_eq!(result, expected, "{i:008b}");
        }
    }

    #[test]
    fn large_tableau() {
        let mut g = ExtendedTableau::zero(300, 3);
        g.apply_h_gate(1);
        g.fork_apply_z_gate(1);
        g.apply_h_gate(78);
        g.fork_apply_z_gate(78);
        g.apply_h_gate(123);
        g.fork_apply_z_gate(123);

        let w1 = BitStringArray::new(300, 8);
        let mut w2 = [false; 300];
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ONE; 8]);
        w2[1] = true;
        assert_eq!(
            g.coeff_ratios(&w1, &w2),
            [
                Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
            ]
        );
        w2[78] = true;
        assert_eq!(
            g.coeff_ratios(&w1, &w2),
            [
                Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
            ]
        );
        w2[123] = true;
        assert_eq!(
            g.coeff_ratios(&w1, &w2),
            [
                Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
            ]
        );
        w2[1] = false;
        assert_eq!(
            g.coeff_ratios(&w1, &w2),
            [
                Complex::ONE,
                Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                -Complex::ONE,
                Complex::ONE,
                Complex::ONE,
            ]
        );
        w2[42] = true;
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ZERO; 8]);
    }

    #[test]
    fn linear_cluster_state() {
        let mut g = ExtendedTableau::zero(8, 0);
        for q in 0..8 {
            g.apply_h_gate(q);
        }
        for q in 0..7 {
            g.apply_cz_gate(q, q + 1);
        }

        let w1 = BitStringArray::from_u8s(&[0b0000_0000]);
        for i in 0b0000_0000..=0b1111_1111 {
            let w2 = bits_to_bools(i);

            let result = g.coeff_ratios(&w1, &w2);

            let adjacent_pairs = (0..7).filter(|&q| w2[q] && w2[q + 1]).count();
            let expected = if adjacent_pairs % 2 == 0 {
                Complex::ONE
            } else {
                -Complex::ONE
            };
            assert_eq!(result[0], expected, "{i:008b}");
        }
    }

    /// This test is here to ensure that coeff_ratios correctly handles phase shifts
    /// resulting from row multiplication between rows across different BitBlocks.
    #[test]
    fn multi_block_phase() {
        let mut g = ExtendedTableau::zero(70, 0);
        for q in 0..70 {
            g.apply_h_gate(q);
        }
        g.apply_cz_gate(0, 64);
        g.apply_cz_gate(0, 69);

        let w1 = BitStringArray::new(70, 1);
        let mut w2 = [false; 70];
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ONE]);
        w2[0] = true;
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ONE]);
        w2[64] = true; // Edge 0-64.
        assert_eq!(g.coeff_ratios(&w1, &w2), [-Complex::ONE]);
        w2[69] = true; // Edges 0-64 and 0-69: this is the case that needs the cross-block phase.
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ONE]);
        w2[64] = false; // Edge 0-69.
        assert_eq!(g.coeff_ratios(&w1, &w2), [-Complex::ONE]);
        w2[0] = false; // No edges.
        assert_eq!(g.coeff_ratios(&w1, &w2), [Complex::ONE]);
    }

    #[test]
    fn compare_apply_phase_shift_reference() {
        let mut rng = Xoshiro128PlusPlus::seed_from_u64(1234);
        for _ in 0..2u64.pow(16) {
            let x1: BitBlock = rng.random();
            let z1: BitBlock = rng.random();
            let x2: BitBlock = rng.random();
            let z2: BitBlock = rng.random();
            let p1: BitBlock = rng.random();
            let p2: BitBlock = rng.random();
            let (mut p1a, mut p2a) = (p1, p2);
            let (mut p1b, mut p2b) = (p1, p2);
            apply_phase_shift(x1, z1, x2, z2, &mut p1a, &mut p2a);
            apply_phase_shift_reference(x1, z1, x2, z2, &mut p1b, &mut p2b);
            assert_eq!((p1a, p2a), (p1b, p2b));
        }
    }

    #[test]
    fn compare_stabilizer_matrix_entry_reference() {
        let rng = &mut Xoshiro128PlusPlus::seed_from_u64(1234);
        let tableau = ExtendedTableau::random(8, 3, 12345);
        for _ in 0..2u64.pow(16) {
            let i = rng.random_range(0..8);
            let row = rng.random_range(0..8);
            let w1: Vec<bool> = rng.random_iter().take(8).collect();
            let w2: Vec<bool> = rng.random_iter().take(8).collect();
            println!("tableau: {:?}", tableau);
            assert_eq!(
                tableau.stabilizer_matrix_entry(i, row, w1.iter().copied(), w2.iter().copied()),
                tableau.stabilizer_matrix_entry_reference(
                    i,
                    row,
                    w1.iter().copied(),
                    w2.iter().copied()
                )
            );
        }
    }
}
