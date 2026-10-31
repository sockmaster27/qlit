use std::fmt::Debug;
use std::mem;

use crate::utils::{bitmask, msb_index};

type BitBlock = u64;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

/// Result returned by [`PlanningTableau::try_fork_apply_z_gate`], indicating whether fork operation was carried out.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum ForkResult {
    Forked,
    /// The tableau was not forked as it would introduce duplicate states.
    /// The contained value, `c`, indicates that applying the given gate to the
    /// state at index `i` would result in a state identical to the one
    /// currently contained in index `i ^ c`, (`i` bitwise XOR'ed by `c`).
    Redundant(usize),
}

/// The fragment of the state of [`crate::tableau::ExtendedTableau`] needed for the planning phase.
#[derive(Clone)]
pub struct PlanningTableau {
    /// The number of qubits in the tableau.
    n: usize,
    /// The current number of active c-columns in the tableau.
    c_cols: usize,
    /// The tableau is laid out in the following way:
    /// ```text
    /// P1 -> x1 z1 x2 z2 ... xn zn | c1 c2 ... ct
    /// P2 -> x1 z1 x2 z2 ... xn zn | c1 c2 ... ct
    /// ...
    /// Pn -> x1 z1 x2 z2 ... xn zn | c1 c2 ... ct
    /// ```
    tableau: Vec<BitBlock>,
    c_col_pivots: Vec<usize>,
    c_col_tags: Vec<usize>,
}
impl PlanningTableau {
    /// Initialize a new tableau with `n` qubits in the initial zero state.
    /// This allocates an extended tableau with capacity for representing a total of 2^`capacity_log2` states.
    pub fn zero(n: usize, capacity_log2: usize) -> Self {
        let mut tableau = vec![0; tableau_block_length(n, capacity_log2)];
        for i in 0..n {
            let block_index = z_column_block_index(n, i / BLOCK_SIZE, i);
            tableau[block_index] = bitmask(i % BLOCK_SIZE);
        }
        PlanningTableau {
            n,
            c_cols: 0,
            tableau,
            c_col_pivots: vec![0; capacity_log2],
            c_col_tags: vec![0; capacity_log2],
        }
    }

    pub fn apply_s_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
            self.tableau[z] ^= self.tableau[x];
        }
    }
    pub fn apply_sdg_gate(&mut self, a: usize) {
        // Since we're not tracking sign bits, these are identical
        self.apply_s_gate(a)
    }
    pub fn apply_h_gate(&mut self, a: usize) {
        let n = self.n;
        for i in 0..column_block_length(n) {
            let x = x_column_block_index(n, i, a);
            let z = z_column_block_index(n, i, a);
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
            self.tableau[za] ^= self.tableau[xb];
            self.tableau[zb] ^= self.tableau[xa];
        }
    }
    pub fn apply_x_gate(&mut self, _a: usize) {
        // Only affects signs, not tracked here
    }
    pub fn apply_y_gate(&mut self, _a: usize) {
        // Only affects signs, not tracked here
    }
    pub fn apply_z_gate(&mut self, _a: usize) {
        // Only affects signs, not tracked here
    }

    /// Attempts to fork the extended tableau such that it represents the current sequence of states,
    /// appended by the same sequence but with the Z-gate applied to qubit `a` of each state,
    /// doubling the number of states represented by the tableau.
    ///
    /// If this forking would introduce duplicate states, the tableau is not mutated.
    ///
    /// Returns a [`ForkResult`] indicating whether the fork was carried out or found to be redundant,
    /// in which case the information about which states would be duplicates is included.
    pub fn try_fork_apply_z_gate(&mut self, a: usize) -> ForkResult {
        let n = self.n;
        let c_cols = self.c_cols;
        let new_c_col = c_cols;
        let x = x_column_block_index(n, 0, a);
        let c = c_column_block_index(n, 0, new_c_col);

        self.tableau.copy_within(x..(x + column_block_length(n)), c);

        for j in 0..c_cols {
            let pivot = self.c_col_pivots[j];
            let block_index = pivot / BLOCK_SIZE;
            let bit_index = pivot % BLOCK_SIZE;
            let bit_mask: BitBlock = bitmask(bit_index);
            let bit_set =
                self.tableau[c_column_block_index(n, block_index, new_c_col)] & bit_mask != 0;
            if bit_set {
                for i in 0..column_block_length(n) {
                    self.tableau[c_column_block_index(n, i, new_c_col)] ^=
                        self.tableau[c_column_block_index(n, i, j)];
                }
                self.c_col_tags[new_c_col] ^= self.c_col_tags[j];
            }
        }

        let mut pivot = None;
        for block_index in 0..column_block_length(n) {
            let block = self.tableau[c_column_block_index(n, block_index, new_c_col)];
            if block != 0 {
                pivot = Some(block_index * BLOCK_SIZE + msb_index(block));
                break;
            }
        }

        if let Some(pivot) = pivot {
            self.c_col_pivots[new_c_col] = pivot;
            self.c_col_tags[new_c_col] |= 1 << c_cols;
            self.c_cols += 1;
            ForkResult::Forked
        } else {
            // No pivot found, i.e. the column is all-zero, meaning that
            // it was exactly equal to the combination of the columns indicated by c_col_tags
            let c = self.c_col_tags[new_c_col];
            self.c_col_tags[new_c_col] = 0;
            ForkResult::Redundant(c)
        }
    }
}

/// Get the index of the i'th block of the j'th column.
#[inline]
fn column_block_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < column_block_length(n));
    j * column_block_length(n) + i
}
/// Get the index of the i'th block of the column representing the x part of the q'th tensor element.
#[inline]
fn x_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q)
}
/// Get the index of the i'th block of the column representing the z part of the q'th tensor element.
#[inline]
fn z_column_block_index(n: usize, i: usize, q: usize) -> usize {
    debug_assert!(q < n);
    column_block_index(n, i, 2 * q + 1)
}
/// Get the index of the i'th block of the j'th c column.
#[inline]
fn c_column_block_index(n: usize, i: usize, j: usize) -> usize {
    column_block_index(n, i, n + n + j)
}

/// Get the block-length of the columns in the tableau.
#[inline]
fn column_block_length(n: usize) -> usize {
    n.div_ceil(BLOCK_SIZE)
}
/// Get the block-length of the tableau.
#[inline]
fn tableau_block_length(n: usize, c_cols: usize) -> usize {
    column_block_length(n) * (n + n + c_cols)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn try_fork_apply_z_gate_redundant() {
        let mut g = PlanningTableau::zero(100, 3);
        // Zero x column is trivially derivable.
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Redundant(0b0));

        g.apply_h_gate(0);
        g.apply_h_gate(70);
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Forked);
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Redundant(0b1));
        assert_eq!(g.try_fork_apply_z_gate(70), ForkResult::Forked);

        // x column of qubit 70 becomes the XOR of the two existing c columns.
        g.apply_cnot_gate(0, 70);
        let before = g.tableau.clone();
        assert_eq!(g.try_fork_apply_z_gate(70), ForkResult::Redundant(0b11));
        assert_eq!(g.tableau, before);

        g.apply_h_gate(99);
        assert_eq!(g.try_fork_apply_z_gate(99), ForkResult::Forked);
    }

    #[test]
    fn try_fork_apply_z_gate_redundant_reduced_basis() {
        let mut g = PlanningTableau::zero(2, 3);
        g.apply_h_gate(0);
        // c0 = [1, 0]
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Forked);
        g.apply_h_gate(1);
        g.apply_cnot_gate(1, 0);
        // c1 = [1, 1], which must be reduced by c0 when building the basis.
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Forked);
        g.apply_cnot_gate(1, 0);
        // x column of qubit 0 is [1, 0] = c0.
        assert_eq!(g.try_fork_apply_z_gate(0), ForkResult::Redundant(0b01));
    }
}
