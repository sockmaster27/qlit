use std::cmp::{max, min};

use num_complex::Complex;

use crate::{
    decompose::{CircuitDecomposed, GateDecomposed},
    tableau_plan::{ForkResult, PlanningTableau},
    utils::{INDEX_TOO_LARGE, N_TOO_LARGE},
};

/// TODO: Find an optimal value for this.
const MAX_BATCH_SIZE_LOG2: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GatePlanned {
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    Sdg(u32),
    H(u32),
    Cnot(u32, u32),
    Cz(u32, u32),

    /// Piz gate where only one branch is applied to the tableau for the given path.
    PizSequential(Complex<f64>, Complex<f64>, u32),
    /// Piz gate applied by forking the extended tableau, so both branches are kept in memory simultaniously.
    PizFork(Complex<f64>, Complex<f64>, u32),
    /// Piz gate where forking would create duplicate states, so they can be merged.
    /// PizDeduplicate(c, ci, cz, a) indicates that for the index of any state, i,
    /// applying the Z gate to this state creates a tableau identical to the one currently at index i ^ c (i bitwise XOR c).
    PizDeduplicate(usize, Complex<f64>, Complex<f64>, u32),
}
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitPlanned {
    qubits: u32,
    gates: Vec<GatePlanned>,
    sequential_gates: usize,
    fork_gates: usize,
}
impl CircuitPlanned {
    pub fn qubits(&self) -> u32 {
        self.qubits
    }

    pub fn gates(&self) -> &[GatePlanned] {
        &self.gates
    }

    pub fn sequential_gates(&self) -> usize {
        self.sequential_gates
    }

    pub fn fork_gates(&self) -> usize {
        self.fork_gates
    }
}

/// Plan which strategy should be used to execute each non-Clifford gate in the given circuit.
///
/// - `min_sequential_gates`: Soft hint for minimum number of sequential gates to be included in the circuit.
///     Useful for taking advantage of available parallelism.
pub fn plan(circuit: &CircuitDecomposed, min_sequential_gates: usize) -> CircuitPlanned {
    use GatePlanned::*;

    let n: usize = circuit.qubits().try_into().expect(N_TOO_LARGE);
    let mut g = PlanningTableau::zero(n, circuit.p_gates());
    // For each successful fork, in order of occurrence (i.e. indexed by the bit positions in `c`),
    // the index of the fork gate and the indices of the PizDeduplicate gates depending on it.
    let mut dedup_dependencies: Vec<(usize, Vec<usize>)> = vec![];
    let mut gates: Vec<GatePlanned> = circuit
        .gates()
        .iter()
        .copied()
        .enumerate()
        .map(|(i, gate)| match gate {
            GateDecomposed::X(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_x_gate(au);
                X(a)
            }
            GateDecomposed::Y(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_y_gate(au);
                Y(a)
            }
            GateDecomposed::Z(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_z_gate(au);
                Z(a)
            }
            GateDecomposed::S(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_s_gate(au);
                S(a)
            }
            GateDecomposed::Sdg(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_sdg_gate(au);
                Sdg(a)
            }
            GateDecomposed::H(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_h_gate(au);
                H(a)
            }
            GateDecomposed::Cnot(a, b) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let bu: usize = b.try_into().expect(INDEX_TOO_LARGE);
                g.apply_cnot_gate(au, bu);
                Cnot(a, b)
            }
            GateDecomposed::Cz(a, b) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let bu: usize = b.try_into().expect(INDEX_TOO_LARGE);
                g.apply_cz_gate(au, bu);
                Cz(a, b)
            }

            GateDecomposed::Piz(ci, cz, a) => {
                // Start by aggressively forking, we later determine which gates should be
                // converted to PizSequential to fit inside memory budget while maximizing number of surviving PizDeduplicate gates.
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                match g.try_fork_apply_z_gate(au) {
                    ForkResult::Forked => {
                        dedup_dependencies.push((i, vec![]));
                        PizFork(ci, cz, a)
                    }
                    ForkResult::Redundant(c) => {
                        // `c` is a plain integer, so bit j (from the least significant end)
                        // refers to the j'th fork.
                        let mut remaining = c;
                        while remaining != 0 {
                            let j = remaining.trailing_zeros() as usize;
                            remaining &= remaining - 1;
                            dedup_dependencies[j].1.push(i);
                        }
                        PizDeduplicate(c, ci, cz, a)
                    }
                }
            }
        })
        .collect();

    let mut sequential_gates = 0;
    let mut fork_gates = dedup_dependencies.len();

    // Naive greedy reduction.
    // Exact optimization is NP-hard.
    // Delete the forks with the fewest dependents first.
    let mut order: Vec<usize> = (0..dedup_dependencies.len()).collect();
    order.sort_by_key(|&ordinal| dedup_dependencies[ordinal].1.len());
    // Delete the least possible number of forks to satisfy MAX_BATCH_SIZE_LOG2 and min_path_len
    let to_delete = max(
        fork_gates.saturating_sub(MAX_BATCH_SIZE_LOG2),
        // min_path_len should never cause a PizDeduplicate to disappear,
        // so bound it by number of forks with no dependents.
        min(
            min_sequential_gates,
            order
                .iter()
                .copied()
                .take_while(|&ordinal| dedup_dependencies[ordinal].1.is_empty())
                .count(),
        ),
    );
    let deleted = &order[..to_delete];

    for &ordinal in deleted {
        let (fork_index, deps) = &dedup_dependencies[ordinal];
        gates[*fork_index] = match gates[*fork_index] {
            PizFork(ci, cz, a) => {
                fork_gates -= 1;
                sequential_gates += 1;
                PizSequential(ci, cz, a)
            }
            _ => unreachable!("dedup_dependencies contains index of non-fork gate"),
        };
        for &dedup_index in deps {
            gates[dedup_index] = match gates[dedup_index] {
                PizDeduplicate(_, ci, cz, a) => {
                    sequential_gates += 1;
                    PizSequential(ci, cz, a)
                }
                // Might have already been converted
                gate => gate,
            };
        }
    }

    // Adjust c's of the surviving PizDeduplicate gates.
    // None of them refer to a deleted fork, so the corresponding bits are all zero.
    // Removing the deleted bits is a bit compression by the mask of surviving forks.
    let deleted_mask = deleted.iter().fold(0, |mask, &i| mask | 1 << i);
    let compressor = BitCompressor::new(!deleted_mask);
    for gate in &mut gates {
        if let PizDeduplicate(c, _, _, _) = gate {
            debug_assert_eq!(*c & deleted_mask, 0);
            *c = compressor.compress(*c);
        }
    }

    CircuitPlanned {
        qubits: circuit.qubits(),
        gates,
        sequential_gates,
        fork_gates,
    }
}

const WORD_BITS_LOG2: usize = usize::BITS.trailing_zeros() as usize;

/// Gathers the bits of a word selected by a fixed mask into the least significant end,
/// preserving their order (equivalent to the BMI2 `pext` instruction).
/// ```text
/// mask:            00111011
/// compress(abcdefgh) -> 000cdegh
/// ```
///
/// The mask-dependent work is done once in [`BitCompressor::new`],
/// so each call to [`BitCompressor::compress`] is a fixed number of shifts (Hacker's Delight, 7-4).
struct BitCompressor {
    mask: usize,
    /// Bits to move right by 2^i in step i.
    move_masks: [usize; WORD_BITS_LOG2],
}
impl BitCompressor {
    fn new(mask: usize) -> Self {
        let mut m = mask;
        // Bits whose number of zeros to the right in `mask` is to be counted.
        let mut mk = !mask << 1;
        let mut move_masks = [0; WORD_BITS_LOG2];
        for (i, mv) in move_masks.iter_mut().enumerate() {
            // Parallel prefix XOR: bit j of mp is the parity of the number of zeros below j.
            let mut mp = mk;
            for j in 0..WORD_BITS_LOG2 {
                mp ^= mp << (1 << j);
            }
            *mv = mp & m;
            m = (m ^ *mv) | (*mv >> (1 << i));
            mk &= !mp;
        }
        BitCompressor { mask, move_masks }
    }

    #[inline]
    fn compress(&self, word: usize) -> usize {
        let mut x = word & self.mask;
        for (i, &mv) in self.move_masks.iter().enumerate() {
            let t = x & mv;
            x = (x ^ t) | (t >> (1 << i));
        }
        x
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Circuit, Gate::*, decompose::decompose};

    #[test]
    fn bit_compressor_removes_unmasked_bits() {
        assert_eq!(BitCompressor::new(!0b1000).compress(0b11101001), 0b01110001);
        assert_eq!(BitCompressor::new(!0b1).compress(0b1), 0b0);
        assert_eq!(BitCompressor::new(!0b1).compress(0b10), 0b1);
        assert_eq!(BitCompressor::new(0b00111011).compress(0b10101101), 0b10101);
        assert_eq!(BitCompressor::new(!0).compress(0b1011), 0b1011);
        assert_eq!(BitCompressor::new(0).compress(usize::MAX), 0);
        assert_eq!(BitCompressor::new(1 << (usize::BITS - 1)).compress(usize::MAX), 1);
    }

    #[test]
    fn bit_compressor_matches_naive() {
        use rand::{RngExt, SeedableRng, rngs::Xoshiro128PlusPlus};

        let naive = |word: usize, mask: usize| {
            let mut result = 0;
            let mut k = 0;
            for j in 0..usize::BITS {
                if mask >> j & 1 == 1 {
                    result |= (word >> j & 1) << k;
                    k += 1;
                }
            }
            result
        };
        let mut rng = Xoshiro128PlusPlus::seed_from_u64(0);
        for _ in 0..1000 {
            let mask = rng.random::<u64>() as usize;
            let compressor = BitCompressor::new(mask);
            for _ in 0..10 {
                let word = rng.random::<u64>() as usize;
                assert_eq!(compressor.compress(word), naive(word, mask));
            }
        }
    }

    #[test]
    fn plan_deletes_forks_with_fewest_dependents() {
        // Forks: T0 (ordinal 0), T1 (ordinal 1), T2 (ordinal 2).
        // Deduplicates: T0 again (c = 0b001) and T2 again (c = 0b100).
        let circuit = Circuit::new(3, [H(0), H(1), H(2), T(0), T(1), T(2), T(0), T(2)]).unwrap();
        let decomposed = decompose(&circuit);
        let planned = plan(&decomposed, 2);

        assert_eq!(planned.sequential_gates(), 1);
        assert_eq!(planned.fork_gates(), 2);
        let piz: Vec<_> = planned
            .gates()
            .iter()
            .filter(|g| {
                matches!(
                    g,
                    GatePlanned::PizSequential(..)
                        | GatePlanned::PizFork(..)
                        | GatePlanned::PizDeduplicate(..)
                )
            })
            .map(|g| match g {
                GatePlanned::PizSequential(..) => ("seq", 0),
                GatePlanned::PizFork(..) => ("fork", 0),
                GatePlanned::PizDeduplicate(c, ..) => ("dedup", *c),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(
            piz,
            vec![
                ("fork", 0),
                ("seq", 0),
                ("fork", 0),
                ("dedup", 0b01),
                ("dedup", 0b10)
            ]
        );
    }
}
