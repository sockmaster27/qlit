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
    let mut deleted: Vec<usize> = order.into_iter().take(to_delete).collect();

    for &ordinal in &deleted {
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
    // Remove from the most significant bit down so lower positions stay valid.
    deleted.sort_unstable_by(|a, b| b.cmp(a));
    for gate in &mut gates {
        if let PizDeduplicate(c, _, _, _) = gate {
            for &i in &deleted {
                debug_assert_eq!(*c >> i & 1, 0);
                *c = remove_bit(*c, i);
            }
        }
    }

    CircuitPlanned {
        qubits: circuit.qubits(),
        gates,
        sequential_gates,
        fork_gates,
    }
}

/// Remove the i'th least significant bit of the given word, shifting the more significant bits down by one to fill the gap.
/// ```text
/// remove_bit(11101001, 3) -> 01110001
///                ^
/// ```
#[inline]
fn remove_bit(word: usize, i: usize) -> usize {
    let low_mask = (1 << i) - 1;
    let high_mask = !(low_mask | 1 << i);
    ((high_mask & word) >> 1) | (low_mask & word)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Circuit, Gate::*, decompose::decompose};

    #[test]
    fn remove_bit_shifts_higher_bits_down() {
        assert_eq!(remove_bit(0b11101001, 3), 0b01110001);
        assert_eq!(remove_bit(0b1, 0), 0b0);
        assert_eq!(remove_bit(0b10, 0), 0b1);
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
