use num_complex::Complex;

use crate::{
    decompose::{CircuitDecomposed, GateDecomposed},
    tableau_plan::{ForkResult, PlanningTableau},
    utils::{INDEX_TOO_LARGE, N_TOO_LARGE, bit_indices},
};

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
    p_gates: usize,
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

    #[cfg(feature = "gpu")]
    pub fn p_gates(&self) -> usize {
        self.p_gates
    }

    pub fn sequential_gates(&self) -> usize {
        self.sequential_gates
    }

    pub fn fork_gates(&self) -> usize {
        self.fork_gates
    }
}

pub fn plan(circuit: &CircuitDecomposed, max_batch_size_log2: usize) -> CircuitPlanned {
    use GatePlanned::*;

    let n: usize = circuit.qubits().try_into().expect(N_TOO_LARGE);
    let mut g = PlanningTableau::zero(n, circuit.p_gates());
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
                // converted to PizSequential to fit inside memory budget while maximizing number of surviving PizDeduplicate.
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                match g.try_fork_apply_z_gate(au) {
                    ForkResult::Forked => {
                        dedup_dependencies.push((i, vec![]));
                        PizFork(ci, cz, a)
                    }
                    ForkResult::Redundant(c) => {
                        for j in bit_indices(c) {
                            let (_, deps) = &mut dedup_dependencies[j];
                            deps.push(i);
                        }
                        PizDeduplicate(c, ci, cz, a)
                    }
                }
            }
        })
        .collect();

    // Naive greedy reduction
    // Exact optimization is NP-hard
    let mut sequential_gates = 0;
    let mut fork_gates = dedup_dependencies.len();
    if max_batch_size_log2 < fork_gates {
        fork_gates = max_batch_size_log2;
        let to_delete = circuit.p_gates() - max_batch_size_log2;
        dedup_dependencies.sort_by_key(|(_, deps)| deps.len());
        for &(fork_index, ref deps) in dedup_dependencies.iter().take(to_delete) {
            gates[fork_index] = match gates[fork_index] {
                PizFork(ci, cz, a) => {
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
        // Adjust c's
        gates = gates
            .into_iter()
            .map(|gate| match gate {
                PizDeduplicate(mut c, ci, cz, a) => {
                    for &(fork_index, _) in dedup_dependencies.iter().take(to_delete) {
                        c = remove_bit(c, fork_index);
                    }
                    PizDeduplicate(c, ci, cz, a)
                }
                gate => gate,
            })
            .collect();
    }

    CircuitPlanned {
        qubits: circuit.qubits(),
        gates,
        p_gates: circuit.p_gates(),
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
