use num_complex::Complex;

use crate::{
    decompose::{CircuitDecomposed, GateDecomposed},
    tableau_plan::{ForkResult, PlanningTableau},
    utils::{INDEX_TOO_LARGE, N_TOO_LARGE},
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
}

pub fn plan(circuit: &CircuitDecomposed, batch_size_log2: usize) -> CircuitPlanned {
    use GatePlanned::*;

    let n: usize = circuit.qubits().try_into().expect(N_TOO_LARGE);
    let path_len = circuit.p_gates() - batch_size_log2;
    let mut g = PlanningTableau::zero(n, circuit.p_gates());
    let mut seen_p_gates = 0;
    let gates = circuit
        .gates()
        .iter()
        .copied()
        .map(|gate| match gate {
            GateDecomposed::X(a) => X(a),
            GateDecomposed::Y(a) => Y(a),
            GateDecomposed::Z(a) => Z(a),
            GateDecomposed::S(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_s_gate(au);
                S(a)
            }
            GateDecomposed::Sdg(a) => {
                let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                g.apply_s_gate(au);
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
                if seen_p_gates < path_len {
                    seen_p_gates += 1;
                    PizSequential(ci, cz, a)
                } else {
                    let au: usize = a.try_into().expect(INDEX_TOO_LARGE);
                    match g.try_fork_apply_z_gate(au) {
                        ForkResult::Forked => PizFork(ci, cz, a),
                        ForkResult::Redundant(c) => PizDeduplicate(c, ci, cz, a),
                    }
                }
            }
        })
        .collect();

    CircuitPlanned {
        qubits: circuit.qubits(),
        gates,
        p_gates: circuit.p_gates(),
    }
}
