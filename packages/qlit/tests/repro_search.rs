//! Temporary: search for a minimal circuit that breaks T-branch deduplication.
use std::panic::{AssertUnwindSafe, catch_unwind};

use num_complex::Complex;
use qlit::{
    BitString, CliffordTCircuit,
    CliffordTGate::{self, *},
    simulate_circuit,
};

fn dense(n: u32, gates: &[CliffordTGate]) -> Vec<Complex<f64>> {
    let n = n as usize;
    let mut s = vec![Complex::ZERO; 1 << n];
    s[0] = Complex::ONE;
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let t = Complex::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let phase = |s: &mut Vec<Complex<f64>>, a: u32, p: Complex<f64>| {
        for (k, v) in s.iter_mut().enumerate() {
            if k >> a & 1 == 1 {
                *v *= p;
            }
        }
    };
    for &g in gates {
        match g {
            X(a) => {
                let s2 = s.clone();
                for k in 0..s.len() {
                    s[k] = s2[k ^ (1 << a)];
                }
            }
            Y(a) => {
                let s2 = s.clone();
                for k in 0..s.len() {
                    let src = s2[k ^ (1 << a)];
                    s[k] = if k >> a & 1 == 1 {
                        src * Complex::I
                    } else {
                        src * -Complex::I
                    };
                }
            }
            Z(a) => phase(&mut s, a, -Complex::ONE),
            S(a) => phase(&mut s, a, Complex::I),
            Sdg(a) => phase(&mut s, a, -Complex::I),
            T(a) => phase(&mut s, a, t),
            Tdg(a) => phase(&mut s, a, t.conj()),
            H(a) => {
                let s2 = s.clone();
                for k in 0..s.len() {
                    let k0 = k & !(1 << a);
                    let k1 = k | (1 << a);
                    s[k] = if k >> a & 1 == 0 {
                        (s2[k0] + s2[k1]) * h
                    } else {
                        (s2[k0] - s2[k1]) * h
                    };
                }
            }
            Cnot(a, b) => {
                let s2 = s.clone();
                for k in 0..s.len() {
                    s[k] = if k >> a & 1 == 1 {
                        s2[k ^ (1 << b)]
                    } else {
                        s2[k]
                    };
                }
            }
            Cz(a, b) => {
                for (k, v) in s.iter_mut().enumerate() {
                    if k >> a & 1 == 1 && k >> b & 1 == 1 {
                        *v = -*v;
                    }
                }
            }
        }
    }
    s
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Ok,
    Panic,
    Wrong,
}

fn check(n: u32, gates: &[CliffordTGate]) -> Outcome {
    let circuit = CliffordTCircuit::new(n, gates.iter().copied()).unwrap();
    let expected = dense(n, gates);
    let r = catch_unwind(AssertUnwindSafe(|| {
        (0..1usize << n)
            .map(|k| {
                let w = (0..n).map(|q| k >> q & 1 == 1).collect();
                simulate_circuit(&w, &circuit)
            })
            .collect::<Vec<_>>()
    }));
    match r {
        Err(_) => Outcome::Panic,
        Ok(got)
            if got
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).norm() < 1e-9) =>
        {
            Outcome::Ok
        }
        Ok(_) => Outcome::Wrong,
    }
}

fn shrink(n: u32, mut gates: Vec<CliffordTGate>, kind: &Outcome) -> Vec<CliffordTGate> {
    loop {
        let mut progress = false;
        for i in 0..gates.len() {
            let mut g2 = gates.clone();
            g2.remove(i);
            if check(n, &g2) == *kind {
                gates = g2;
                progress = true;
                break;
            }
        }
        if !progress {
            return gates;
        }
    }
}

#[test]
#[ignore]
fn search() {
    std::panic::set_hook(Box::new(|_| {}));
    for kind in [Outcome::Panic, Outcome::Wrong] {
        let mut best: Option<(u32, Vec<CliffordTGate>)> = None;
        'outer: for n in 2..=4u32 {
            for len in 4..=24 {
                for t in 1..=5.min(len) {
                    for seed in 0..300 {
                        let c = CliffordTCircuit::random(n, len, t, seed);
                        if check(n, c.gates()) == kind {
                            let g = shrink(n, c.gates().to_vec(), &kind);
                            if best.as_ref().is_none_or(|(_, b)| g.len() < b.len()) {
                                best = Some((n, g));
                            }
                            if best.as_ref().unwrap().1.len() <= 7 {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        let _ = std::panic::take_hook();
        println!("{kind:?}: {best:?}");
        std::panic::set_hook(Box::new(|_| {}));
    }
}

#[test]
#[ignore]
fn remaining() {
    let gates = [H(0), T(0), H(1), H(0), Cnot(1, 0), H(1), Tdg(1)];
    let circuit = CliffordTCircuit::new(2, gates).unwrap();
    let expected = dense(2, &gates);
    println!("expected {expected:?}");
    for k in 0..4usize {
        let w = (0..2).map(|q| k >> q & 1 == 1).collect();
        println!("{k}: {}", simulate_circuit(&w, &circuit));
    }
}

#[test]
#[ignore]
fn known() {
    std::panic::set_hook(Box::new(|_| {}));
    for gates in [
        vec![H(0), H(1), Tdg(0), Tdg(1), Cnot(0, 1), H(0), Tdg(0)],
        vec![Cnot(0, 1), H(0), Tdg(0), H(1), Tdg(1), Tdg(0)],
        vec![H(0), T(0), H(1), H(0), Cnot(1, 0), H(1), Tdg(1)],
    ] {
        println!("{:?}: {gates:?}", check(2, &gates));
    }
}

#[test]
#[ignore]
fn bench_circuit() {
    let circuit = CliffordTCircuit::random(8, 256, 10, 123);
    simulate_circuit(&BitString::zero(8), &circuit);
}

#[test]
#[ignore]
fn variant() {
    let gates = [H(0), T(0), H(1), H(0), Cnot(1, 0), H(1), Z(0), Tdg(1)];
    println!("expected {:?}", dense(2, &gates));
    std::panic::set_hook(Box::new(|_| {}));
    println!("{:?}", check(2, &gates));
}
