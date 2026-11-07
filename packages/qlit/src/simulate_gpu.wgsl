const BLOCK_SIZE: u32 = 32;
alias BitBlock = u32;

@group(0) @binding(0) var<uniform> n: u32;
// The current number of active c columns in the tableau.
@group(0) @binding(1) var<uniform> c_cols: u32;
@group(0) @binding(2) var<storage, read_write> ws: array<BitBlock>;
@group(0) @binding(3) var<storage, read_write> w_coeffs: array<Complex>;


@group(1) @binding(0) var<storage, read> gates: array<u32>;
@group(1) @binding(1) var<storage, read> qubit_params: array<u32>;
@group(1) @binding(2) var<uniform> initial_path_index: u32;
@group(1) @binding(3) var<storage, read> path: array<u32>;
@group(1) @binding(4) var<storage, read> ci_params: array<Complex>;
@group(1) @binding(5) var<storage, read> cz_params: array<Complex>;

@compute
@workgroup_size(64)
fn apply_gates(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each contained state.
    // The corresponding updates to the tableau are applied on the CPU.
    let state_index = id.x;
    if state_index >= contained_states() {
        return;
    }

    // Only advance p at the end of each case, after all uses of the parameters it points to.
    // On the macos-15-intel CI runners, Metal otherwise ends up reading a parameter after the increment
    // when a branch using it diverges between threads, making a gate act on the next gate's qubit.
    var p: u32 = 0;
    var path_index = initial_path_index;
    for (var i: u32 = 0; i < arrayLength(&gates); i += 1) {
        switch gates[i] {
            case 0: {
                // CNOT
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                if w_bit(state_index, a) {
                    flip_w_bit(state_index, b);
                }
                p += 2;
            }
            case 1: {
                // CZ
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                if w_bit(state_index, a) && w_bit(state_index, b) {
                    w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                }
                p += 2;
            }
            case 2: {
                // X
                let a = qubit_params[p];
                flip_w_bit(state_index, a);
                p += 1;
            }
            case 3: {
                // Y
                let a = qubit_params[p];
                if w_bit(state_index, a) {
                    w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                } else {
                    w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                }
                flip_w_bit(state_index, a);
                p += 1;
            }
            case 4: {
                // Z
                let a = qubit_params[p];
                if w_bit(state_index, a) {
                    w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                }
                p += 1;
            }
            case 5: {
                // S
                let a = qubit_params[p];
                if w_bit(state_index, a) {
                    w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                }
                p += 1;
            }
            case 6: {
                // Sdg
                let a = qubit_params[p];
                if w_bit(state_index, a) {
                    w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                }
                p += 1;
            }
            case 7: {
                // Piz branch
                let a = qubit_params[p];
                if path[path_index] == 0u {
                    w_coeffs[state_index] = mul(w_coeffs[state_index], ci_params[path_index]);
                } else {
                    if w_bit(state_index, a) {
                        w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                    }
                    w_coeffs[state_index] = mul(w_coeffs[state_index], cz_params[path_index]);
                }
                path_index += 1;
                p += 1;
            }
            default: {}
        }
    }
}


// Shared between the piz gate compute shaders.
@group(1) @binding(0) var<uniform> qubit: u32;
@group(1) @binding(1) var<uniform> ci: Complex;
@group(1) @binding(2) var<uniform> cz: Complex;

@compute
@workgroup_size(64)
fn fork_apply_piz_gate(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each contained state.
    // The corresponding fork of the tableau is performed on the CPU.
    if id.x >= contained_states() {
        return;
    }
    let index_i = id.x;
    let index_z = id.x + contained_states();

    for (var b = 0u; b < w_block_size(); b += 1) {
        ws[index_z * w_block_size() + b] = ws[index_i * w_block_size() + b];
    }
    w_coeffs[index_z] = w_coeffs[index_i];

    w_coeffs[index_i] = mul(w_coeffs[index_i], ci);

    if w_bit(index_z, qubit) {
        w_coeffs[index_z] = mul_minus_one(w_coeffs[index_z]);
    }
    w_coeffs[index_z] = mul(w_coeffs[index_z], cz);
}

@compute
@workgroup_size(64)
fn dedup_apply_piz_gate_c_zero(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let state_index = id.x;
    if state_index >= contained_states() {
        return;
    }

    var z_coeff = w_coeffs[state_index];
    if w_bit(state_index, qubit) {
        z_coeff = mul_minus_one(z_coeff);
    }
    w_coeffs[state_index] = add(mul(ci, w_coeffs[state_index]), mul(cz, z_coeff));
}

@group(1) @binding(3) var<uniform> c: u32;
// Shared between dedup_apply_piz_gate and compute_output.
// The extended stabilizer tableau, laid out column-wise with interleaved x and z columns in the following way:
// P1 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// P2 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// ...
// Pn -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// See ExtendedTableau in tableau.rs.
// All mutation of the tableau happens on the CPU, which uploads it in RREF whenever it is needed here.
@group(1) @binding(4) var<storage, read> tableau: array<BitBlock>;
// Shared between dedup_apply_piz_gate and compute_output.
// Pivot bookkeeping established by ExtendedTableau::bring_into_rref, uploaded alongside the tableau.
// For each row, holds the column of its pivot plus one (0 if the row has no pivot).
@group(1) @binding(5) var<storage, read> pivots: array<u32>;

@compute
@workgroup_size(64)
fn dedup_apply_piz_gate(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let m = id.x;
    if m >= contained_states() / 2 {
        return;
    }

    let i = xor_pair(m, c);
    w_coeffs[i.y] = mul(w_coeffs[i.y], coeff_ratio(i.y, i.x, i.y));
    for (var q: u32 = 0; q < n; q += 1) {
        ws[w_bit_index(i.y, q)] = ws[w_bit_index(i.x, q)];
    }

    var z_coeff1 = w_coeffs[i.y];
    var z_coeff2 = w_coeffs[i.x];
    if w_bit(i.y, qubit) {
        z_coeff1 = mul_minus_one(z_coeff1);
    }
    if w_bit(i.x, qubit) {
        z_coeff2 = mul_minus_one(z_coeff2);
    }
    
    w_coeffs[i.x] = add(mul(ci, w_coeffs[i.x]), mul(cz, z_coeff1));
    w_coeffs[i.y] = add(mul(ci, w_coeffs[i.y]), mul(cz, z_coeff2));
}

fn xor_pair(m: u32, c: u32) -> vec2<u32> {
    let low_mask = (1u << countTrailingZeros(c)) - 1;
    let high_bits = (m & ~low_mask) << 1;
    let low_bits = m & low_mask;
    let i = high_bits | low_bits;
    return vec2(i, i ^ c);
}


// What update_before_h needs from the single relevant row of the (RREF) tableau, extracted on the CPU.
struct HRow {
    // The qubit that the H gate is about to be applied to.
    qubit: u32,
    // The part of the coefficient ratio between any w and w with the qubit'th bit flipped
    // that is shared by all states and basis states.
    // If this is zero, all the ratios are zero and the remaining fields are unspecified.
    ratio: Complex,
    // Bitmask where bit j holds the j'th c bit of the row.
    c_mask: u32,
    // The z bits of the row, packed in the same way as each w in ws.
    z: array<BitBlock>,
}
@group(1) @binding(0) var<storage, read> h_row: HRow;

@compute
@workgroup_size(64)
fn update_before_h(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each contained state.
    let state_index = id.x;
    if state_index >= contained_states() {
        return;
    }

    let r = coeff_ratio_flipped_bit(state_index);
    if !eq(r, Complex(-1.0, 0.0)) {
        w_coeffs[state_index] = mul(w_coeffs[state_index], div_real(add_real(r, 1.0), sqrt(2.0)));
        unset_w_bit(state_index, h_row.qubit);
    } else {
        if w_bit(state_index, h_row.qubit) {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], -2.0 / sqrt(2.0));
        } else {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], 2.0 / sqrt(2.0));
        }
        set_w_bit(state_index, h_row.qubit);
    }
}

// Same as coeff_ratio but with w1 and s both being state_index and w2 being w1 but with the h_row.qubit'th bit flipped.
// Only needs the single row described by h_row.
fn coeff_ratio_flipped_bit(state_index: u32) -> Complex {
    // Respect the sign of the state_index'th state.
    var negations = countOneBits(state_index & h_row.c_mask);
    // Every Z or Y element at a position where w1 is 1 contributes a factor of -1.
    let state_offset = w_block_size() * state_index;
    for (var b: u32 = 0; b < w_block_size(); b += 1) {
        negations += countOneBits(ws[state_offset + b] & h_row.z[b]);
    }

    if (negations % 2u) != 0u {
        return mul_minus_one(h_row.ratio);
    }
    return h_row.ratio;
}


@group(1) @binding(0) var<storage, read> w_target: array<u32>;
@group(1) @binding(1) var<storage, read_write> output: array<Complex>;

@compute
@workgroup_size(64)
fn compute_output(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each contained state.
    let state_index = id.x;
    if state_index >= contained_states() {
        return;
    }

    let r = coeff_ratio_target(state_index);
    output[state_index] = mul(w_coeffs[state_index], r);
}


// Assume tableau in RREF.
// Get the ratio between the basis state ws[w1] and ws[w2].
// Respects the sign of the s'th state.
fn coeff_ratio(w1: u32, w2: u32, s: u32) -> Complex {
    // We encode the result as i^phase, and let the phase wrap around modulo 4.
    var phase: u32 = 0;

    // Determine the sign of the derived stabilizer.
    for (var row: u32 = 0; row < n; row += 1) {
        let pivot_plus1 = pivots[row];
        if pivot_plus1 != 0 {
            let q = pivot_plus1 - 1;
            if w_bit(w1, q) != w_bit(w2, q) && row_negative(s, row) {
                phase += 2;
            }
        }
    }

    for (var q2: u32 = 0; q2 < n; q2 += 1) {
        // Derive the q2'th tensor element of the stabilizer,
        // including the phase change caused by multiplication of the individual Pauli matrices.
        var x_acc = false;
        var z_acc = false;
        for (var row2: u32 = 0; row2 < n; row2 += 1) {
            let pivot_plus1 = pivots[row2];
            if pivot_plus1 != 0 {
                let q = pivot_plus1 - 1;
                if w_bit(w1, q) != w_bit(w2, q) {
                    let x_row = x_bit(row2, q2);
                    let z_row = z_bit(row2, q2);
                    phase += product_phase(x_acc, z_acc, x_row, z_row);
                    x_acc = x_acc != x_row;
                    z_acc = z_acc != z_row;
                }
            }
        }

        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        let differs = w_bit(w1, q2) != w_bit(w2, q2);
        if x_acc != differs {
            // Multiply by 0
            return Complex(0.0, 0.0);
        }
        if x_acc && z_acc {
            // Y
            phase += 1;
        }
        if z_acc && w_bit(w1, q2) {
            // Z or Y
            phase += 2;
        }
    }
    return i_pow(phase);
}

// Assume tableau in RREF.
// Same as coeff_ratio but with w1 and s both being state_index and w2 being ws_target.
fn coeff_ratio_target(state_index: u32) -> Complex {
    // We encode the result as i^phase, and let the phase wrap around modulo 4.
    var phase: u32 = 0;

    // Determine the sign of the derived stabilizer.
    for (var row: u32 = 0; row < n; row += 1) {
        let pivot_plus1 = pivots[row];
        if pivot_plus1 != 0 {
            let q = pivot_plus1 - 1;
            if w_bit(state_index, q) != (w_target[q] == 1) && row_negative(state_index, row) {
                phase += 2;
            }
        }
    }

    for (var q2: u32 = 0; q2 < n; q2 += 1) {
        // Derive the q2'th tensor element of the stabilizer,
        // including the phase change caused by multiplication of the individual Pauli matrices.
        var x_acc = false;
        var z_acc = false;
        for (var row2: u32 = 0; row2 < n; row2 += 1) {
            let pivot_plus1 = pivots[row2];
            if pivot_plus1 != 0 {
                let q = pivot_plus1 - 1;
                if w_bit(state_index, q) != (w_target[q] == 1) {
                    let x_row = x_bit(row2, q2);
                    let z_row = z_bit(row2, q2);
                    phase += product_phase(x_acc, z_acc, x_row, z_row);
                    x_acc = x_acc != x_row;
                    z_acc = z_acc != z_row;
                }
            }
        }

        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        let differs = w_bit(state_index, q2) != (w_target[q2] == 1);
        if x_acc != differs {
            // Multiply by 0
            return Complex(0.0, 0.0);
        }
        if x_acc && z_acc {
            // Y
            phase += 1;
        }
        if z_acc && w_bit(state_index, q2) {
            // Z or Y
            phase += 2;
        }
    }
    return i_pow(phase);
}

// Get the exponent of the phase, i^phase, resulting from multiplying the Pauli matrices (x1, z1) and (x2, z2) in that order.
fn product_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> u32 {
    let x_1 = x1 && !z1;
    let y_1 = x1 && z1;
    let z_1 = !x1 && z1;
    let x_2 = x2 && !z2;
    let y_2 = x2 && z2;
    let z_2 = !x2 && z2;
    // XY = +iZ
    // YZ = +iX
    // ZX = +iY
    if (x_1 && y_2) || (y_1 && z_2) || (z_1 && x_2) {
        return 1u;
    }
    // YX = -iZ
    // ZY = -iX
    // XZ = -iY
    if (y_1 && x_2) || (z_1 && y_2) || (x_1 && z_2) {
        return 3u;
    }
    return 0u;
}


// Get whether the given row is negative or not, i.e. the contents of the sign bit.
//
// This will respect the sign of the s'th state.
fn row_negative(s: u32, row: u32) -> bool {
    let row_block_index = row / BLOCK_SIZE;
    let row_bit_index = row % BLOCK_SIZE;
    let row_bitmask = bitmask(row_bit_index);
    var r = tableau[r_column_block_index(row_block_index)];
    for (var j: u32 = 0; j < c_cols; j += 1) {
        if (s & (1u << j)) != 0u {
            r ^= tableau[c_column_block_index(row_block_index, j)];
        }
    }
    return (r & row_bitmask) != 0u;
}

// Get the index of the i'th block of the j'th column.
fn column_block_index(i: u32, j: u32) -> u32 {
    return j * column_block_length() + i;
}
// Get the index of the i'th block of the column representing the x part of the q'th tensor element.
fn x_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q);
}
// Get the index of the i'th block of the column representing the z part of the q'th tensor element.
fn z_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q + 1);
}
// Get the index of the i'th block of the r column.
fn r_column_block_index(i: u32) -> u32 {
    return column_block_index(i, n + n);
}
// Get the index of the i'th block of the j'th c column.
fn c_column_block_index(i: u32, j: u32) -> u32 {
    return column_block_index(i, n + n + 1 + j);
}

// Get the block-length of the columns in the tableau.
fn column_block_length() -> u32 {
    return div_ceil(n, BLOCK_SIZE);
}

// Get the number of states currently represented by the tableau.
fn contained_states() -> u32 {
    return 1u << c_cols;
}

// Get the value of the bit corresponding to the j'th column in the `row`th row.
fn bit(row: u32, j: u32) -> bool {
    let row_block_index = row / BLOCK_SIZE;
    let row_bit_index = row % BLOCK_SIZE;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[column_block_index(row_block_index, j)] & row_bitmask) != 0u;
}
// Get the value of the x bit corresponding to the q'th tensor element in the `row`th row.
fn x_bit(row: u32, q: u32) -> bool {
    return bit(row, 2 * q);
}
// Get the value of the z bit corresponding to the q'th tensor element in the `row`th row.
fn z_bit(row: u32, q: u32) -> bool {
    return bit(row, 2 * q + 1);
}

fn w_block_size() -> u32 {
    return div_ceil(n, BLOCK_SIZE);
}
fn w_bit(state_index: u32, q: u32) -> bool {
    let state_offset = w_block_size() * state_index;
    let block_index = state_offset + (q / BLOCK_SIZE);
    let bit_index = q % BLOCK_SIZE;
    return (ws[block_index] & bitmask(bit_index)) != 0;
}
fn flip_w_bit(state_index: u32, q: u32) {
    let state_offset = w_block_size() * state_index;
    let block_index = state_offset + (q / BLOCK_SIZE);
    let bit_index = q % BLOCK_SIZE;
    ws[block_index] ^= bitmask(bit_index);
}
fn set_w_bit(state_index: u32, q: u32) {
    let state_offset = w_block_size() * state_index;
    let block_index = state_offset + (q / BLOCK_SIZE);
    let bit_index = q % BLOCK_SIZE;
    ws[block_index] |= bitmask(bit_index);
}
fn unset_w_bit(state_index: u32, q: u32) {
    let state_offset = w_block_size() * state_index;
    let block_index = state_offset + (q / BLOCK_SIZE);
    let bit_index = q % BLOCK_SIZE;
    ws[block_index] &= ~bitmask(bit_index);
}

// Get the bitmask for the i'th bit, e.g.
// bitmask(0) -> 10000000
// bitmask(1) -> 01000000
// bitmask(6) -> 00000010
fn bitmask(i: u32) -> BitBlock {
    return 1u << (BLOCK_SIZE - 1u - i);
}
// Divide a by b and round up.
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}


struct Complex {
    re: f32,
    im: f32,
}
fn eq(c1: Complex, c2: Complex) -> bool {
    return c1.re == c2.re && c1.im == c2.im;
}
fn add(c1: Complex, c2: Complex) -> Complex {
    return Complex(c1.re + c2.re, c1.im + c2.im);
}
fn add_real(c1: Complex, f2: f32) -> Complex {
    return Complex(c1.re + f2, c1.im);
}
fn div_real(c1: Complex, f2: f32) -> Complex {
    return Complex(c1.re / f2, c1.im / f2);
}
fn mul(c1: Complex, c2: Complex) -> Complex {
    return Complex(
        c1.re * c2.re - c1.im * c2.im,
        c1.re * c2.im + c1.im * c2.re,
    );
}
fn mul_real(c1: Complex, f2: f32) -> Complex {
    return Complex(
        c1.re * f2,
        c1.im * f2,
    );
}
fn mul_minus_one(c: Complex) -> Complex {
    return Complex(-c.re, -c.im);
}
fn mul_i(c: Complex) -> Complex {
    return Complex(-c.im, c.re);
}
fn mul_minus_i(c: Complex) -> Complex {
    return Complex(c.im, -c.re);
}
// Get i^phase.
fn i_pow(phase: u32) -> Complex {
    switch phase % 4 {
        case 0: {
            return Complex(1.0, 0.0);
        }
        case 1: {
            return Complex(0.0, 1.0);
        }
        case 2: {
            return Complex(-1.0, 0.0);
        }
        default: {
            return Complex(0.0, -1.0);
        }
    }
}
