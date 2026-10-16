const BLOCK_SIZE: u32 = 32;
alias BitBlock = u32;

@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<uniform> max_batches: u32;
@group(0) @binding(2) var<storage, read> path: array<u32>;
// The extended stabilizer tableau, laid out column-wise with interleaved x and z columns in the following way:
// P1 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// P2 -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// ...
// Pn -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct
// (E -> x1 z1 x2 z2 ... xn zn | r | c1 c2 ... ct)
// See ExtendedTableau in tableau.rs.
@group(0) @binding(3) var<storage, read_write> tableau: array<BitBlock>;
// The current number of active c columns in the tableau.
@group(0) @binding(4) var<storage, read_write> c_cols: u32;
@group(0) @binding(5) var<storage, read_write> ws: array<u32>;
@group(0) @binding(6) var<storage, read_write> w_coeffs: array<Complex>;

fn w_bit_index(state_index: u32, q: u32) -> u32 {
    return state_index * n + q;
}

@group(1) @binding(0) var<uniform> qubit: u32;


@compute
@workgroup_size(64)
fn zero(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index >= column_block_length() {
        return;
    }

    // Assume all buffers are zero-initialized.

    if id.x == 0 {
        w_coeffs[0] = Complex(1.0, 0.0);
    }

    for (var i: u32 = 0; i < BLOCK_SIZE; i += 1) {
        let q = block_index * BLOCK_SIZE + i;
        if q < n {
            let b = z_column_block_index(block_index, q);
            tableau[b] = bitmask(i);
        }
    }
}


@group(1) @binding(0) var<storage, read> gates: array<u32>;
@group(1) @binding(1) var<storage, read> qubit_params: array<u32>;
@group(1) @binding(2) var<storage, read> ci_params: array<Complex>;
@group(1) @binding(3) var<storage, read> cz_params: array<Complex>;
@group(1) @binding(4) var<uniform> initial_seen_p_gates: u32;

@compute
@workgroup_size(64)
fn apply_gates(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block), and one thread to each contained state.
    let block_index = id.x;
    let state_index = id.x;
    let is_block_thread = block_index < column_block_length();
    let is_state_thread = state_index < contained_states();
    if !is_block_thread && !is_state_thread {
        return;
    }

    var p: u32 = 0;
    var seen_p_gates = initial_seen_p_gates;
    for (var i: u32 = 0; i < arrayLength(&gates); i += 1) {
        switch gates[i] {
            case 0: {
                // CNOT
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let xa = x_column_block_index(block_index, a);
                    let za = z_column_block_index(block_index, a);
                    let xb = x_column_block_index(block_index, b);
                    let zb = z_column_block_index(block_index, b);
                    tableau[r] ^= tableau[xa] & tableau[zb] & ~(tableau[xb] ^ tableau[za]);
                    tableau[za] ^= tableau[zb];
                    tableau[xb] ^= tableau[xa];
                }
                if is_state_thread {
                    ws[w_bit_index(state_index, b)] ^= ws[w_bit_index(state_index, a)];
                }
            }
            case 1: {
                // CZ
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let xa = x_column_block_index(block_index, a);
                    let za = z_column_block_index(block_index, a);
                    let xb = x_column_block_index(block_index, b);
                    let zb = z_column_block_index(block_index, b);
                    tableau[r] ^= (tableau[xb] & tableau[zb])
                        ^ (tableau[xa] & tableau[xb] & ~(tableau[zb] ^ tableau[za]))
                        ^ (tableau[xb] & (tableau[zb] ^ tableau[xa]));
                    tableau[za] ^= tableau[xb];
                    tableau[zb] ^= tableau[xa];
                }
                if is_state_thread {
                    if ws[w_bit_index(state_index, a)] == 1u && ws[w_bit_index(state_index, b)] == 1u {
                        w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                    }
                }
            }
            case 2: {
                // X
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[z];
                }
                if is_state_thread {
                    // Flip bit
                    ws[w_bit_index(state_index, a)] ^= 1u;
                }
            }
            case 3: {
                // Y
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] ^ tableau[z];
                }
                if is_state_thread {
                    if ws[w_bit_index(state_index, a)] == 1u {
                        w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                    } else {
                        w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                    }
                    ws[w_bit_index(state_index, a)] ^= 1u;
                }
            }
            case 4: {
                // Z
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x];
                }
                if is_state_thread {
                    if ws[w_bit_index(state_index, a)] == 1u {
                        w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                    }
                }
            }
            case 5: {
                // S
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] & tableau[z];
                    tableau[z] ^= tableau[x];
                }
                if is_state_thread {
                    if ws[w_bit_index(state_index, a)] == 1u {
                        w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                    }
                }
            }
            case 6: {
                // Sdg
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[z] ^= tableau[x];
                    tableau[r] ^= tableau[x] & tableau[z];
                }
                if is_state_thread {
                    if ws[w_bit_index(state_index, a)] == 1u {
                        w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                    }
                }
            }
            case 7: {
                // H
                // Updates to ws and w_coeffs are handled by update_before_h.
                let a = qubit_params[p];
                p += 1;
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] & tableau[z];
                    let temp = tableau[z];
                    tableau[z] = tableau[x];
                    tableau[x] = temp;
                }
            }
            case 8: {
                // Piz branch
                let a = qubit_params[p];
                p += 1;
                if path[seen_p_gates] == 0u {
                    if is_state_thread {
                        w_coeffs[state_index] = mul(w_coeffs[state_index], ci_params[seen_p_gates]);
                    }
                } else {
                    if is_block_thread {
                        let r = r_column_block_index(block_index);
                        let x = x_column_block_index(block_index, a);
                        tableau[r] ^= tableau[x];
                    }
                    if is_state_thread {
                        if ws[w_bit_index(state_index, a)] == 1u {
                            w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                        }
                        w_coeffs[state_index] = mul(w_coeffs[state_index], cz_params[seen_p_gates]);
                    }
                }
                seen_p_gates += 1;
            }
            default: {}
        }
    }
}


@group(1) @binding(1) var<uniform> ci: Complex;
@group(1) @binding(2) var<uniform> cz: Complex;

@compute
@workgroup_size(64)
fn fork_apply_piz_gate(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    if id.x < column_block_length() {
        let block_index = id.x;

        let x = x_column_block_index(block_index, qubit);
        let c = c_column_block_index(block_index, c_cols);
        tableau[c] = tableau[x];
    }

    // Assign one thread to each contained state.
    if id.x < contained_states() {
        let index_i = id.x;
        let index_z = id.x + contained_states();

        for (var q = 0u; q < n; q += 1) {
            ws[w_bit_index(index_z, q)] = ws[w_bit_index(index_i, q)];
        }
        w_coeffs[index_z] = w_coeffs[index_i];

        w_coeffs[index_i] = mul(w_coeffs[index_i], ci);

        if ws[w_bit_index(index_z, qubit)] == 1u {
            w_coeffs[index_z] = mul_minus_one(w_coeffs[index_z]);
        }
        w_coeffs[index_z] = mul(w_coeffs[index_z], cz);
    }
}


struct PassInput {
    col: u32,
    a: u32,
}
@group(1) @binding(0) var<storage, read> input: PassInput;
@group(1) @binding(1) var<storage, read_write> input_next: PassInput;
@group(1) @binding(2) var<storage, read_write> pivot_out: u32;

@compute
@workgroup_size(64)
fn elimination_pass(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index >= column_block_length() {
        return;
    }

    if id.x == 0 {
        input_next.col = input.col + 1;
    }

    let aux_row = n;
    let aux_block_index = aux_row / BLOCK_SIZE;
    let aux_bit_index = aux_row % BLOCK_SIZE;

    // Find pivot row.
    var pivot_found = false;
    var pivot: u32 = 0;
    let a_block_index = input.a / BLOCK_SIZE;
    for (var i = a_block_index; i < column_block_length(); i += 1) {
        // Bitmask blocking out the auxiliary row.
        var aux_mask: BitBlock = ~0u;
        if i == aux_block_index {
            aux_mask = ~bitmask(aux_bit_index);
        }
        let block = tableau[x_column_block_index(i, input.col)] & aux_mask;
        if block != 0 {
            let row = BLOCK_SIZE * i + lsb_index(block);
            if row >= input.a {
                pivot = row;
                pivot_found = true;
                // Continue to search for the bottom-most one
            }
        }
    }

    if !pivot_found {
        if id.x == 0 {
            // Since we use the pivot_out variable to swap the pivot and a rows,
            // we set it to a here so the swap operation is a no-op.
            pivot_out = input.a;
            input_next.a = input.a;
        }
        return;
    }

    if id.x == 0 {
        pivot_out = pivot;
        input_next.a = input.a + 1;
    }

    let pivot_block_index = pivot / BLOCK_SIZE;
    let pivot_bit_index = pivot % BLOCK_SIZE;

    // Bitmask blocking out the pivot row.
    var pivot_mask: BitBlock = ~0u;
    if block_index == pivot_block_index {
        pivot_mask = ~bitmask(pivot_bit_index);
    }
    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
    let mask = tableau[x_column_block_index(block_index, input.col)] & pivot_mask;
    if mask == 0 {
        return;
    }

    // Determine phase change caused by multiplication of the individual Pauli matrices.
    // We encode phase as `phase = 2*phase_bit2 + phase_bit1`,
    // but in a bit block so we can operate on all rows in the block at once.
    // Since i^phase works modulo 4, we can just use two bits and let additions/subtractions wrap around.
    var phase_bit1: BitBlock = 0u;
    var phase_bit2: BitBlock = 0u;
    for (var col2 = 0u; col2 < n; col2 += 1) {
        let x1 = tableau[x_column_block_index(block_index, col2)];
        let z1 = tableau[z_column_block_index(block_index, col2)];
        // Fill these blocks with the bits in the pivot row.
        var x2: BitBlock = 0u;
        if x_bit(pivot, col2) {
            x2 = ~0u;
        }
        var z2: BitBlock = 0u;
        if z_bit(pivot, col2) {
            z2 = ~0u;
        }

        // XY = +iZ
        // YZ = +iX
        // ZX = +iY
        let add = (x_block(x1, z1) & y_block(x2, z2))
            | (y_block(x1, z1) & z_block(x2, z2))
            | (z_block(x1, z1) & x_block(x2, z2));
        phase_bit2 ^= add & phase_bit1;
        phase_bit1 ^= add;

        // YX = -iZ
        // ZY = -iX
        // XZ = -iY
        let sub = (y_block(x1, z1) & x_block(x2, z2))
            | (z_block(x1, z1) & y_block(x2, z2))
            | (x_block(x1, z1) & z_block(x2, z2));
        phase_bit2 ^= sub & ~phase_bit1;
        phase_bit1 ^= sub;
    }
    // A valid stabilizer row can only ever have a prefix of +1 or -1.
    // phase_bit1 being 1 implies a phase of either 1 or 3, making the prefix i or -i respectively.
    // This should never be able to happen, and we cannot represent it.
    // phase_bit2 = 1  =>  phase = 2  =>  i^2 = -1    flip the sign bit.
    // phase_bit2 = 0  =>  phase = 0  =>  i^0 = +1    do nothing.
    tableau[r_column_block_index(block_index)] ^= phase_bit2 & mask;

    // XOR
    for (var j = 0u; j < n + n + 1 + c_cols; j += 1) {
        if bit(pivot, j)  {
            tableau[column_block_index(block_index, j)] ^= mask;
        }
    }
}

@compute
@workgroup_size(64)
fn swap_pass(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each column.
    let j = id.x;
    if j >= n + n + 1 + c_cols {
        return;
    }

    // Swap these two rows.
    let row1 = input.a;
    let row2 = pivot_out;

    let row1_block_index = row1 / BLOCK_SIZE;
    let row2_block_index = row2 / BLOCK_SIZE;
    let row1_bit_index = row1 % BLOCK_SIZE;
    let row2_bit_index = row2 % BLOCK_SIZE;
    let row1_bitmask = bitmask(row1_bit_index);
    let row2_bitmask = bitmask(row2_bit_index);

    let block1 = column_block_index(row1_block_index, j);
    let block2 = column_block_index(row2_block_index, j);
    let bit1 = (tableau[block1] & row1_bitmask) != 0;
    let bit2 = (tableau[block2] & row2_bitmask) != 0;
    if bit1 && !bit2 {
        tableau[block1] = unset_bit(tableau[block1], row1_bit_index);
        tableau[block2] = set_bit(tableau[block2], row2_bit_index);
    } else if !bit1 && bit2 {
        tableau[block1] = set_bit(tableau[block1], row1_bit_index);
        tableau[block2] = unset_bit(tableau[block2], row2_bit_index);
    }
}


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
        ws[w_bit_index(state_index, qubit)] = 0;
    } else {
        if ws[w_bit_index(state_index, qubit)] == 0 {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], 2.0 / sqrt(2.0));
        } else {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], -2.0 / sqrt(2.0));
        }
        ws[w_bit_index(state_index, qubit)] = 1;
    }
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

    let r = coeff_ratio(state_index);
    output[state_index] = mul(w_coeffs[state_index], r);
}


// Assume tableau in RREF.
fn coeff_ratio_flipped_bit(state_index: u32) -> Complex {
    let flipped_bit = qubit;

    // Find pivot row.
    var row: u32 = 0;
    for (var j: u32 = 0; j < n; j += 1) {
        if x_bit(j, flipped_bit) {
            row = j;
            break;
        }
    }

    var res = Complex(1.0, 0.0);
    if row_negative(state_index, row) {
        res = Complex(-1.0, 0.0);
    }
    for (var q: u32 = 0; q < n; q += 1) {
        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        if i(row, q) && (flipped_bit != q) {
            // Multiply by 1
        } else if x(row, q) && (flipped_bit == q) {
            // Multiply by 1
        } else if y(row, q) && (flipped_bit == q) {
            if ws[w_bit_index(state_index, q)] == 0 {
                res = mul_i(res);
            } else {
                res = mul_minus_i(res);
            }
        } else if z(row, q) && (flipped_bit != q) {
            if ws[w_bit_index(state_index, q)] == 0 {
                // Multiply by 1
            } else {
                res = mul_minus_one(res);
            }
        } else {
            // Multiply by 0
            return Complex(0.0, 0.0);
        }
    }
    return res;
}

// Assume tableau in RREF.
//
// Since the auxiliary row is shared between all contained states,
// the stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ
// is derived one tensor element at a time instead of being stored in the auxiliary row.
fn coeff_ratio(state_index: u32) -> Complex {
    // We encode the result as i^phase, and let the phase wrap around modulo 4.
    var phase: u32 = 0;

    // Determine the sign of the derived stabilizer.
    var row: u32 = 0;
    for (var q: u32 = 0; q < n; q += 1) {
        if x_bit(row, q) {
            if ws[w_bit_index(state_index, q)] != w_target[q] && row_negative(state_index, row) {
                phase += 2;
            }
            row += 1;
        }
    }

    for (var q2: u32 = 0; q2 < n; q2 += 1) {
        // Derive the q2'th tensor element of the stabilizer,
        // including the phase change caused by multiplication of the individual Pauli matrices.
        var x_acc = false;
        var z_acc = false;
        var row2: u32 = 0;
        for (var q: u32 = 0; q < n; q += 1) {
            if x_bit(row2, q) {
                if ws[w_bit_index(state_index, q)] != w_target[q] {
                    let x_row = x_bit(row2, q2);
                    let z_row = z_bit(row2, q2);
                    phase += product_phase(x_acc, z_acc, x_row, z_row);
                    x_acc = x_acc != x_row;
                    z_acc = z_acc != z_row;
                }
                row2 += 1;
            }
        }

        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        let w1_bit = ws[w_bit_index(state_index, q2)];
        let differs = w1_bit != w_target[q2];
        if x_acc != differs {
            // Multiply by 0
            return Complex(0.0, 0.0);
        }
        if x_acc && z_acc {
            // Y
            phase += 1;
        }
        if z_acc && w1_bit == 1 {
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
    // Make room for the auxiliary row.
    return div_ceil(n + 1, BLOCK_SIZE);
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

fn i(row: u32, q: u32) -> bool {
    return !x_bit(row, q) && !z_bit(row, q);
}
fn x(row: u32, q: u32) -> bool {
    return x_bit(row, q) && !z_bit(row, q);
}
fn y(row: u32, q: u32) -> bool {
    return x_bit(row, q) && z_bit(row, q);
}
fn z(row: u32, q: u32) -> bool {
    return !x_bit(row, q) && z_bit(row, q);
}

fn x_block(x: BitBlock, z: BitBlock) -> BitBlock {
    return x & ~z;
}
fn z_block(x: BitBlock, z: BitBlock) -> BitBlock {
    return ~x & z;
}
fn y_block(x: BitBlock, z: BitBlock) -> BitBlock {
    return x & z;
}

// Get the bitmask for the i'th bit, e.g.
// bitmask(0) -> 10000000
// bitmask(1) -> 01000000
// bitmask(6) -> 00000010
fn bitmask(i: u32) -> BitBlock {
    return 1u << (BLOCK_SIZE - 1u - i);
}

// Get the index of the least significant (right-most) bit in the given block, e.g.
// lsb_index(10000000)
//           ^0
// lsb_index(01000000)
//            ^1
// lsb_index(11010000)
//              ^3
fn lsb_index(block: BitBlock) -> u32 {
    let trailing_zeros = countTrailingZeros(block);
    return BLOCK_SIZE - 1u - trailing_zeros;
}

// Set the i'th bit of the given block, i.e. set the bit to 1.
// set_bit(00000000, 0) -> 10000000
// set_bit(10001000, 1) -> 11001000
// set_bit(10011000, 6) -> 10011010
fn set_bit(b: BitBlock, i: u32) -> BitBlock {
    return b | bitmask(i);
}
// Unset the i'th bit of the given block, i.e. set the bit to 0.
// unset_bit(11111111, 0) -> 01111111
// unset_bit(01110111, 1) -> 00110111
// unset_bit(01100111, 6) -> 01100101
fn unset_bit(b: BitBlock, i: u32) -> BitBlock {
    return b & ~bitmask(i);
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
