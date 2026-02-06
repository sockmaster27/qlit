const BLOCK_SIZE: u32 = 32;
alias BitBlock = u32;

// T = C_I*I + C_Z*Z
const C_I: Complex = Complex(
    0.5 + 0.5 * inverseSqrt(2.0),
    0.5 * inverseSqrt(2.0),
);
const C_Z: Complex = Complex(
    0.5 - 0.5 * inverseSqrt(2.0),
    -0.5 * inverseSqrt(2.0),
);

// Tdg = C_I_DG*I + C_Z_DG*Z
const C_I_DG: Complex = Complex(
    0.5 + 0.5 * inverseSqrt(2.0),
    -0.5 * inverseSqrt(2.0),
);
const C_Z_DG: Complex = Complex(
    0.5 - 0.5 * inverseSqrt(2.0),
    0.5 * inverseSqrt(2.0),
);

@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<uniform> max_batches: u32;
@group(0) @binding(2) var<storage, read> path: array<u32>;
@group(0) @binding(3) var<storage, read_write> tableau: array<BitBlock>;
@group(0) @binding(4) var<storage, read_write> active_batches: u32;
@group(0) @binding(5) var<uniform> seen_t_gates: u32;
@group(0) @binding(6) var<storage, read_write> ws: array<u32>;
@group(0) @binding(7) var<storage, read_write> w_coeffs: array<Complex>;

fn w_bit_index(batch_index: u32, q: u32) -> u32 {
    return batch_index * n + q;
}

@group(1) @binding(0) var<uniform> qubit: u32;


@compute
@workgroup_size(64)
fn zero(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to row (block).
    let block_index = id.x;
    if block_index >= single_column_block_length() {
        return;
    }

    // Assume all buffers are zero-initialized.

    if id.x == 0 {
        active_batches = 1;
        w_coeffs[0] = Complex(1.0, 0.0);
    }
    
    for (var i: u32 = 0; i < BLOCK_SIZE; i += 1) {
        let b = z_column_block_index(0, block_index, block_index * BLOCK_SIZE + i);
        tableau[b] = bitmask(i);
    }
}


@group(1) @binding(0) var<storage, read> gates: array<u32>;
@group(1) @binding(1) var<storage, read> qubit_params: array<u32>;

@compute
@workgroup_size(64)
fn apply_gates(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    if id.x >= active_column_block_length() {
        return;
    }
    let batch_index = id.x / single_column_block_length();
    let block_index = id.x % single_column_block_length();

    var p: u32 = 0;
    for (var i: u32 = 0; i < arrayLength(&gates); i += 1) {
        switch gates[i] {
            case 0: {
                // CNOT
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
                let r = r_column_block_index(batch_index, block_index);
                let xa = x_column_block_index(batch_index, block_index, a);
                let za = z_column_block_index(batch_index, block_index, a);
                let xb = x_column_block_index(batch_index, block_index, b);
                let zb = z_column_block_index(batch_index, block_index, b);
                tableau[r] ^= tableau[xa] & tableau[zb] & ~(tableau[xb] ^ tableau[za]);
                tableau[za] ^= tableau[zb];
                tableau[xb] ^= tableau[xa];
                if block_index == 0 {
                    ws[w_bit_index(batch_index, b)] ^= ws[w_bit_index(batch_index, a)];
                }
            }
            case 1: {
                // CZ
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
                let r = r_column_block_index(batch_index, block_index);
                let xa = x_column_block_index(batch_index, block_index, a);
                let za = z_column_block_index(batch_index, block_index, a);
                let xb = x_column_block_index(batch_index, block_index, b);
                let zb = z_column_block_index(batch_index, block_index, b);
                tableau[r] ^= (tableau[xb] & tableau[zb])
                    ^ (tableau[xa] & tableau[xb] & ~(tableau[zb] ^ tableau[za]))
                    ^ (tableau[xb] & (tableau[zb] ^ tableau[xa]));
                tableau[za] ^= tableau[xb];
                tableau[zb] ^= tableau[xa];
                if block_index == 0 {
                    if ws[w_bit_index(batch_index, a)] == 1u && ws[w_bit_index(batch_index, b)] == 1u {
                        w_coeffs[batch_index] = mul_minus_one(w_coeffs[batch_index]);
                    }
                }
            }
            case 2: {
                // X
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let z = z_column_block_index(batch_index, block_index, a);
                tableau[r] ^= tableau[z];
                if block_index == 0 {
                    // Flip bit
                    ws[w_bit_index(batch_index, a)] ^= 1u;
                }
            }
            case 3: {
                // Y
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let x = x_column_block_index(batch_index, block_index, a);
                let z = z_column_block_index(batch_index, block_index, a);
                tableau[r] ^= tableau[x] ^ tableau[z];
                if block_index == 0 {
                    if ws[w_bit_index(batch_index, a)] == 1u {
                        w_coeffs[batch_index] = mul_minus_i(w_coeffs[batch_index]);
                    } else {
                        w_coeffs[batch_index] = mul_i(w_coeffs[batch_index]);
                    }
                    ws[w_bit_index(batch_index, a)] ^= 1u;
                }
            }
            case 4: {
                // Z
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let x = x_column_block_index(batch_index, block_index, a);
                tableau[r] ^= tableau[x];
                if block_index == 0 {
                    if ws[w_bit_index(batch_index, a)] == 1u {
                        w_coeffs[batch_index] = mul_minus_one(w_coeffs[batch_index]);
                    }
                }
            }
            case 5: {
                // S
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let x = x_column_block_index(batch_index, block_index, a);
                let z = z_column_block_index(batch_index, block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                tableau[z] ^= tableau[x];
                if block_index == 0 {
                    if ws[w_bit_index(batch_index, a)] == 1u {
                        w_coeffs[batch_index] = mul_i(w_coeffs[batch_index]);
                    }
                }
            }
            case 6: {
                // Sdg
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let x = x_column_block_index(batch_index, block_index, a);
                let z = z_column_block_index(batch_index, block_index, a);
                tableau[z] ^= tableau[x];
                tableau[r] ^= tableau[x] & tableau[z];
                if block_index == 0 {
                    if ws[w_bit_index(batch_index, a)] == 1u {
                        w_coeffs[batch_index] = mul_minus_i(w_coeffs[batch_index]);
                    }
                }
            }
            case 7: {
                // H
                // Updates to ws and w_coeffs are handled by update_before_h.
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(batch_index, block_index);
                let x = x_column_block_index(batch_index, block_index, a);
                let z = z_column_block_index(batch_index, block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                let temp = tableau[z];
                tableau[z] = tableau[x];
                tableau[x] = temp;
            }
            case 8: {
                // T branch
                if path[seen_t_gates] == 0u {
                    if block_index == 0 {
                        w_coeffs[batch_index] = mul(w_coeffs[batch_index], C_I);
                    }
                } else {
                    let a = qubit_params[p];
                    let r = r_column_block_index(batch_index, block_index);
                    let x = x_column_block_index(batch_index, block_index, a);
                    tableau[r] ^= tableau[x];
                    if block_index == 0 {
                        if ws[w_bit_index(batch_index, a)] == 1u {
                            w_coeffs[batch_index] = mul_minus_one(w_coeffs[batch_index]);
                        }
                        w_coeffs[batch_index] = mul(w_coeffs[batch_index], C_Z);
                    }
                }
                p += 1;
            }
            case 9: {
                // Tdg branch
                if path[seen_t_gates] == 0u {
                    if block_index == 0 {
                        w_coeffs[batch_index] = mul(w_coeffs[batch_index], C_I_DG);
                    }
                } else {
                    let a = qubit_params[p];
                    let r = r_column_block_index(batch_index, block_index);
                    let x = x_column_block_index(batch_index, block_index, a);
                    tableau[r] ^= tableau[x];
                    if block_index == 0 {
                        if ws[w_bit_index(batch_index, a)] == 1u {
                            w_coeffs[batch_index] = mul_minus_one(w_coeffs[batch_index]);
                        }
                        w_coeffs[batch_index] = mul(w_coeffs[batch_index], C_Z_DG);
                    }
                }
                p += 1;
            }
            default: {}
        }
    }
}


@compute
@workgroup_size(64)
fn apply_t_gate_parallel(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    apply_gate_parallel(id.x, C_I, C_Z);
}

@compute
@workgroup_size(64)
fn apply_tdg_gate_parallel(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    apply_gate_parallel(id.x, C_I_DG, C_Z_DG);
}

fn apply_gate_parallel(id: u32, c_i: Complex, c_z: Complex) {
    // Assign one thread to each row (block).
    if id >= active_column_block_length() {
        return;
    }
    let batch_index = id / single_column_block_length();
    let block_index = id % single_column_block_length();
    for (var q: u32 = 0u; q < n; q += 1) {
        let x1 = x_column_block_index(batch_index, block_index, q);
        let x2 = x_column_block_index(batch_index + active_batches, block_index, q);
        tableau[x2] = tableau[x1];
        let z1 = z_column_block_index(batch_index, block_index, q);
        let z2 = z_column_block_index(batch_index + active_batches, block_index, q);
        tableau[z2] = tableau[z1];
    }
    
    let x = x_column_block_index(batch_index, block_index, qubit);
    let r1 = r_column_block_index(batch_index, block_index);
    let r2 = r_column_block_index(batch_index + active_batches, block_index);
    tableau[r2] = tableau[r1] ^ tableau[x];

    if block_index == 0 {
        let index_i = batch_index;
        let index_z = batch_index + active_batches;
        for (var q = 0u; q < n; q += 1) {
            ws[w_bit_index(index_z, q)] = ws[w_bit_index(index_i, q)];
        }
        w_coeffs[index_z] = w_coeffs[index_i];

        w_coeffs[index_i] = mul(w_coeffs[index_i], c_i);
        w_coeffs[index_z] = mul(w_coeffs[index_z], c_z);
        if ws[w_bit_index(index_z, qubit)] == 1u {
            w_coeffs[index_z] = mul_minus_one(w_coeffs[index_z]);
        }
    }
}


@group(1) @binding(0) var<storage, read> col_in: u32;
@group(1) @binding(1) var<storage, read_write> col_out: u32;
@group(1) @binding(2) var<storage, read> a_in: u32;
@group(1) @binding(3) var<storage, read_write> a_out: u32;
@group(1) @binding(4) var<storage, read_write> pivot_out: u32;

@compute
@workgroup_size(64)
fn elimination_pass(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    if id.x >= active_column_block_length() {
        return;
    }
    let batch_index = id.x / single_column_block_length();
    let block_index = id.x % single_column_block_length();

    if id.x == 0 {
        col_out = col_in + 1;
    }

    let aux_row = n;
    let aux_block_index = aux_row / BLOCK_SIZE;
    let aux_bit_index = aux_row % BLOCK_SIZE;

    // Find pivot row.
    var pivot_found = false;
    var pivot: u32 = 0;
    let a_block_index = a_in / BLOCK_SIZE;
    for (var i = a_block_index; i < single_column_block_length(); i += 1) {
        // Bitmask blocking out the auxiliary row.
        var aux_mask: BitBlock = ~0u;
        if i == aux_block_index {
            aux_mask = ~bitmask(aux_bit_index);
        }
        let block = tableau[x_column_block_index(batch_index, i, col_in)] & aux_mask;
        if block != 0 {
            let row = BLOCK_SIZE * i + lsb_index(block);
            if row >= a_in {
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
            pivot_out = a_in;
            a_out = a_in;
        }
        return;
    }

    if id.x == 0 {
        pivot_out = pivot;
        a_out = a_in + 1;
    }

    let pivot_block_index = pivot / BLOCK_SIZE;
    let pivot_bit_index = pivot % BLOCK_SIZE;

    // Bitmask blocking out the pivot row.
    var pivot_mask: BitBlock = ~0u;
    if block_index == pivot_block_index {
        pivot_mask = ~bitmask(pivot_bit_index);
    }
    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
    let mask = tableau[x_column_block_index(batch_index, block_index, col_in)] & pivot_mask;
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
        let x1 = tableau[x_column_block_index(batch_index, block_index, col2)];
        let z1 = tableau[z_column_block_index(batch_index, block_index, col2)];
        // Fill these blocks with the bits in the pivot row.
        var x2: BitBlock = 0u;
        if x_bit(batch_index, pivot, col2) { 
            x2 = ~0u;
        }
        var z2: BitBlock = 0u;
        if z_bit(batch_index, pivot, col2) {
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
    tableau[r_column_block_index(batch_index, block_index)] ^= phase_bit2 & mask;

    // XOR
    for (var j = 0u; j < n + n + 1; j += 1) {
        if bit(batch_index, pivot, j) == true {
            tableau[column_block_index(batch_index, block_index, j)] ^= mask;
        }
    }
}

@compute
@workgroup_size(64)
fn swap_pass(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each batch.
    let batch_index = id.x;
    if batch_index >= active_batches {
        return;
    }

    // Swap these two rows.
    let row1 = a_in;
    let row2 = pivot_out;

    let row1_block_index = row1 / BLOCK_SIZE;
    let row2_block_index = row2 / BLOCK_SIZE;
    let row1_bit_index = row1 % BLOCK_SIZE;
    let row2_bit_index = row2 % BLOCK_SIZE;
    let row1_bitmask = bitmask(row1_bit_index);
    let row2_bitmask = bitmask(row2_bit_index);

    for (var j: u32 = 0; j < n + n + 1; j += 1) {
        let block1 = column_block_index(batch_index, row1_block_index, j);
        let block2 = column_block_index(batch_index, row2_block_index, j);
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
}


@compute
@workgroup_size(64)
fn update_before_h(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each batch.
    let batch_index = id.x;
    if batch_index >= active_batches {
        return;
    }

    let r = coeff_ratio_flipped_bit(batch_index);
    if !eq(r, Complex(-1.0, 0.0)) {
        w_coeffs[batch_index] = mul(w_coeffs[batch_index], div_real(add_real(r, 1.0), sqrt(2.0)));
        ws[w_bit_index(batch_index, qubit)] = 0;
    } else {
        if ws[w_bit_index(batch_index, qubit)] == 0 {
            w_coeffs[batch_index] = mul_real(w_coeffs[batch_index], 2.0 / sqrt(2.0));
        } else {
            w_coeffs[batch_index] = mul_real(w_coeffs[batch_index], -2.0 / sqrt(2.0));
        }
        ws[w_bit_index(batch_index, qubit)] = 1;
    }
}


@group(1) @binding(0) var<storage, read> w_target: array<u32>;
@group(1) @binding(1) var<storage, read_write> output: array<Complex>;

@compute
@workgroup_size(64)
fn compute_output(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each batch.
    let batch_index = id.x;
    if batch_index >= active_batches {
        return;
    }

    let r = coeff_ratio(batch_index);
    output[batch_index] = mul(w_coeffs[batch_index], r);
}


// Assume tableau in RREF.
fn coeff_ratio_flipped_bit(batch_index: u32) -> Complex {
    let flipped_bit = qubit;

    // Find pivot row.
    var row: u32 = 0;
    for (var j: u32 = 0; j < n; j += 1) {
        if x_bit(batch_index, j, flipped_bit) {
            row = j;
            break;
        }
    }

    var res = Complex(1.0, 0.0);
    if row_negative(batch_index, row) {
        res = Complex(-1.0, 0.0);
    } 
    for (var q: u32 = 0; q < n; q += 1) {
        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        if i(batch_index, row, q) && (flipped_bit != q) {
            // Multiply by 1
        } else if x(batch_index, row, q) && (flipped_bit == q) {
            // Multiply by 1
        } else if y(batch_index, row, q) && (flipped_bit == q) {
            if ws[w_bit_index(batch_index, q)] == 0 {
                res = mul_i(res);
            } else {
                res = mul_minus_i(res);
            }
        } else if z(batch_index, row, q) && (flipped_bit != q) {
            if ws[w_bit_index(batch_index, q)] == 0 {
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
fn coeff_ratio(batch_index: u32) -> Complex {
    let aux_row = n;
    let aux_block_index = aux_row / BLOCK_SIZE;
    let aux_bit_index = aux_row % BLOCK_SIZE;
    let aux_bitmask = bitmask(aux_bit_index);

    // Reset the auxiliary row.
    for (var j: u32 = 0; j < n + n + 1; j += 1) {
        tableau[column_block_index(batch_index, aux_block_index, j)] &= ~aux_bitmask;
    }
    // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
    var row: u32 = 0;
    for (var q: u32 = 0; q < n; q += 1) {
        if x_bit(batch_index, row, q) {
            if ws[w_bit_index(batch_index, q)] != w_target[q] {
                multiply_rows_into(batch_index, row, aux_row);
            }
            row += 1;
        }
    }

    var res = Complex(1.0, 0.0);
    if row_negative(batch_index, aux_row) {
        res = Complex(-1.0, 0.0);
    } 
    for (var q: u32 = 0; q < n; q += 1) {
        if i(batch_index, aux_row, q) && (ws[w_bit_index(batch_index, q)] == w_target[q]) {
            // Multiply by 1
        } else if x(batch_index, aux_row, q) && (ws[w_bit_index(batch_index, q)] != w_target[q]) {
            // Multiply by 1
        } else if y(batch_index, aux_row, q) && (ws[w_bit_index(batch_index, q)] != w_target[q]) {
            if ws[w_bit_index(batch_index, q)] == 0 {
                res = mul_i(res);
            } else {
                res = mul_minus_i(res);
            }
        } else if z(batch_index, aux_row, q) && (ws[w_bit_index(batch_index, q)] == w_target[q]) {
            if ws[w_bit_index(batch_index, q)] == 0 {
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

// Set row with index `dst` to be the product of the `src` and `dst` rows.
//
// NOTE: Since all stabilizers must commute, multiplication order is irrelevant.
fn multiply_rows_into(batch_index: u32, src: u32, dst: u32) {
    let src_block_index = src / BLOCK_SIZE;
    let dst_block_index = dst / BLOCK_SIZE;
    let src_bit_index = src % BLOCK_SIZE;
    let dst_bit_index = dst % BLOCK_SIZE;
    let src_bitmask = bitmask(src_bit_index);
    let dst_bitmask = bitmask(dst_bit_index);

    // Determine phase shift.
    var phase: u32 = 0;
    for (var q: u32 = 0; q < n; q += 1) {
        if x(batch_index, src, q) && y(batch_index, dst, q) {
            phase += 1;
        } else if x(batch_index, src, q) && z(batch_index, dst, q) {
            phase += 3;
        } else if y(batch_index, src, q) && z(batch_index, dst, q) {
            phase += 1;
        } else if y(batch_index, src, q) && x(batch_index, dst, q) {
            phase += 3;
        } else if z(batch_index, src, q) && x(batch_index, dst, q) {
            phase += 1;
        } else if z(batch_index, src, q) && y(batch_index, dst, q) {
            phase += 3;
        }
        phase %= 4;
    }
    if phase == 2 {
        // Negate the sign bit.
        tableau[r_column_block_index(batch_index, dst_block_index)] ^= dst_bitmask;
    }
    // phase == 0 implies multiplication by 1, do nothing.
    // Any other phase should be impossible since stabilizers must commute.

    // XOR
    for (var j: u32 = 0; j < n + n + 1; j += 1) {
        let src_block = column_block_index(batch_index, src_block_index, j);
        let dst_block = column_block_index(batch_index, dst_block_index, j);
        tableau[dst_block] ^= align_bit_to(
            tableau[src_block] & src_bitmask,
            src_bit_index,
            dst_bit_index,
        );
    }
}


// Get whether the given row is negative or not, i.e. the contents of the sign bit.
fn row_negative(batch_index: u32, row: u32) -> bool {
    let row_block_index = row / BLOCK_SIZE;
    let row_bit_index = row % BLOCK_SIZE;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[r_column_block_index(batch_index, row_block_index)] & row_bitmask) != 0;
}

// Get the index of the i'th block of the `j`th column.
fn column_block_index(batch_index: u32, i: u32, j: u32) -> u32 {
    let batch_start_block = batch_index * single_column_block_length();
    return (batch_start_block + i) * (n + n + 1) + j;
}
// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
fn x_column_block_index(batch_index: u32, i: u32, q: u32) -> u32 {
    return column_block_index(batch_index, i, 2 * q);
}
// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
fn z_column_block_index(batch_index: u32, i: u32, q: u32) -> u32 {
    return column_block_index(batch_index, i, 2 * q + 1);
}
// Get the index of the i'th block of the r column.
fn r_column_block_index(batch_index: u32, i: u32) -> u32 {
    return column_block_index(batch_index, i, 2 * n);
}

// Get the block-length of the columns in a single tableau batch.
fn single_column_block_length() -> u32 {
    // Make room for the auxiliary row.
    return div_ceil(n + 1, BLOCK_SIZE);
}
// Get the block-length of the combined active tableau batches.
fn active_column_block_length() -> u32 {
    return single_column_block_length() * active_batches;
}

// Get the value of the bit corresponding to the `j`th column in the `row`th row.
fn bit(batch_index: u32, row: u32, j: u32) -> bool {
    let row_block_index = row / BLOCK_SIZE;
    let row_bit_index = row % BLOCK_SIZE;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[column_block_index(batch_index, row_block_index, j)] & row_bitmask) != 0u;
}
// Get the value of the x bit corresponding to the `q`th tensor element in the `row`th row.
fn x_bit(batch_index: u32, row: u32, q: u32) -> bool {
    return bit(batch_index, row, 2 * q);
}
// Get the value of the z bit corresponding to the `q`th tensor element in the `row`th row.
fn z_bit(batch_index: u32, row: u32, q: u32) -> bool {
    return bit(batch_index, row, 2 * q + 1);
}

fn i(batch_index: u32, row: u32, q: u32) -> bool {
    return !x_bit(batch_index, row, q) && !z_bit(batch_index, row, q);
}
fn x(batch_index: u32, row: u32, q: u32) -> bool {
    return x_bit(batch_index, row, q) && !z_bit(batch_index, row, q);
}
fn y(batch_index: u32, row: u32, q: u32) -> bool {
    return x_bit(batch_index, row, q) && z_bit(batch_index, row, q);
}
fn z(batch_index: u32, row: u32, q: u32) -> bool {
    return !x_bit(batch_index, row, q) && z_bit(batch_index, row, q);
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

// Bit-shift the given block such that the `src`th bit is moved to the `dst`th position.
fn align_bit_to(block: BitBlock, src: u32, dst: u32) -> BitBlock {
    if dst < src {
        return block << (src - dst);
    } else {
        return block >> (dst - src);
    }
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
