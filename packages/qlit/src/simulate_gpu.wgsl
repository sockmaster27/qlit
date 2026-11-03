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
// See ExtendedTableau in tableau.rs.
@group(0) @binding(3) var<storage, read_write> tableau: array<BitBlock>;
// The current number of active c columns in the tableau.
@group(0) @binding(4) var<uniform> c_cols: u32;
@group(0) @binding(5) var<storage, read_write> ws: array<BitBlock>;
@group(0) @binding(6) var<storage, read_write> w_coeffs: array<Complex>;
// Pivot bookkeeping established by bring_into_rref, instead of physically reordering the rows of the tableau.
// For each row, holds the column of its pivot plus one (0 if the row has no pivot).
@group(0) @binding(7) var<storage, read_write> row_pivots: array<u32>;

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
@group(1) @binding(4) var<uniform> initial_path_index: u32;

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

    // Only advance p at the end of each case, after all uses of the parameters it points to.
    // On the macos-15-intel CI runners, Metal otherwise ends up reading a parameter that is only used
    // inside `if is_block_thread` after the increment when that branch diverges between threads,
    // making e.g. H act on the next gate's qubit.
    var p: u32 = 0;
    var path_index = initial_path_index;
    for (var i: u32 = 0; i < arrayLength(&gates); i += 1) {
        switch gates[i] {
            case 0: {
                // CNOT
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
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
                if is_state_thread && w_bit(state_index, a) {
                    flip_w_bit(state_index, b);
                }
                p += 2;
            }
            case 1: {
                // CZ
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let xa = x_column_block_index(block_index, a);
                    let za = z_column_block_index(block_index, a);
                    let xb = x_column_block_index(block_index, b);
                    let zb = z_column_block_index(block_index, b);
                    tableau[r] ^= tableau[xa] & tableau[xb] & (tableau[za] ^ tableau[zb]);
                    tableau[za] ^= tableau[xb];
                    tableau[zb] ^= tableau[xa];
                }
                if is_state_thread {
                    if w_bit(state_index, a) && w_bit(state_index, b) {
                        w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                    }
                }
                p += 2;
            }
            case 2: {
                // X
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[z];
                }
                if is_state_thread {
                    flip_w_bit(state_index, a);
                }
                p += 1;
            }
            case 3: {
                // Y
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] ^ tableau[z];
                }
                if is_state_thread {
                    if w_bit(state_index, a) {
                        w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                    } else {
                        w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                    }
                    flip_w_bit(state_index, a);
                }
                p += 1;
            }
            case 4: {
                // Z
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x];
                }
                if is_state_thread {
                    if w_bit(state_index, a) {
                        w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                    }
                }
                p += 1;
            }
            case 5: {
                // S
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] & tableau[z];
                    tableau[z] ^= tableau[x];
                }
                if is_state_thread {
                    if w_bit(state_index, a) {
                        w_coeffs[state_index] = mul_i(w_coeffs[state_index]);
                    }
                }
                p += 1;
            }
            case 6: {
                // Sdg
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[z] ^= tableau[x];
                    tableau[r] ^= tableau[x] & tableau[z];
                }
                if is_state_thread {
                    if w_bit(state_index, a) {
                        w_coeffs[state_index] = mul_minus_i(w_coeffs[state_index]);
                    }
                }
                p += 1;
            }
            case 7: {
                // H
                // Updates to ws and w_coeffs are handled by update_before_h.
                let a = qubit_params[p];
                if is_block_thread {
                    let r = r_column_block_index(block_index);
                    let x = x_column_block_index(block_index, a);
                    let z = z_column_block_index(block_index, a);
                    tableau[r] ^= tableau[x] & tableau[z];
                    let temp = tableau[z];
                    tableau[z] = tableau[x];
                    tableau[x] = temp;
                }
                p += 1;
            }
            case 8: {
                // Piz branch
                let a = qubit_params[p];
                if path[path_index] == 0u {
                    if is_state_thread {
                        w_coeffs[state_index] = mul(w_coeffs[state_index], ci_params[path_index]);
                    }
                } else {
                    if is_block_thread {
                        let r = r_column_block_index(block_index);
                        let x = x_column_block_index(block_index, a);
                        tableau[r] ^= tableau[x];
                    }
                    if is_state_thread {
                        if w_bit(state_index, a) {
                            w_coeffs[state_index] = mul_minus_one(w_coeffs[state_index]);
                        }
                        w_coeffs[state_index] = mul(w_coeffs[state_index], cz_params[path_index]);
                    }
                }
                path_index += 1;
                p += 1;
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
    var z_coeff1 = w_coeffs[i.y];
    var z_coeff2 = w_coeffs[i.x];
    if w_bit(i.y, qubit) {
        z_coeff1 = mul_minus_one(z_coeff1);
    }
    if w_bit(i.x, qubit) {
        z_coeff2 = mul_minus_one(z_coeff2);
    }
    z_coeff1 = mul(z_coeff1, coeff_ratio(i.y, i.x, i.x));
    z_coeff2 = mul(z_coeff2, coeff_ratio(i.x, i.y, i.y));
    
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


const RREF_WORKGROUP_SIZE: u32 = 256;

// Bitmask with zeros in indices corresponding to rows where pivots have already been seen.
@group(1) @binding(0) var<storage, read_write> pivot_mask: array<BitBlock>;

// The pivot row selected for the current column, plus one (0 if none was found).
var<workgroup> pivot_plus1: u32;
// Whether any other rows than the pivot row have a set bit in the current column, i.e. need to be eliminated.
var<workgroup> needs_elimination: bool;

// Bring the tableau's x part into reduced row echelon form.
// This is done by a single workgroup, in order to avoid the overhead of dispatching each step separately.
// The workgroup loops over the columns, alternating between selecting a pivot row (done by a single thread)
// and eliminating the pivot column from all other rows (done by all threads).
// This mirrors ExtendedTableau::bring_into_rref in tableau.rs.
@compute
@workgroup_size(RREF_WORKGROUP_SIZE)
fn bring_into_rref(
    @builtin(local_invocation_index) thread: u32
) {
    // The FXC compiler (DX12 backend) rejects barriers in loops that contain any thread-dependent control flow
    // around other loops. To appease it, every loop in here has a trip count that is the same for all threads,
    // and thread-dependent conditions only guard simple statements inside the loop bodies.
    for (var k = 0u; k < strided_rounds(column_block_length()); k += 1) {
        let i = k * RREF_WORKGROUP_SIZE + thread;
        if i < column_block_length() {
            pivot_mask[i] = ~0u;
        }
    }
    storageBarrier();
    workgroupBarrier();

    for (var col = 0u; col < n; col += 1) {
        select_pivot(col, thread == 0);
        storageBarrier();
        workgroupBarrier();

        // Usually, the tableau is already (mostly) in RREF, and there is nothing to eliminate.
        // workgroupUniformLoad ends in a barrier, which lets the compiler know that all threads agree on the result.
        if workgroupUniformLoad(&needs_elimination) {
            let pivot = pivot_plus1 - 1;
            // Assign each thread a strided subset of the rows (blocks).
            for (var k = 0u; k < strided_rounds(column_block_length()); k += 1) {
                let i = k * RREF_WORKGROUP_SIZE + thread;
                eliminate(i, pivot, col, i < column_block_length());
            }
        }
        storageBarrier();
        workgroupBarrier();
    }

    // Reset the pivots of all-zero rows.
    for (var k = 0u; k < strided_rounds(n); k += 1) {
        let row = k * RREF_WORKGROUP_SIZE + thread;
        if row < n && (pivot_mask[row / BLOCK_SIZE] & bitmask(row % BLOCK_SIZE)) != 0 {
            row_pivots[row] = 0;
        }
    }
}

// The number of rounds needed for the threads of the workgroup to cover `count` items in a strided loop.
fn strided_rounds(count: u32) -> u32 {
    return div_ceil(count, RREF_WORKGROUP_SIZE);
}

// Select the pivot row of the given column and store it in pivot_plus1.
// This is deliberately done by a single (leader) thread, before any of the other threads start modifying the tableau.
// The other threads run through the same loops without doing anything, see bring_into_rref.
// No rows are physically moved. Among the rows that have a set bit in the column and haven't been used as pivots yet,
// we pick the one with the greatest previous pivot column, breaking ties in favor of the bottom-most row.
fn select_pivot(col: u32, leader: bool) {
    var pivot_found = false;
    var pivot: u32 = 0;
    var m: u32 = 0;
    var set_bits: u32 = 0;
    for (var i = 0u; i < column_block_length(); i += 1) {
        var block = 0u;
        if leader {
            let column_block = tableau[x_column_block_index(i, col)];
            set_bits += countOneBits(column_block);
            block = column_block & pivot_mask[i];
        }
        // Visit the set bits in order of decreasing bit index.
        for (var k = 0u; k < BLOCK_SIZE; k += 1) {
            let bit_index = BLOCK_SIZE - 1u - k;
            if (block & bitmask(bit_index)) != 0 {
                let row = BLOCK_SIZE * i + bit_index;
                if !pivot_found || m <= row_pivots[row] {
                    pivot = row;
                    pivot_found = true;
                    m = row_pivots[row];
                }
                row_pivots[row] = col + 1;
            }
        }
    }

    if leader {
        if pivot_found {
            pivot_mask[pivot / BLOCK_SIZE] = unset_bit(pivot_mask[pivot / BLOCK_SIZE], pivot % BLOCK_SIZE);
            pivot_plus1 = pivot + 1;
            // The pivot row itself accounts for one of the set bits.
            needs_elimination = set_bits > 1;
        } else {
            pivot_plus1 = 0;
            needs_elimination = false;
        }
    }
}

// Eliminate the pivot column from all rows of the given block other than the pivot row.
// If `in_use` is false, nothing is done.
fn eliminate(block_index: u32, pivot: u32, col: u32, in_use: bool) {
    let pivot_block_index = pivot / BLOCK_SIZE;
    let pivot_bit_index = pivot % BLOCK_SIZE;

    // Bitmask blocking out the pivot row.
    var pivot_mask: BitBlock = ~0u;
    if block_index == pivot_block_index {
        pivot_mask = ~bitmask(pivot_bit_index);
    }
    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
    var mask: BitBlock = 0u;
    if in_use {
        mask = tableau[x_column_block_index(block_index, col)] & pivot_mask;
    }
    multiply_rows(block_index, pivot, mask);
}

// Multiply the pivot row into all rows of the given block selected by the mask.
// Nothing is done if the mask is empty.
fn multiply_rows(block_index: u32, pivot: u32, mask: BitBlock) {
    // Determine phase change caused by multiplication of the individual Pauli matrices.
    // We encode phase as `phase = 2*phase_bit2 + phase_bit1`,
    // but in a bit block so we can operate on all rows in the block at once.
    // Since i^phase works modulo 4, we can just use two bits and let additions/subtractions wrap around.
    var phase_bit1: BitBlock = 0u;
    var phase_bit2: BitBlock = 0u;
    for (var col2 = 0u; col2 < n; col2 += 1) {
        if mask != 0 {
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
    }
    // A valid stabilizer row can only ever have a prefix of +1 or -1.
    // phase_bit1 being 1 implies a phase of either 1 or 3, making the prefix i or -i respectively.
    // This should never be able to happen, and we cannot represent it.
    // phase_bit2 = 1  =>  phase = 2  =>  i^2 = -1    flip the sign bit.
    // phase_bit2 = 0  =>  phase = 0  =>  i^0 = +1    do nothing.
    if mask != 0 {
        tableau[r_column_block_index(block_index)] ^= phase_bit2 & mask;
    }

    // XOR
    for (var j = 0u; j < n + n + 1 + c_cols; j += 1) {
        if mask != 0 && bit(pivot, j) {
            tableau[column_block_index(block_index, j)] ^= mask;
        }
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
        unset_w_bit(state_index, qubit);
    } else {
        if w_bit(state_index, qubit) {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], -2.0 / sqrt(2.0));
        } else {
            w_coeffs[state_index] = mul_real(w_coeffs[state_index], 2.0 / sqrt(2.0));
        }
        set_w_bit(state_index, qubit);
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
        let pivot_plus1 = row_pivots[row];
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
            let pivot_plus1 = row_pivots[row2];
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
        let pivot_plus1 = row_pivots[row];
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
            let pivot_plus1 = row_pivots[row2];
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

// Assume tableau in RREF.
// Same as coeff_ratio but with w1 and s both being state_index and w2 being w1 but with the qubit'th bit flipped.
fn coeff_ratio_flipped_bit(state_index: u32) -> Complex {
    let flipped_bit = qubit;

    // Find pivot row.
    var row: u32 = 0;
    var found = false;
    for (var j: u32 = 0; j < n; j += 1) {
        if row_pivots[j] == flipped_bit + 1 {
            row = j;
            found = true;
            break;
        }
    }
    if !found {
        return Complex(0.0, 0.0);
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
            if w_bit(state_index, q) {
                res = mul_minus_i(res);
            } else {
                res = mul_i(res);
            }
        } else if z(row, q) && (flipped_bit != q) {
            if w_bit(state_index, q) {
                res = mul_minus_one(res);
            } else {
                // Multiply by 1
            }
        } else {
            // Multiply by 0
            return Complex(0.0, 0.0);
        }
    }
    return res;
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
