@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<uniform> max_batches: u32;
@group(0) @binding(2) var<storage, read_write> active_batches: u32;
@group(0) @binding(3) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;
alias BitBlock = u32;


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

    if id.x == 0 {
        active_batches = 1;
    }
    
    for (var j: u32 = 0; j < n + n + 1; j += 1) {
        let b = column_block_index(block_index, j);
        tableau[b] = 0;
    }
    
    for (var i: u32 = 0; i < block_size; i += 1) {
        let b = z_column_block_index(block_index, block_index * block_size + i);
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
    let block_index = id.x;
    if block_index >= active_column_block_length() {
        return;
    }

    var p: u32 = 0;
    for (var i: u32 = 0; i < arrayLength(&gates); i += 1) {
        switch gates[i] {
            case 0: {
                // CNOT
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
                let r = r_column_block_index(block_index);
                let xa = x_column_block_index(block_index, a);
                let za = z_column_block_index(block_index, a);
                let xb = x_column_block_index(block_index, b);
                let zb = z_column_block_index(block_index, b);
                tableau[r] ^= tableau[xa] & tableau[zb] & ~(tableau[xb] ^ tableau[za]);
                tableau[za] ^= tableau[zb];
                tableau[xb] ^= tableau[xa];
            }
            case 1: {
                // CZ
                let a = qubit_params[p];
                let b = qubit_params[p + 1];
                p += 2;
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
            case 2: {
                // H
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                let temp = tableau[z];
                tableau[z] = tableau[x];
                tableau[x] = temp;
            }
            case 3: {
                // S
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                tableau[z] ^= tableau[x];
            }
            case 4: {
                // Sdg
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[z] ^= tableau[x];
                tableau[r] ^= tableau[x] & tableau[z];
            }
            case 5: {
                // X
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[z];
            }
            case 6: {
                // Y
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[x] ^ tableau[z];
            }
            case 7: {
                // Z
                let a = qubit_params[p];
                p += 1;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                tableau[r] ^= tableau[x];
            }
            default: {}
        }
    }
}


@group(1) @binding(0) var<uniform> qubit: u32;

@compute
@workgroup_size(64)
fn split_batches(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index >= active_column_block_length() {
        return;
    }

    for (var q: u32 = 0u; q < n; q += 1) {
        let x1 = x_column_block_index(block_index, q);
        let x2 = x_column_block_index(block_index + active_column_block_length(), q);
        tableau[x2] = tableau[x1];
        let z1 = z_column_block_index(block_index, q);
        let z2 = z_column_block_index(block_index + active_column_block_length(), q);
        tableau[z2] = tableau[z1];
    }
    
    let x = x_column_block_index(block_index, qubit);
    let r1 = r_column_block_index(block_index);
    let r2 = r_column_block_index(block_index + active_column_block_length());
    tableau[r2] = tableau[r1] ^ tableau[x];
}


@group(1) @binding(0) var<storage, read_write> col_in: u32;
@group(1) @binding(1) var<storage, read_write> col_out: u32;
@group(1) @binding(2) var<storage, read_write> a_in: u32;
@group(1) @binding(3) var<storage, read_write> a_out: u32;
@group(1) @binding(4) var<storage, read_write> pivot_out: u32;

@compute
@workgroup_size(1)
fn init_bring_into_rref(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Only one thread is needed.
    if id.x != 0 {
        return;
    }
    
    col_in = 0;
    a_in = 0;
}

@compute
@workgroup_size(64)
fn elimination_pass(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index >= active_column_block_length() {
        return;
    }
    let batch_index = block_index / single_column_block_length();
    let batch_start_block = batch_index * single_column_block_length();
    let batch_start_row = batch_start_block * block_size;

    if id.x == 0 {
        col_out = col_in + 1;
    }

    let aux_row = batch_start_row + n;
    let aux_block_index = aux_row / block_size;
    let aux_bit_index = aux_row % block_size;

    // Find pivot row.
    var pivot_found = false;
    var pivot: u32 = 0;
    let a_block_index = batch_start_block + (a_in / block_size);
    for (var i = a_block_index; i < batch_start_block + single_column_block_length(); i += 1) {
        // Bitmask blocking out the auxiliary row.
        var aux_mask: BitBlock = ~0u;
        if i == aux_block_index {
            aux_mask = ~bitmask(aux_bit_index);
        }
        let block = tableau[x_column_block_index(i, col_in)] & aux_mask;
        if block != 0 {
            let row = block_size * i + lsb_index(block);
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

    let pivot_block_index = pivot / block_size;
    let pivot_bit_index = pivot % block_size;

    // Bitmask blocking out the pivot row.
    var pivot_mask: BitBlock = ~0u;
    if block_index == pivot_block_index {
        pivot_mask = ~bitmask(pivot_bit_index);
    }
    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
    let mask = tableau[x_column_block_index(block_index, col_in)] & pivot_mask;
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
    for (var j = 0u; j < n + n + 1; j += 1) {
        if bit(pivot, j) == true {
            tableau[column_block_index(block_index, j)] ^= mask;
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
    let batch_start_block = batch_index * single_column_block_length();

    // Swap these two rows.
    let row1 = a_in;
    let row2 = pivot_out;

    let row1_block_index = batch_start_block + (row1 / block_size);
    let row2_block_index = batch_start_block + (row2 / block_size);
    let row1_bit_index = row1 % block_size;
    let row2_bit_index = row2 % block_size;
    let row1_bitmask = bitmask(row1_bit_index);
    let row2_bitmask = bitmask(row2_bit_index);

    for (var j: u32 = 0; j < n + n + 1; j += 1) {
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
}


@group(1) @binding(0) var<storage, read> w1s: array<u32>;
@group(1) @binding(1) var<uniform> flipped_bit: u32;        // only used in coeff_ratio_flipped_bit
@group(1) @binding(1) var<storage, read> w2: array<u32>;    // only used in coeff_ratio
@group(1) @binding(2) var<storage, read_write> factors: array<u32>;
@group(1) @binding(3) var<storage, read_write> phases: array<u32>;

fn w1(batch_index: u32, q: u32) -> u32 {
    return w1s[batch_index * n + q];
}

@compute
@workgroup_size(1)
fn coeff_ratios_flipped_bit(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each batch.
    let batch_index = id.x;
    if batch_index >= active_batches {
        return;
    }
    let batch_start_row = batch_index * single_column_block_length() * block_size;

    // Find pivot row.
    var row: u32 = 0;
    for (var j: u32 = 0; j < n; j += 1) {
        if x_bit(batch_start_row + j, flipped_bit) {
            row = batch_start_row + j;
            break;
        }
    }

    var res_phase: u32 = 0; // i^0 = 1
    if row_negative(row) {
        res_phase = 2; // i^2 = -1
    } 
    for (var q: u32 = 0; q < n; q += 1) {
        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        if i(row, q) && (flipped_bit != q) {
            // Multiply by 1, no phase change
        } else if x(row, q) && (flipped_bit == q) {
            // Multiply by 1, no phase change
        } else if y(row, q) && (flipped_bit == q) {
            if w1(batch_index, q) == 0 {
                // Multiply by i
                res_phase += 1;
            } else {
                // Multiply by -i
                res_phase += 3;
            }
        } else if z(row, q) && (flipped_bit != q) {
            if w1(batch_index, q) == 0 {
                // Multiply by 1, no phase change
            } else {
                // Multiply by -1
                res_phase += 2;
            }
        } else {
            // Multiply by 0
            factors[batch_index] = 0;
            phases[batch_index] = 0;
            return;
        }
        res_phase %= 4;
    }
    factors[batch_index] = 1;
    phases[batch_index] = res_phase;
}

@compute
@workgroup_size(1)
fn coeff_ratios(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each batch.
    let batch_index = id.x;
    if batch_index >= active_batches {
        return;
    }
    let batch_start_row = batch_index * single_column_block_length() * block_size;

    let aux_row = batch_start_row + n;
    let aux_block_index = aux_row / block_size;
    let aux_bit_index = aux_row % block_size;
    let aux_bitmask = bitmask(aux_bit_index);

    // Reset the auxiliary row.
    for (var r: u32 = 0; r < n + n + 1; r += 1) {
        tableau[column_block_index(aux_block_index, r)] &= ~aux_bitmask;
    }
    // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
    var row: u32 = batch_start_row;
    for (var q: u32 = 0; q < n; q += 1) {
        if x_bit(row, q) {
            if w1(batch_index, q) != w2[q] {
                multiply_rows_into(row, aux_row);
            }
            row += 1;
        }
    }

    var res_phase: u32 = 0; // i^0 = 1
    if row_negative(aux_row) {
        res_phase = 2; // i^2 = -1
    } 
    for (var q: u32 = 0; q < n; q += 1) {
        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        if i(aux_row, q) && (w1(batch_index, q) == w2[q]) {
            // Multiply by 1, no phase change
        } else if x(aux_row, q) && (w1(batch_index, q) != w2[q]) {
            // Multiply by 1, no phase change
        } else if y(aux_row, q) && (w1(batch_index, q) != w2[q]) {
            if w1(batch_index, q) == 0 {
                // Multiply by i
                res_phase += 1;
            } else {
                // Multiply by -i
                res_phase += 3;
            }
        } else if z(aux_row, q) && (w1(batch_index, q) == w2[q]) {
            if w1(batch_index, q) == 0 {
                // Multiply by 1, no phase change
            } else {
                // Multiply by -1
                res_phase += 2;
            }
        } else {
            // Multiply by 0
            factors[batch_index] = 0;
            phases[batch_index] = 0;
            return;
        }
        res_phase %= 4;
    }
    factors[batch_index] = 1;
    phases[batch_index] = res_phase;
}

// Set row with index `dst` to be the product of the `src` and `dst` rows.
//
// NOTE: Since all stabilizers must commute, multiplication order is irrelevant.
fn multiply_rows_into(src: u32, dst: u32) {
    let src_block_index = src / block_size;
    let dst_block_index = dst / block_size;
    let src_bit_index = src % block_size;
    let dst_bit_index = dst % block_size;
    let src_bitmask = bitmask(src_bit_index);
    let dst_bitmask = bitmask(dst_bit_index);

    // Determine phase shift.
    var phase: u32 = 0;
    for (var q: u32 = 0; q < n; q += 1) {
        if x(src, q) && y(dst, q) {
            phase += 1;
        } else if x(src, q) && z(dst, q) {
            phase += 3;
        } else if y(src, q) && z(dst, q) {
            phase += 1;
        } else if y(src, q) && x(dst, q) {
            phase += 3;
        } else if z(src, q) && x(dst, q) {
            phase += 1;
        } else if z(src, q) && y(dst, q) {
            phase += 3;
        }
        phase %= 4;
    }
    if phase == 2 {
        // Negate the sign bit.
        tableau[r_column_block_index(dst_block_index)] ^= dst_bitmask;
    }
    // phase == 0 implies multiplication by 1, do nothing.
    // Any other phase should be impossible since stabilizers must commute.

    // XOR
    for (var j: u32 = 0; j < n + n + 1; j += 1) {
        let src_block = column_block_index(src_block_index, j);
        let dst_block = column_block_index(dst_block_index, j);
        tableau[dst_block] ^= align_bit_to(
            tableau[src_block] & src_bitmask,
            src_bit_index,
            dst_bit_index,
        );
    }
}


// Get whether the given row is negative or not, i.e. the contents of the sign bit.
fn row_negative(row: u32) -> bool {
    let row_block_index = row / block_size;
    let row_bit_index = row % block_size;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[r_column_block_index(row_block_index)] & row_bitmask) != 0;
}

// Get the index of the i'th block of the `j`th column.
fn column_block_index(i: u32, j: u32) -> u32 {
    return j * column_block_length() + i;
}
// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
fn x_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q);
}
// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
fn z_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q + 1);
}
// Get the index of the i'th block of the r column.
fn r_column_block_index(i: u32) -> u32 {
    return column_block_index(i, 2 * n);
}

// Get the block-length of the columns in a single tableau batch.
fn single_column_block_length() -> u32 {
    // Make room for the auxiliary row.
    return div_ceil(n + 1, block_size);
}
// Get the block-length of the columns of all the combined tableau batches.
fn column_block_length() -> u32 {
    // Make room for the auxiliary row.
    return single_column_block_length() * max_batches;
}
// Get the block-length of the combined active tableau batches.
fn active_column_block_length() -> u32 {
    // Make room for the auxiliary row.
    return single_column_block_length() * active_batches;
}

// Get the value of the bit corresponding to the `j`th column in the `row`th row.
fn bit(row: u32, j: u32) -> bool {
    let row_block_index = row / block_size;
    let row_bit_index = row % block_size;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[column_block_index(row_block_index, j)] & row_bitmask) != 0u;
}
// Get the value of the x bit corresponding to the `q`th tensor element in the `row`th row.
fn x_bit(row: u32, q: u32) -> bool {
    return bit(row, 2 * q);
}
// Get the value of the z bit corresponding to the `q`th tensor element in the `row`th row.
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
fn bitmask(i: u32) -> u32 {
    return 1u << (block_size - 1u - i);
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
    return block_size - 1u - trailing_zeros;
}

// Set the i'th bit of the given block, i.e. set the bit to 1.
// set_bit(00000000, 0) -> 10000000
// set_bit(10001000, 1) -> 11001000
// set_bit(10011000, 6) -> 10011010
fn set_bit(b: BitBlock, i: u32) -> BitBlock {
    return b | bitmask(i);
}
// Unset the i'th bit of the given block, i.e. set the bit to 0.
// set_bit(11111111, 0) -> 01111111
// set_bit(01110111, 1) -> 00110111
// set_bit(01100111, 6) -> 01100101
fn unset_bit(b: BitBlock, i: u32) -> BitBlock {
    return b & ~bitmask(i);
}

// Divide a by b and round up.
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}
