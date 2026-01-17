@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;
alias BitBlock = u32;

@group(1) @binding(0) var<storage, read> w1: array<u32>;
@group(1) @binding(1) var<storage, read> w2: array<u32>;
@group(1) @binding(2) var<storage, read_write> factor: u32;
@group(1) @binding(3) var<storage, read_write> phase: u32;

@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Only one thread is needed.
    if id.x != 0 {
        return;
    }

    let aux_row = n;
    let aux_block_index = aux_row / block_size;
    let aux_bit_index = aux_row % block_size;
        let aux_bitmask = bitmask(aux_bit_index);

    // Reset the auxiliary row.
    for (var r: u32 = 0; r < n + n + 1; r += 1) {
        tableau[column_block_index(aux_block_index, r)] &= ~aux_bitmask;
    }
    // Derive a stabilizer with anti-diagonal Pauli matrices in the positions where w1 and w2 differ.
    var row: u32 = 0;
    for (var q: u32 = 0; q < n; q += 1) {
        if x_bit(row, q) {
            if w1[q] != w2[q] {
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
        if i(aux_row, q) && (w1[q] == w2[q]) {
            // Multiply by 1, no phase change
        } else if x(aux_row, q) && (w1[q] != w2[q]) {
            // Multiply by 1, no phase change
        } else if y(aux_row, q) && (w1[q] != w2[q]) {
            if w1[q] == 0 {
                // Multiply by i
                res_phase += 1;
            } else {
                // Multiply by -i
                res_phase += 3;
            }
        } else if z(aux_row, q) && (w1[q] == w2[q]) {
            if w1[q] == 0 {
                // Multiply by 1, no phase change
            } else {
                // Multiply by -1
                res_phase += 2;
            }
        } else {
            // Multiply by 0
            factor = 0;
            phase = 0;
            return;
        }
        res_phase %= 4;
    }
    factor = 1;
    phase = res_phase;
}

/// Set row with index `dst` to be the product of the `src` and `dst` rows.
///
/// NOTE: Since all stabilizers must commute, multiplication order is irrelevant.
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

/// Get the value of the bit corresponding to the `j`th column in the `row`th row.
fn bit(row: u32, j: u32) -> bool {
    let row_block_index = row / block_size;
    let row_bit_index = row % block_size;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[column_block_index(row_block_index, j)] & row_bitmask) != 0u;
}
/// Get the value of the x bit corresponding to the `q`th tensor element in the `row`th row.
fn x_bit(row: u32, q: u32) -> bool {
    return bit(row, 2 * q);
}
/// Get the value of the z bit corresponding to the `q`th tensor element in the `row`th row.
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

// Get the bitmask for the i'th bit, e.g.
// bitmask(0) -> 10000000
// bitmask(1) -> 01000000
// bitmask(6) -> 00000010
fn bitmask(i: u32) -> u32 {
    return 1u << (block_size - 1u - i);
}

/// Get the index of the least significant (right-most) bit in the given block, e.g.
/// lsb_index(10000000)
///           ^0
/// lsb_index(01000000)
///            ^1
/// lsb_index(11010000)
///              ^3
fn lsb_index(block: BitBlock) -> u32 {
    let trailing_zeros = countTrailingZeros(block);
    return block_size - 1u - trailing_zeros;
}

/// Get the index of the i'th block of the `j`th column.
fn column_block_index(i: u32, j: u32) -> u32 {
    return j * column_block_length() + i;
}
/// Get the index of the i'th block of the r column.
fn r_column_block_index(i: u32) -> u32 {
    return column_block_index(i, 2 * n);
}

/// Get the block-length of the columns in the tableau.
fn column_block_length() -> u32 {
    // Make room for the auxiliary row.
    return div_ceil(n + 1, block_size);
}

// Divide a by b and round up.
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

/// Get whether the given row is negative or not, i.e. the contents of the sign bit.
fn row_negative(row: u32) -> bool {
    let row_block_index = row / block_size;
    let row_bit_index = row % block_size;
    let row_bitmask = bitmask(row_bit_index);
    return (tableau[r_column_block_index(row_block_index)] & row_bitmask) != 0;
}

/// Bit-shift the given block such that the `src`th bit is moved to the `dst`th position.
fn align_bit_to(block: BitBlock, src: u32, dst: u32) -> BitBlock {
    if dst < src {
        return block << (src - dst);
    } else {
        return block >> (dst - src);
    }
}
