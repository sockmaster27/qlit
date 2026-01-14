@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;
alias BitBlock = u32;

@group(1) @binding(0) var<storage, read> row1: u32;
@group(1) @binding(1) var<storage, read> row2: u32;


// Swap the two rows.
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each column.
    let j = id.x;
    if j > (n + n + 1) {
        return;
    }

    let row1_block_index = row1 / block_size;
    let row2_block_index = row2 / block_size;
    let row1_bit_index = row1 % block_size;
    let row2_bit_index = row2 % block_size;
    let block1 = column_block_index(row1_block_index, j);
    let block2 = column_block_index(row2_block_index, j);
    let bit1 = (block1 & bitmask(row1_bit_index)) != 0u;
    let bit2 = (block2 & bitmask(row2_bit_index)) != 0u;
    if bit1 && !bit2 {
        tableau[block1] = unset_bit(tableau[block1], row1_bit_index);
        tableau[block2] = set_bit(tableau[block2], row2_bit_index);
    } else if !bit1 && bit2 {
        tableau[block1] = set_bit(tableau[block1], row1_bit_index);
        tableau[block2] = unset_bit(tableau[block2], row2_bit_index);
    }
}

// Get the bitmask for the i'th bit, e.g.
// bitmask(0) -> 10000000
// bitmask(1) -> 01000000
// bitmask(6) -> 00000010
fn bitmask(i: u32) -> u32 {
    return 1u << (block_size - 1u - i);
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

/// Get the index of the i'th block of the `j`th column.
fn column_block_index(i: u32, j: u32) -> u32 {
    return j * column_block_length() + i;
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
