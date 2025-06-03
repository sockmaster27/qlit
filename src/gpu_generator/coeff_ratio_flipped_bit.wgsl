@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;
alias BitBlock = u32;

@group(1) @binding(0) var<storage> w1: array<u32>;
@group(1) @binding(1) var<uniform> flipped_bit: u32;
@group(1) @binding(2) var<uniform> flipped_bit: u32;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index > column_block_length() {
        return;
    }

    let aux_row = n;
    let aux_block_index = aux_row / block_size;
    let aux_bit_index = aux_row % block_size;

    // Find pivot row.
    var row = 0;
    for (var j = 0; j < n; j += 1) {
        if x_bit(j, flipped_bit) == true {
            row = j;
            break;
        }
    }
            
    var res;
    if self.row_negative(row) {
        
    } else {
        Complex::ONE
    };
    for q in 0..n {
        // Note that we're indexing into the matrix at position P[w2, w1] (w2 and w1 are reversed).
        res *= match (
            self.tensor_element(row, q),
            w1.next().unwrap().borrow(),
            w2.next().unwrap().borrow(),
        ) {
            (Pauli::I, false, false) => Complex::ONE,
            (Pauli::I, true, true) => Complex::ONE,

            (Pauli::X, false, true) => Complex::ONE,
            (Pauli::X, true, false) => Complex::ONE,

            (Pauli::Y, false, true) => Complex::I,
            (Pauli::Y, true, false) => -Complex::I,

            (Pauli::Z, false, false) => Complex::ONE,
            (Pauli::Z, true, true) => -Complex::ONE,

            _ => return Complex::ZERO,
        };
        }
}

fn x(x: BitBlock, z: BitBlock) -> BitBlock {
    return x & ~z;
}
fn z(x: BitBlock, z: BitBlock) -> BitBlock {
    return ~x & z;
}
fn y(x: BitBlock, z: BitBlock) -> BitBlock {
    return x & z;
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
/// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
fn x_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q);
}
/// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
fn z_column_block_index(i: u32, q: u32) -> u32 {
    return column_block_index(i, 2 * q + 1);
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
