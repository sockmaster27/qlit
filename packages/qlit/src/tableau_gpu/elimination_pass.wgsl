@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;
alias BitBlock = u32;

// The index of the column we wish to perform elimination on.
@group(1) @binding(0) var<uniform> col: u32;
@group(1) @binding(1) var<storage, read> a: u32;
@group(1) @binding(2) var<storage, read_write> a_out: u32;
@group(1) @binding(3) var<storage, read_write> pivot_out: u32;


// Perform elimination on the column with index col.
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each row (block).
    let block_index = id.x;
    if block_index >= column_block_length() {
        return;
    }

    let aux_row = n;
    let aux_block_index = aux_row / block_size;
    let aux_bit_index = aux_row % block_size;

    // Find pivot row.
    var pivot_found = false;
    var pivot: u32 = 0;
    let a_block_index = a / block_size;
    for (var i = a_block_index; i <  column_block_length(); i += 1) {
        // Bitmask blocking out the auxiliary row.
        var aux_mask: BitBlock = ~0u;
        if i == aux_block_index {
            aux_mask = ~bitmask(aux_bit_index);
        }
        let block = tableau[x_column_block_index(i, col)] & aux_mask;
        if block != 0 {
            let row = block_size * i + lsb_index(block);
            if row >= a {
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
            pivot_out = a;
            a_out = a;
        }
        return;
    }

    if id.x == 0 {
        pivot_out = pivot;
        a_out = a + 1;
    }

    let pivot_block_index = pivot / block_size;
    let pivot_bit_index = pivot % block_size;

    // Bitmask blocking out the pivot row.
    var pivot_mask: BitBlock = ~0u;
    if block_index == pivot_block_index {
        pivot_mask = ~bitmask(pivot_bit_index);
    }
    // The bitmask with a 1 in the position of all rows that should be multiplied by the pivot.
    let mask = tableau[x_column_block_index(block_index, col)] & pivot_mask;
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
        let add = (x(x1, z1) & y(x2, z2))
            | (y(x1, z1) & z(x2, z2))
            | (z(x1, z1) & x(x2, z2));
        phase_bit2 ^= add & phase_bit1;
        phase_bit1 ^= add;

        // YX = -iZ
        // ZY = -iX
        // XZ = -iY
        let sub = (y(x1, z1) & x(x2, z2))
            | (z(x1, z1) & y(x2, z2))
            | (x(x1, z1) & z(x2, z2));
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
