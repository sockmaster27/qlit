@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>;
const block_size: u32 = 32;

@group(1) @binding(0) var<uniform> a: u32;
    
// Apply the given gates to the tableau.
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

    let r = r_column_block_index(block_index);
    let x = x_column_block_index(block_index, a);
    tableau[r] ^= tableau[x];
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
