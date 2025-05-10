@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>;
const block_size: u32 = 32;

// The gates to apply to the tableau.
// The first element is the gate type.
// If this is S or H, the following element is the qubit index, a.
// If this is CNOT, the following two elements are the control and target qubit indices, a and b.
@group(1) @binding(0) var<storage, read> gates: array<u32>;

const s_gate: u32 = 0;
const h_gate: u32 = 1;
const cnot_gate: u32 = 2;
    
// Apply the given gates to the tableau.
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each (block) row.
    let block_index = id.x;
    if block_index >= column_block_length() {
        return;
    }

    var j: u32 = 0;
    while j < arrayLength(&gates) {
        switch gates[j] {
            case s_gate: {
                let a = gates[j + 1];
                j += 2;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                tableau[z] ^= tableau[x];
            }
            case h_gate: {
                let a = gates[j + 1];
                j += 2;
                let r = r_column_block_index(block_index);
                let x = x_column_block_index(block_index, a);
                let z = z_column_block_index(block_index, a);
                tableau[r] ^= tableau[x] & tableau[z];
                let temp = tableau[z];
                tableau[z] = tableau[x];
                tableau[x] = temp;

            }
            case cnot_gate: {
                let a = gates[j+ 1];
                let b = gates[j + 2];
                j += 3;
                let r = r_column_block_index(block_index);
                let xa = x_column_block_index(block_index, a);
                let za = z_column_block_index(block_index, a);
                let xb = x_column_block_index(block_index, b);
                let zb = z_column_block_index(block_index, b);
                tableau[r] ^= tableau[xa] & tableau[zb] & ~(tableau[xb] ^ tableau[za]);
                tableau[za] ^= tableau[zb];
                tableau[xb] ^= tableau[xa];
            }
            default: {
                // Unreachable
            }
        }
    }
}

// Get the index of the i'th block of the column representing the x part of the `q`th tensor element.
// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn x_column_block_index(i: u32, q: u32) -> u32 {
    return 2 * q * column_block_length() + i;
}
// Get the index of the i'th block of the column representing the z part of the `q`th tensor element.
// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn z_column_block_index(i: u32, q: u32) -> u32 {
    return (2 * q + 1) * column_block_length() + i;
}
// Get the index of the i'th block of the r column.
// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn r_column_block_index(i: u32) -> u32 {
    return 2 * n * column_block_length() + i;
}

// Get the length of each tableau column in blocks.
fn column_block_length() -> u32 {
    // The stabilizer and destabilizer parts of each
    // column takes up a whole number of bit blocks.
    return 2 * div_ceil(n, block_size);
}

// Divide a by b and round up.
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}
