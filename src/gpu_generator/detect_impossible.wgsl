@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;

// The bit-string we wish to check for.
// This is simply an array of integers being either 0 or 1.
@group(1) @binding(0) var<storage, read> w: array<u32>;
// Is set to 1 if any measurement was impossible.
@group(1) @binding(1) var<storage, read_write> impossible: atomic<u32>;

// Assuming that all qubits have been collapsed, detect if any qubits are present in a state
// indicating that it is deterministically impossible to measure the corresponding bit in w.
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each qubit.
    let a = id.x;
    if a >= n {
        return;
    }

    // The bit we're checking for.
    // true=1
    // false=0
    let b = w[a] == 1;

    // Check if stabilized by Z[a] or -Z[a].
    var negative = false;
    for (var block_index = 0u; block_index < div_ceil(n, block_size); block_index += 1) {
        // Check if there's an even number of set bits in the r column,
        // only counting rows where Z[a]Dl = -DlZ[a],
        // a.e. rows where the xa bit is 1 in the destabilizer.
        // We do this by counting the number of 1 bits in r & xa.
        let destabilized_rs = tableau[r_column_block_index(block_index)]
            & tableau[x_column_block_index(div_ceil(n, block_size) + block_index, a)];
        if countOneBits(destabilized_rs) % 2 == 1 {
            negative = !negative;
        }
    }
    // If it's the wrong one, it's impossible to measure the bit.
    if negative != b {
        atomicStore(&impossible, 1u);
    }
}

// Get the index of the a'th block of the column representing the x part of the `q`th tensor element.
// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn x_column_block_index(a: u32, q: u32) -> u32 {
    return 2 * q * column_block_length() + a;
}
// Get the index of the a'th block of the r column.
// The first half of the blocks will contain the stabilizer parts and the second half the destabilizer parts.
fn r_column_block_index(a: u32) -> u32 {
    return 2 * n * column_block_length() + a;
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
