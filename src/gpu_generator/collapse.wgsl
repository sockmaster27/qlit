@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read_write> tableau: array<u32>; 
const block_size: u32 = 32;

// The index of the qubit we wish to collapse.
@group(1) @binding(0) var<uniform> a: u32;
// The value we wish to collapse the qubit to, (1 or 0).
@group(1) @binding(1) var<uniform> w: u32;
// Is set to 1 if the measurement was not deterministic.
@group(1) @binding(2) var<storage, read_write> nondeterministic: atomic<u32>;

// Collapse the a'th qubit to the given w value, if possible.
// NOTE: This does not update the tableau 100% correctly.
//       The resulting state is only meant to be used for collapsing following qubits,
//       or determining whether the measurement was deterministically impossible.
@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    // Assign one thread to each column.
    // We can ignore the z columns, such that q in [0, n) is an x column and q=n is the r column.
    let q = id.x + a;
    if q >= n + 1 {
        return;
    }
    let is_r_col = q == n;
    
    for (var k_block_index = 0u; k_block_index < div_ceil(n, block_size); k_block_index += 1) {
        let xk_block_index = x_column_block_index(k_block_index, a);
        let contains_set_bit = tableau[xk_block_index] != 0;
        if contains_set_bit {
            let k_bit_index = msb_index(tableau[xk_block_index]);
            let k_bitmask = bitmask(k_bit_index);

            // If a k was found the reading was non-deterministic.
            atomicStore(&nondeterministic, 1u);

            // The block containing the k'th stabilizer row.
            var kp_block_index: u32;
            // The block containing the k'th destabilizer row.
            var kd_block_index: u32;
            // The index of the first block of the q'th column.
            // All other blocks in the same column follow this one.
            var q_first_block: u32;
            if is_r_col {
                kp_block_index = r_column_block_index(k_block_index);
                kd_block_index = r_column_block_index(div_ceil(n, block_size) + k_block_index);
                q_first_block = r_column_block_index(0);
            } else {
                kp_block_index = x_column_block_index(k_block_index, q);
                kd_block_index = x_column_block_index(div_ceil(n, block_size) + k_block_index, q);
                q_first_block = x_column_block_index(0, q);
            } 

            // For all l!=k, where Z[a]Pl = -PlZ[a] (or Z[a]Dl = -DlZ[a]) set Pl to PlPk (or Dl to DlPk).
            // This column is only updated if the corresponding bit is set in the k'th row.
            let k_bit_set = (tableau[kp_block_index] & k_bitmask) != 0;
            if k_bit_set {
                for (var l_block_index = k_block_index; l_block_index < column_block_length(); l_block_index += 1) {
                    tableau[q_first_block + l_block_index] ^= tableau[x_column_block_index(l_block_index, a)];
                }
            }

            // Set Dk to Pk.
            if k_bit_set {
                tableau[kd_block_index] = set_bit(tableau[kd_block_index], k_bit_index);
            } else {
                tableau[kd_block_index] = unset_bit(tableau[kd_block_index], k_bit_index);
            }

            // Set Pk to Z[a] or -Z[a].
            // The x bits are already set to zero (besides the a'th one but we assume that we won't look at that anymore).
            // We can ignore the z bits.
            if is_r_col {
                if w == 1 {
                    tableau[kp_block_index] = set_bit(tableau[r_column_block_index(k_block_index)], k_bit_index);
                } else {
                    tableau[kp_block_index] = unset_bit(tableau[r_column_block_index(k_block_index)], k_bit_index);
                }
            }

            return;
        }
    }
}

// Get the bitmask for the i'th bit, e.g.
// bitmask(0) -> 10000000
// bitmask(1) -> 01000000
// bitmask(6) -> 00000010
fn bitmask(i: u32) -> u32 {
    return 1u << (block_size - 1u - i);
}

// Get the index of the most significant (left-most) bit in the given block, e.g.
// msb_index(10010100)
//           ^0
// msb_index(01100001)
//            ^1
// msb_index(00000010)
//                 ^6
fn msb_index(b: u32) -> u32 {
    return countLeadingZeros(b);
}

// Set the i'th bit of the given block, i.e. set the bit to 1.
// set_bit(00000000, 0) -> 10000000
// set_bit(10001000, 1) -> 11001000
// set_bit(10011000, 6) -> 10011010
fn set_bit(b: u32, i: u32) -> u32 {
    return b | bitmask(i);
}

// Unset the i'th bit of the given block, i.e. set the bit to 0.
// set_bit(11111111, 0) -> 01111111
// set_bit(01110111, 1) -> 00110111
// set_bit(01100111, 6) -> 01100101
fn unset_bit(b: u32, i: u32) -> u32 {
    return b & ~bitmask(i);
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
