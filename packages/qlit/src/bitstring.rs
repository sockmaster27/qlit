use std::mem;

type BitBlock = u8;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

pub struct BitStringArray {
    string_length: usize,
    inner: Vec<BitBlock>,
}
impl BitStringArray {
    pub fn new(string_length: usize, string_count: usize) -> Self {
        let string_block_length = string_length.div_ceil(BLOCK_SIZE);
        Self {
            string_length,
            inner: vec![0; string_block_length * string_count],
        }
    }

    fn index(&self, i: usize, j: usize) -> (usize, usize) {
        debug_assert!(j < self.string_length);
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let block_index = i * string_block_length + j / BLOCK_SIZE;
        let bit_index = j % BLOCK_SIZE;
        (block_index, bit_index)
    }

    pub fn get(&self, i: usize, j: usize) -> bool {
        let (block_index, bit_index) = self.index(i, j);
        (self.inner[block_index] & bitmask(bit_index)) != 0
    }
    pub fn set(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        self.inner[block_index] |= bitmask(bit_index);
    }
    pub fn unset(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        self.inner[block_index] &= !bitmask(bit_index);
    }
    pub fn flip(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        self.inner[block_index] ^= bitmask(bit_index);
    }
    pub fn copy_within(&mut self, src: usize, dst: usize) {
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let src_start = src * string_block_length;
        let dst_start = dst * string_block_length;
        self.inner
            .copy_within(src_start..src_start + string_block_length, dst_start);
    }

    pub fn iter_string<'a>(&'a self, i: usize) -> impl Iterator<Item = bool> + 'a {
        BitStringArrayIter {
            array: self,
            i,
            j: 0,
        }
    }
}

pub struct BitStringArrayIter<'a> {
    array: &'a BitStringArray,
    i: usize,
    j: usize,
}
impl<'a> Iterator for BitStringArrayIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.array.string_length {
            return None;
        }
        let r = self.array.get(self.i, self.j);
        self.j += 1;
        Some(r)
    }
}

/// Get the bitmask for the i'th bit, e.g.
/// ```text
/// bitmask(0) -> 10000000
/// bitmask(1) -> 01000000
/// bitmask(6) -> 00000010
/// ```
///
/// # Panics
/// If `i` is greater than or equal to `BLOCK_SIZE` in debug mode.
fn bitmask(i: usize) -> BitBlock {
    debug_assert!(i < BLOCK_SIZE);
    1 << (BLOCK_SIZE - 1 - i)
}
