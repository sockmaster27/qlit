use std::mem;

use crate::utils::{bitmask, set_bit, unset_bit};

type BitBlock = u8;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

pub struct BitStringArray {
    string_length: usize,
    inner: Vec<BitBlock>,
}
impl BitStringArray {
    pub fn new(string_length: usize, capacity: usize) -> Self {
        let string_block_length = string_length.div_ceil(BLOCK_SIZE);
        let mut inner = Vec::with_capacity(capacity);
        inner.resize(string_block_length, 0);
        Self {
            string_length,
            inner,
        }
    }

    #[cfg(test)]
    pub fn singleton_from_u8(s: u8) -> Self {
        Self {
            string_length: 8,
            inner: vec![s],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        self.inner.len() / string_block_length
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> (usize, usize) {
        debug_assert!(j < self.string_length);
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let block_index = i * string_block_length + j / BLOCK_SIZE;
        let bit_index = j % BLOCK_SIZE;
        (block_index, bit_index)
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> bool {
        let (block_index, bit_index) = self.index(i, j);
        let bit_mask: BitBlock = bitmask(bit_index);
        (self.inner[block_index] & bit_mask) != 0
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        self.inner[block_index] = set_bit(self.inner[block_index], bit_index);
    }
    #[inline]
    pub fn unset(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        self.inner[block_index] = unset_bit(self.inner[block_index], bit_index);
    }
    #[inline]
    pub fn flip(&mut self, i: usize, j: usize) {
        let (block_index, bit_index) = self.index(i, j);
        let bit_mask: BitBlock = bitmask(bit_index);
        self.inner[block_index] ^= bit_mask;
    }
    #[inline]
    pub fn extend_from_within(&mut self, src: usize) {
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let src_start = src * string_block_length;
        self.inner
            .extend_from_within(src_start..src_start + string_block_length);
    }

    #[inline]
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.array.string_length {
            return None;
        }
        let r = self.array.get(self.i, self.j);
        self.j += 1;
        Some(r)
    }
}
