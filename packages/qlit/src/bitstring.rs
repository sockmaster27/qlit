use std::fmt::Debug;
use std::mem;

use crate::utils::{bitmask, flip_bit, set_bit, unset_bit};

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

    #[cfg(test)]
    pub fn from_u8s(s: &[u8]) -> Self {
        Self {
            string_length: 8,
            inner: s.to_owned(),
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
        self.inner[block_index] = flip_bit(self.inner[block_index], bit_index);
    }
    #[inline]
    pub fn copy_within(&mut self, src: usize, dst: usize) {
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let src_start = src * string_block_length;
        let dst_start = dst * string_block_length;
        self.inner
            .copy_within(src_start..src_start + string_block_length, dst_start);
    }

    /// Get whether or not the bitstring at the i'th position of the array is identical to the one at i-1.
    /// Return false if i=0.
    pub fn equal_to_previous(&self, i: usize) -> bool {
        if i == 0 {
            return false;
        }
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        for j in 0..string_block_length {
            let block_index = i * string_block_length + j / BLOCK_SIZE;
            let block_index_prev = (i - 1) * string_block_length + j / BLOCK_SIZE;
            if self.inner[block_index_prev] != self.inner[block_index] {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn iter_string<'a>(&'a self, i: usize) -> BitStringArrayIter<'a> {
        BitStringArrayIter {
            array: self,
            i,
            j: 0,
        }
    }
}
impl Debug for BitStringArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries((0..self.len()).map(|i| {
                self.iter_string(i)
                    .map(|b| if b { '1' } else { '0' })
                    .collect::<String>()
            }))
            .finish()
    }
}

#[derive(Clone)]
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
