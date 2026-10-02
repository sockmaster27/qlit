use std::mem;

use crate::utils::{bitmask, set_bit, unset_bit};

type BitBlock = u8;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

pub struct BitString {
    length: usize,
    inner: Vec<BitBlock>,
}
impl BitString {
    pub fn zero(length: usize) -> Self {
        let block_length = length.div_ceil(BLOCK_SIZE);
        Self {
            length,
            inner: vec![0; block_length],
        }
    }
    pub fn from_u8_ltr(s: u8) -> Self {
        Self {
            length: 8,
            inner: vec![s],
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    fn index(&self, i: usize) -> (usize, usize) {
        let len = self.length;
        debug_assert!(
            i < len,
            "index out of bounds: the len is {len} but the index is {i}"
        );
        let block_index = i / BLOCK_SIZE;
        let bit_index = i % BLOCK_SIZE;
        (block_index, bit_index)
    }

    pub fn get(&self, i: usize) -> bool {
        let (block_index, bit_index) = self.index(i);
        let bit_mask: BitBlock = bitmask(bit_index);
        (self.inner[block_index] & bit_mask) != 0
    }
    pub fn set(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        self.inner[block_index] = set_bit(self.inner[block_index], bit_index);
    }
    pub fn unset(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        self.inner[block_index] = unset_bit(self.inner[block_index], bit_index);
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> {
        BitStringIter {
            length: self.length,
            inner: &self.inner,
            i: 0,
        }
    }
}
impl From<&[bool]> for BitString {
    fn from(value: &[bool]) -> Self {
        let mut r = Self::zero(value.len());
        for (i, &b) in value.iter().enumerate() {
            if b {
                r.set(i);
            }
        }
        r
    }
}

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
    pub fn singleton_from_u8_ltr(s: u8) -> Self {
        Self {
            string_length: 8,
            inner: vec![s],
        }
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
    pub fn copy_within(&mut self, src: usize, dst: usize) {
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let src_start = src * string_block_length;
        let dst_start = dst * string_block_length;
        self.inner
            .copy_within(src_start..src_start + string_block_length, dst_start);
    }

    #[inline]
    pub fn iter_string<'a>(&'a self, i: usize) -> impl Iterator<Item = bool> + 'a {
        let string_block_length = self.string_length / BLOCK_SIZE;
        let start = i * string_block_length;
        BitStringIter {
            length: self.string_length,
            inner: &self.inner[start..],
            i: 0,
        }
    }
}

pub struct BitStringIter<'a> {
    length: usize,
    inner: &'a [BitBlock],
    i: usize,
}
impl<'a> Iterator for BitStringIter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i >= self.length {
            return None;
        }
        let block_index = i / BLOCK_SIZE;
        let bit_index = i % BLOCK_SIZE;
        let bit_mask: BitBlock = bitmask(bit_index);
        let r = (self.inner[block_index] & bit_mask) != 0;
        self.i += 1;
        Some(r)
    }
}
