use std::mem;
use std::{borrow::Borrow, fmt::Debug};

use rand::{RngExt, SeedableRng, rngs::Xoshiro128PlusPlus};

use crate::utils::{bitmask, flip_bit, set_bit, unset_bit};

type BitBlock = u8;
const BLOCK_SIZE: usize = mem::size_of::<BitBlock>() * 8;

#[derive(Clone, PartialEq, Eq)]
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
    pub fn from_u16_ltr(s: u16) -> Self {
        Self {
            length: 16,
            inner: s.to_be_bytes().into(),
        }
    }
    pub fn from_u32_ltr(s: u32) -> Self {
        Self {
            length: 32,
            inner: s.to_be_bytes().into(),
        }
    }
    pub fn from_u64_ltr(s: u64) -> Self {
        Self {
            length: 64,
            inner: s.to_be_bytes().into(),
        }
    }
    pub fn from_u128_ltr(s: u128) -> Self {
        Self {
            length: 128,
            inner: s.to_be_bytes().into(),
        }
    }
    pub fn random(length: usize, seed: u64) -> Self {
        // Xoshiro128++ (unlike SmallRng) is portable across platforms, so a given
        // seed always produces the same circuit regardless of pointer width.
        // Useful for CI testing and benchmarking.
        let rng = Xoshiro128PlusPlus::seed_from_u64(seed);
        Self {
            length,
            inner: rng
                .random_iter()
                .take(length.div_ceil(BLOCK_SIZE))
                .collect(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns (block_index, bit_index) pair for the i'th bit in the string.
    ///
    /// # Panics
    /// If `i >= self.length`. It can therefore be trusted that the returned block_index is inside the bounds of 'self.inner'.
    #[inline]
    fn index(&self, i: usize) -> (usize, usize) {
        let len = self.length;
        assert!(
            i < len,
            "index out of bounds: the len is {len} but the index is {i}"
        );
        let block_index = i / BLOCK_SIZE;
        let bit_index = i % BLOCK_SIZE;
        (block_index, bit_index)
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        let (block_index, bit_index) = self.index(i);
        let bit_mask: BitBlock = bitmask(bit_index);
        // SAFETY: Bounds check performed by self.index
        unsafe { (self.inner.get_unchecked(block_index) & bit_mask) != 0 }
    }
    #[inline]
    pub fn set(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        // SAFETY: Bounds check performed by self.index
        unsafe {
            *self.inner.get_unchecked_mut(block_index) =
                set_bit(*self.inner.get_unchecked(block_index), bit_index);
        }
    }
    #[inline]
    pub fn unset(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        // SAFETY: Bounds check performed by self.index
        unsafe {
            *self.inner.get_unchecked_mut(block_index) =
                unset_bit(*self.inner.get_unchecked(block_index), bit_index);
        }
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = bool> {
        BitStringIter { inner: &self, i: 0 }
    }
}
impl<B: Borrow<bool>> FromIterator<B> for BitString {
    fn from_iter<T: IntoIterator<Item = B>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (s, _) = iter.size_hint();
        let mut inner = Vec::with_capacity(s.div_ceil(BLOCK_SIZE));
        let mut length = 0;
        let mut block = 0;
        let mut bit_index = 0;
        for b in iter {
            length += 1;
            if *b.borrow() {
                block = set_bit(block, bit_index);
            }
            bit_index += 1;
            if bit_index == BLOCK_SIZE {
                inner.push(block);
                block = 0;
                bit_index = 0;
            }
        }
        if bit_index != 0 {
            inner.push(block);
        }
        inner.shrink_to_fit();
        Self { length, inner }
    }
}
impl<T: AsRef<[bool]>> From<T> for BitString {
    #[inline]
    fn from(value: T) -> Self {
        Self::from_iter(value.as_ref())
    }
}
impl IntoIterator for BitString {
    type Item = bool;

    type IntoIter = BitStringIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        Vec::into_iter(vec![true]);
        BitStringIntoIter { inner: self, i: 0 }
    }
}

#[derive(Clone)]
pub struct BitStringIter<'a> {
    inner: &'a BitString,
    i: usize,
}
impl<'a> Iterator for BitStringIter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.inner.length {
            return None;
        }
        let r = self.inner.get(self.i);
        self.i += 1;
        Some(r)
    }
}
#[derive(Clone)]
pub struct BitStringIntoIter {
    inner: BitString,
    i: usize,
}
impl Iterator for BitStringIntoIter {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.inner.length {
            return None;
        }
        let r = self.inner.get(self.i);
        self.i += 1;
        Some(r)
    }
}

#[derive(PartialEq, Eq)]
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
    pub fn from_u8s_ltr(s: &[u8]) -> Self {
        Self {
            string_length: 8,
            inner: s.to_owned(),
        }
    }
    #[cfg(test)]
    pub fn from_u16s_ltr(s: &[u16]) -> Self {
        Self {
            string_length: 16,
            inner: s.iter().map(|x| x.to_be_bytes()).flatten().collect(),
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
            let block_index = i * string_block_length + j;
            let block_index_prev = (i - 1) * string_block_length + j;
            if self.inner[block_index_prev] != self.inner[block_index] {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn iter_string<'a>(&'a self, i: usize) -> BitStringArrayIter<'a> {
        BitStringArrayIter {
            inner: self,
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
    inner: &'a BitStringArray,
    i: usize,
    j: usize,
}
impl<'a> Iterator for BitStringArrayIter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.inner.string_length {
            return None;
        }
        let r = self.inner.get(self.i, self.j);
        self.j += 1;
        Some(r)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn copy_within() {
        let mut array = BitStringArray::from_u16s_ltr(&[
            0b0000_0000_0000_0000,
            0b0110_1001_1001_0110,
            0b0000_0000_0000_0000,
            0b0000_0000_0000_0000,
        ]);
        array.copy_within(1, 2);
        assert_eq!(
            array,
            BitStringArray::from_u16s_ltr(&[
                0b0000_0000_0000_0000,
                0b0110_1001_1001_0110,
                0b0110_1001_1001_0110,
                0b0000_0000_0000_0000,
            ])
        );
    }

    #[test]
    fn equal_to_previous1() {
        let array = BitStringArray::from_u16s_ltr(&[
            0b0000_0000_0000_0000,
            0b0110_1001_1001_0110,
            0b0110_1001_1001_0110,
            0b0000_0000_0000_0000,
        ]);
        assert_eq!(array.equal_to_previous(0), false);
    }
    #[test]
    fn equal_to_previous2() {
        let array = BitStringArray::from_u16s_ltr(&[
            0b0110_1001_0110_1001,
            0b0110_1001_1001_0110,
            0b0110_1001_1001_0110,
            0b0000_0000_0000_0000,
        ]);
        assert_eq!(array.equal_to_previous(1), false);
    }
    #[test]
    fn equal_to_previous3() {
        let array = BitStringArray::from_u16s_ltr(&[
            0b1001_0110_1001_0110,
            0b0110_1001_1001_0110,
            0b0110_1001_1001_0110,
            0b0000_0000_0000_0000,
        ]);
        assert_eq!(array.equal_to_previous(2), true);
    }
    #[test]
    fn equal_to_previous4() {
        let array = BitStringArray::from_u16s_ltr(&[
            0b0000_0000_0000_0000,
            0b0110_1001_1001_0110,
            0b0110_1001_1001_0110,
            0b0000_0000_0000_0000,
        ]);
        assert_eq!(array.equal_to_previous(2), true);
    }
}
