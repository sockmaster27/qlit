use std::mem;

use crate::utils::{bitmask, set_bit, unset_bit};

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

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
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

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        let (block_index, bit_index) = self.index(i);
        let bit_mask: BitBlock = bitmask(bit_index);
        (self.inner[block_index] & bit_mask) != 0
    }
    #[inline]
    pub fn set(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        self.inner[block_index] = set_bit(self.inner[block_index], bit_index);
    }
    #[inline]
    pub fn unset(&mut self, i: usize) {
        let (block_index, bit_index) = self.index(i);
        self.inner[block_index] = unset_bit(self.inner[block_index], bit_index);
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = bool> {
        BitStringIter::new(self.length, &self.inner)
    }
}
impl<T: AsRef<[bool]>> From<T> for BitString {
    #[inline]
    fn from(value: T) -> Self {
        let slice = value.as_ref();
        let mut r = Self::zero(slice.len());
        for (i, &b) in slice.iter().enumerate() {
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
        let string_block_length = self.string_length.div_ceil(BLOCK_SIZE);
        let start = i * string_block_length;
        BitStringIter::new(self.string_length, &self.inner[start..])
    }
}

pub struct BitStringIter<'a> {
    remaining: usize,
    mask: BitBlock,
    data: &'a [BitBlock],
}
impl<'a> BitStringIter<'a> {
    fn new(length: usize, data: &'a [BitBlock]) -> Self {
        Self {
            remaining: length,
            mask: bitmask(0),
            data,
        }
    }
}
impl<'a> Iterator for BitStringIter<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let r = (self.data[0] & self.mask) != 0;
        self.mask >>= 1;
        if self.mask == 0 {
            self.mask = bitmask(0);
            self.data = &self.data[1..]
        }
        Some(r)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bitstring_zero() {
        let b = BitString::zero(20);
        assert_eq!(b.len(), 20);
        assert_eq!(b.iter().collect::<Vec<_>>(), vec![false; 20]);
    }

    #[test]
    fn bitstring_from_u8_ltr() {
        let b = BitString::from_u8_ltr(0b011_01001);
        assert_eq!(b.len(), 8);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![false, true, true, false, true, false, false, true]
        );
    }
    #[test]
    fn bitstring_from_u16_ltr() {
        let b = BitString::from_u16_ltr(0b0110_1001_0110_1001);
        assert_eq!(b.len(), 16);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![
                false, true, true, false, true, false, false, true, false, true, true, false, true,
                false, false, true,
            ]
        );
    }
    #[test]
    fn bitstring_from_u32_ltr() {
        let b = BitString::from_u32_ltr(0b0110_1001_0110_1001_0110_1001_0110_1001);
        assert_eq!(b.len(), 32);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![
                false, true, true, false, true, false, false, true, false, true, true, false, true,
                false, false, true, false, true, true, false, true, false, false, true, false,
                true, true, false, true, false, false, true,
            ]
        );
    }
    #[test]
    fn bitstring_from_u64_ltr() {
        let b = BitString::from_u64_ltr(
            0b0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001,
        );
        assert_eq!(b.len(), 64);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![
                false, true, true, false, true, false, false, true, false, true, true, false, true,
                false, false, true, false, true, true, false, true, false, false, true, false,
                true, true, false, true, false, false, true, false, true, true, false, true, false,
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false, false, true, false, true, true, false, true, false, false,
                true,
            ]
        );
    }
    #[test]
    fn bitstring_from_u128_ltr() {
        let b = BitString::from_u128_ltr(
            0b0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001_0110_1001,
        );
        assert_eq!(b.len(), 128);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![
                false, true, true, false, true, false, false, true, false, true, true, false, true,
                false, false, true, false, true, true, false, true, false, false, true, false,
                true, true, false, true, false, false, true, false, true, true, false, true, false,
                false, true, false, true, true, false, true, false, false, true, false, true, true,
                false, true, false, false, true, false, true, true, false, true, false, false,
                true, false, true, true, false, true, false, false, true, false, true, true, false,
                true, false, false, true, false, true, true, false, true, false, false, true,
                false, true, true, false, true, false, false, true, false, true, true, false, true,
                false, false, true, false, true, true, false, true, false, false, true, false,
                true, true, false, true, false, false, true, false, true, true, false, true, false,
                false, true,
            ]
        );
    }

    #[test]
    fn bitstring_from_bool_vec() {
        let v = vec![
            false, true, true, false, true, false, false, true, false, false,
        ];
        let b: BitString = From::from(&v);
        assert_eq!(b.len(), 10);
        assert_eq!(b.iter().collect::<Vec<_>>(), v);
    }

    #[test]
    fn bitstring_get1() {
        let b = BitString::from_u32_ltr(0b0000_0000_0000_0000_0100_0000_0000_0000);
        assert_eq!(b.get(17), true);
    }
    #[test]
    fn bitstring_get2() {
        let b = BitString::from_u32_ltr(0b0000_0000_0000_0000_0100_0000_0000_0000);
        assert_eq!(b.get(16), false);
    }

    #[test]
    fn bitstring_set1() {
        let mut b = BitString::from([false; 33]);
        b.set(32);
        assert_eq!(b.get(32), true);
    }
    #[test]
    fn bitstring_set2() {
        let mut b = BitString::from([true; 33]);
        b.set(32);
        assert_eq!(b.get(32), true);
    }

    #[test]
    fn bitstring_unset1() {
        let mut b = BitString::from([true; 33]);
        b.unset(7);
        assert_eq!(b.get(7), false);
    }
    #[test]
    fn bitstring_unset2() {
        let mut b = BitString::from([false; 33]);
        b.unset(7);
        assert_eq!(b.get(7), false);
    }
}
