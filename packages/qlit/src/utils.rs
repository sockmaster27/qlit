use std::{
    mem,
    ops::{BitAndAssign, ShlAssign},
};

use num_traits::PrimInt;

pub trait BitBlock: PrimInt + ShlAssign<usize> + BitAndAssign {
    const SIZE: usize = mem::size_of::<Self>() * 8;
}
impl<B: PrimInt + ShlAssign<usize> + BitAndAssign> BitBlock for B {}

/// Convert the 8 bits to a vector of 8 booleans, e.g.
/// ```text
/// bits_to_bools(10010110) -> [true, false, false, true, false, true, true, false]
/// ```
#[cfg(test)]
pub fn bits_to_bools(bits: u8) -> Vec<bool> {
    (0..8).map(|b| bits & (0b1000_0000 >> b) != 0).collect()
}

/// Get an iterator over the indices of the set bits in the given block, e.g.
/// ```text
/// bit_indices(10000000) -> [0]
///             ^
/// bit_indices(00000001) -> [7]
///                    ^
/// bit_indices(01101000) -> [1, 2, 4]
///              ^^ ^
/// ```
#[inline]
pub fn bit_indices(block: impl BitBlock) -> impl Iterator<Item = usize> {
    SetBitIndexIterator { block, offset: 0 }
}
struct SetBitIndexIterator<B: BitBlock> {
    block: B,
    offset: usize,
}
impl<B: BitBlock> Iterator for SetBitIndexIterator<B> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.block == B::zero() {
            return None;
        }
        let leading_zeros: usize = self.block.leading_zeros().try_into().unwrap();
        self.block <<= leading_zeros;
        self.block <<= 1;
        self.offset += leading_zeros + 1;
        Some(self.offset - 1)
    }
}

/// Get the bitmask for the i'th bit, e.g.
/// ```text
/// bitmask(0) -> 10000000
/// bitmask(1) -> 01000000
/// bitmask(6) -> 00000010
/// ```
#[inline]
pub fn bitmask<B: BitBlock>(i: usize) -> B {
    debug_assert!(i < B::SIZE);
    B::one() << (B::SIZE - 1 - i)
}

/// Set the i'th bit of the given block, i.e. set the bit to 1.
/// ```text
/// set_bit(00000000, 0) -> 10000000
/// set_bit(00110111, 1) -> 01110111
/// set_bit(01100101, 6) -> 01100111
/// ```
#[inline]
pub fn set_bit<B: BitBlock>(block: B, i: usize) -> B {
    debug_assert!(i < B::SIZE);
    block | bitmask::<B>(i)
}

/// Unset the i'th bit of the given block, i.e. set the bit to 0.
/// ```text
/// unset_bit(11111111, 0) -> 01111111
/// unset_bit(01110111, 1) -> 00110111
/// unset_bit(01100111, 6) -> 01100101
/// ```
#[inline]
pub fn unset_bit<B: BitBlock>(block: B, i: usize) -> B {
    debug_assert!(i < B::SIZE);
    block & !bitmask::<B>(i)
}

/// Flips the i'th bit of the given block.
/// ```text
/// flip_bit(11111111, 0) -> 01111111
/// flip_bit(00110111, 1) -> 01110111
/// flip_bit(01100111, 6) -> 01100101
/// ```
#[inline]
pub fn flip_bit<B: BitBlock>(block: B, i: usize) -> B {
    debug_assert!(i < B::SIZE);
    block ^ bitmask::<B>(i)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bit_indices() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000;
        let indices: Vec<usize> = bit_indices(block).collect();
        assert_eq!(indices, vec![0, 2, 4, 32, 34, 36]);
    }
    #[test]
    fn test_bit_indices2() {
        let block: u64 =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001;
        let indices: Vec<usize> = bit_indices(block).collect();
        assert_eq!(indices, vec![63]);
    }

    #[test]
    fn test_bitmask1() {
        let res: u64 = bitmask(36);
        assert_eq!(
            res,
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_1000_0000_0000_0000_0000_0000_0000
        );
    }
    #[test]
    fn test_bitmask2() {
        let res: u64 = bitmask(63);
        assert_eq!(
            res,
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001
        );
    }

    #[test]
    fn test_set_bit1() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1000_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            set_bit(block, 34),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000
        );
    }
    #[test]
    fn test_set_bit2() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1011_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            set_bit(block, 35),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1011_1000_0000_0000_0000_0000_0000_0000
        );
    }

    #[test]
    fn test_unset_bit1() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            unset_bit(block, 34),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1000_1000_0000_0000_0000_0000_0000_0000
        );
    }
    #[test]
    fn test_unset_bit2() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            unset_bit(block, 35),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000
        );
    }

    #[test]
    fn test_flip_bit1() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            flip_bit(block, 34),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1000_1000_0000_0000_0000_0000_0000_0000
        );
    }
    #[test]
    fn test_flip_bit2() {
        let block: u64 =
            0b1010_1000_0000_0000_0000_0000_0000_0000_1010_1000_0000_0000_0000_0000_0000_0000;
        assert_eq!(
            flip_bit(block, 35),
            0b1010_1000_0000_0000_0000_0000_0000_0000_1011_1000_0000_0000_0000_0000_0000_0000
        );
    }
}
