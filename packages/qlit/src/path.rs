use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
pub struct Path {
    // TODO: Support more than 63 bits (?)
    bits: AtomicU64,
    len: usize,
}
impl Path {
    pub fn zero(len: usize) -> Self {
        Path {
            bits: AtomicU64::new(0),
            len,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn bit(&self, index: usize) -> bool {
        self.bits.load(Ordering::Relaxed) & (1 << (index % 64)) != 0
    }
    pub fn incr(&self) -> (Self, bool) {
        let v = self.bits.fetch_add(1, Ordering::Relaxed);
        v & (1 << self.len) != 0
    }
}
