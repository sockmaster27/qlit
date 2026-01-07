#[derive(Clone, Debug)]
pub struct Path {
    // TODO: Support more than 63 bits (?)
    bits: u64,
    len: usize,
}
impl Path {
    pub fn zero(len: usize) -> Self {
        Path { bits: 0, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn bit(&self, index: usize) -> bool {
        self.bits & (1 << (index % 64)) != 0
    }
    pub fn incr(&mut self) -> bool {
        self.bits += 1;
        self.bits & (1 << self.len) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_0() {
        let mut path = Path::zero(0);
        assert!(path.incr());
    }

    #[test]
    fn len_1() {
        let mut path = Path::zero(1);
        assert!(!path.incr()); // 0 -> 1
        assert!(path.incr()); // 1 -> 0
    }

    #[test]
    fn len_2() {
        let mut path = Path::zero(2);
        assert!(!path.incr()); // 00 -> 01
        assert!(!path.incr()); // 01 -> 10
        assert!(!path.incr()); // 10 -> 11
        assert!(path.incr()); // 11 -> 00
    }
}
