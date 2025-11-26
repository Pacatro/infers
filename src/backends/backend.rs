use std::fmt::{Debug, Display};

use num_traits::Num;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{}", id),
        }
    }
}

pub trait Backend<T>: Clone + Debug {
    type Storage: Clone + Debug;

    fn device() -> Device;

    fn init(data: &[T]) -> Self::Storage;

    fn zeros(size: usize) -> Self::Storage
    where
        T: Num + Clone;

    fn read(storage: &Self::Storage, index: usize) -> T;

    fn write(storage: &mut Self::Storage, index: usize, value: T);

    fn add(lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage
    where
        T: Num + Clone + Copy;

    // TODO: More operations
}
