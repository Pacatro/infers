use std::fmt::{Debug, Display};

use crate::InfersResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

pub trait Backend<T>: Clone + Debug + Copy {
    type Storage: Clone + Debug;

    fn device() -> Device;

    fn init(data: &[T]) -> InfersResult<Self::Storage>;

    fn zeros(size: usize) -> InfersResult<Self::Storage>;

    fn read(storage: &Self::Storage, index: usize) -> T;

    fn write(storage: &mut Self::Storage, index: usize, value: T);

    fn copy_to_host(storage: &Self::Storage) -> InfersResult<Vec<T>>;

    fn add(lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage;

    // TODO: More operations
}
