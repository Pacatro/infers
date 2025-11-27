use num_traits::Num;
use std::fmt::Debug;

use crate::{
    InfersResult,
    backends::{Backend, Device},
};

#[derive(Debug, Clone, Copy)]
pub struct Cpu;

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug,
{
    type Storage = Vec<T>;

    fn device() -> Device {
        Device::Cpu
    }

    fn init(data: &[T]) -> InfersResult<Self::Storage> {
        Ok(data.to_vec())
    }

    fn zeros(size: usize) -> Self::Storage {
        vec![T::zero(); size]
    }

    fn read(storage: &Self::Storage, index: usize) -> T {
        storage[index]
    }

    fn write(storage: &mut Self::Storage, index: usize, value: T) {
        storage[index] = value;
    }

    fn copy_to_host(storage: &Self::Storage) -> InfersResult<Vec<T>>
    where
        T: Num + Clone + Copy,
    {
        Ok(storage.to_vec())
    }

    fn add(lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a + b).collect()
    }
}
