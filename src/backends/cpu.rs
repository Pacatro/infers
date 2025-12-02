use num_traits::Num;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::{
    InfersResult,
    backends::{Backend, Device},
};

/// Represents the CPU backend.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct Cpu;

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug + Send + Sync,
{
    type Storage = Vec<T>;

    fn device() -> Device {
        Device::Cpu
    }

    fn init(data: &[T]) -> InfersResult<Self::Storage> {
        Ok(data.to_vec())
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

    fn add(lhs: &Self::Storage, rhs: &Self::Storage, _size: usize) -> Self::Storage {
        lhs.par_iter()
            .zip(rhs.par_iter())
            .map(|(&a, &b)| a + b)
            .collect()
    }

    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, _size: usize) -> Self::Storage {
        lhs.par_iter()
            .zip(rhs.par_iter())
            .map(|(&a, &b)| a - b)
            .collect()
    }
}
