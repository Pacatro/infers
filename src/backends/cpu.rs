use num_traits::Num;
use std::fmt::Debug;

use crate::backends::{Backend, Device};

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

    fn init(data: &[T]) -> Self::Storage {
        data.to_vec()
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

    fn add(lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        lhs.iter().zip(rhs.iter()).map(|(&a, &b)| a + b).collect()
    }
}
