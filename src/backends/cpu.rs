use num_traits::{FromPrimitive, Num};
use rayon::prelude::*;
use std::{fmt::Debug, ops::AddAssign};

use crate::{
    InfersResult,
    backends::{Backend, Device},
};

/// Represents the CPU backend.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on the CPU.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cpu;

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug + Send + Sync + PartialOrd + FromPrimitive + AddAssign,
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

    fn relu(input: &Self::Storage, _size: usize) -> Self::Storage {
        input
            .par_iter()
            .map(|&x| {
                let zero = T::zero();
                if x > zero { x } else { zero }
            })
            .collect()
    }

    fn gemm(
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        alpha: T,
        beta: T,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self::Storage {
        // See this for optimization: https://salykova.github.io/gemm-cpu
        let mut c = vec![T::zero(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();

                for p in 0..k {
                    sum += lhs[i * k + p] * rhs[p * n + j];
                }

                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }

        c
    }
}
