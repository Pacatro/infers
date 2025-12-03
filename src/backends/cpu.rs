use num_traits::{FromPrimitive, Num};
use rayon::prelude::*;
use std::fmt::Debug;

use crate::{
    InfersResult,
    backends::{Backend, Device},
    tensor::Tensor,
};

/// Represents the CPU backend.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on the CPU.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cpu;

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug + Send + Sync + PartialOrd + FromPrimitive,
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

    fn gemm(lhs: &Tensor<Self, T>, rhs: &Tensor<Self, T>, alpha: T, beta: T) -> Tensor<Self, T> {
        let m = lhs.shape[0];
        let k = lhs.shape[1];
        let n = rhs.shape[1];

        assert_eq!(
            k, rhs.shape[0],
            "mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})",
            m, k, k, n
        );

        let mut c = Tensor::<Self, T>::zeros(&[m, n]);
        let zero = T::zero();

        for i in 0..m {
            for j in 0..n {
                let mut sum = zero;

                for p in 0..k {
                    let a_ip = lhs.get(&[i, p]);
                    let b_pj = rhs.get(&[p, j]);
                    sum = sum + a_ip * b_pj;
                }
                let c_old = if beta != zero { c.get(&[i, j]) } else { zero };
                c.set(&[i, j], alpha * sum + beta * c_old);
            }
        }

        c
    }
}
