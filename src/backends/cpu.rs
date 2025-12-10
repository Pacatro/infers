use num_traits::{FromPrimitive, Num};
use rayon::prelude::*;
use std::{fmt::Debug, iter::Sum, ops::AddAssign};

use crate::{
    InfersResult,
    backends::{Backend, Device, GemmParams},
};

/// Represents the CPU backend.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct Cpu;

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug + Send + Sync + PartialOrd + FromPrimitive + AddAssign + Sum,
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

    fn mul(lhs: &Self::Storage, rhs: &Self::Storage, _size: usize) -> Self::Storage {
        lhs.par_iter()
            .zip(rhs.par_iter())
            .map(|(&a, &b)| a * b)
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

    fn gemm(params: GemmParams<T, Self::Storage>) -> Self::Storage {
        // See this for optimization: https://salykova.github.io/gemm-cpu
        // Also this implementation looks interesting: https://github.com/Krish120003/gemm-rust/
        let mut c = vec![T::zero(); params.m * params.n];

        for i in 0..params.m {
            for j in 0..params.n {
                let mut sum = T::zero();
                for p in 0..params.k {
                    let lhs_idx = i * params.lhs_strides[0] + p * params.lhs_strides[1];
                    let rhs_idx = p * params.rhs_strides[0] + j * params.rhs_strides[1];
                    sum += params.lhs[lhs_idx] * params.rhs[rhs_idx];
                }
                c[i * params.n + j] = params.alpha * sum + params.beta * c[i * params.n + j];
            }
        }

        c
    }

    fn dot(lhs: &Self::Storage, rhs: &Self::Storage, _size: usize) -> Self::Storage {
        let sum = lhs
            .par_iter()
            .zip(rhs.par_iter())
            .map(|(&a, &b)| a * b)
            .sum();
        vec![sum]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_read_cpu() {
        let storage = vec![1., 2., 3., 4.];
        assert_eq!(Cpu::read(&storage, 0), 1.);
    }

    #[test]
    fn test_backend_write_cpu() {
        let mut storage = vec![1., 2., 3., 4.];
        Cpu::write(&mut storage, 0, 10.);
        assert_eq!(storage, vec![10., 2., 3., 4.]);
    }

    #[test]
    fn test_backend_copy_to_host_cpu() {
        let storage = vec![1., 2., 3., 4.];
        let host_data = Cpu::copy_to_host(&storage).unwrap();
        assert_eq!(host_data, vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_backend_add_cpu() {
        let lhs = vec![1., 2., 3., 4.];
        let rhs = vec![5., 6., 7., 8.];
        let result = Cpu::add(&lhs, &rhs, 4);
        assert_eq!(result, vec![6., 8., 10., 12.]);
    }

    #[test]
    fn test_backend_sub_cpu() {
        let lhs = vec![5., 6., 7., 8.];
        let rhs = vec![1., 2., 3., 4.];
        let result = Cpu::sub(&lhs, &rhs, 4);
        assert_eq!(result, vec![4., 4., 4., 4.]);
    }

    #[test]
    fn test_backend_relu_cpu() {
        let input = vec![-1., -2., 3., 4.];
        let result = Cpu::relu(&input, 4);
        assert_eq!(result, vec![0., 0., 3., 4.]);
    }

    #[test]
    fn test_backend_gemm_cpu() {
        let lhs = vec![1., 2., 3., 4.];
        let rhs = vec![5., 6., 7., 8.];
        let lhs_strides = vec![2, 1];
        let rhs_strides = vec![2, 1];
        let result = Cpu::gemm(GemmParams {
            lhs: &lhs,
            rhs: &rhs,
            lhs_strides,
            rhs_strides,
            alpha: 1.,
            beta: 0.,
            m: 2,
            n: 2,
            k: 2,
        });
        assert_eq!(result, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn test_backend_dot_cpu() {
        let lhs = vec![1., 2., 3., 4.];
        let rhs = vec![5., 6., 7., 8.];
        let result = Cpu::dot(&lhs, &rhs, 4);
        assert_eq!(result, vec![70.]);
    }
}
