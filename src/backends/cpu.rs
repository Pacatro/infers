use std::{fmt::Debug, iter::Sum, ops::AddAssign};

use num_traits::{FromPrimitive, Num};

use crate::{
    backends::{Backend, Device, GemmParams},
    tensor::{Layout, Shape},
};
use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub struct Cpu;

fn elementwise<T, F>(
    lhs: &[T],
    lhs_layout: &Layout,
    rhs: &[T],
    rhs_layout: &Layout,
    output_shape: &Shape,
    operation: F,
) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    (0..output_shape.num_elements())
        .map(|index| {
            let lhs_index = lhs_layout.physical_index_from_flat(index, output_shape);
            let rhs_index = rhs_layout.physical_index_from_flat(index, output_shape);
            operation(lhs[lhs_index], rhs[rhs_index])
        })
        .collect()
}

impl<T> Backend<T> for Cpu
where
    T: Num + Clone + Copy + Debug + Send + Sync + PartialOrd + FromPrimitive + AddAssign + Sum,
{
    type Storage = Vec<T>;

    fn device() -> Device {
        Device::Cpu
    }

    fn from_host(data: Vec<T>) -> Result<Self::Storage> {
        Ok(data)
    }

    fn read(storage: &Self::Storage, index: usize) -> Result<T> {
        Ok(storage[index])
    }

    fn to_host(storage: &Self::Storage, layout: &Layout) -> Result<Vec<T>> {
        if layout.is_contiguous() {
            return Ok(storage[..layout.shape().num_elements()].to_vec());
        }

        Ok((0..layout.shape().num_elements())
            .map(|index| {
                let physical = layout.physical_index_from_flat(index, layout.shape());
                storage[physical]
            })
            .collect())
    }

    fn add(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> Result<Self::Storage> {
        Ok(elementwise(
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output_shape,
            |a, b| a + b,
        ))
    }

    fn sub(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> Result<Self::Storage> {
        Ok(elementwise(
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output_shape,
            |a, b| a - b,
        ))
    }

    fn mul(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> Result<Self::Storage> {
        Ok(elementwise(
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output_shape,
            |a, b| a * b,
        ))
    }

    fn relu(input: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        Ok(Self::to_host(input, layout)?
            .into_iter()
            .map(|value| if value > T::zero() { value } else { T::zero() })
            .collect())
    }

    fn gemm(params: GemmParams<T, Self::Storage>) -> Result<Self::Storage> {
        let mut output = vec![T::zero(); params.m * params.n];
        let lhs_strides = params.lhs_layout.strides();
        let rhs_strides = params.rhs_layout.strides();

        for row in 0..params.m {
            for column in 0..params.n {
                let mut sum = T::zero();
                for inner in 0..params.k {
                    let lhs_index = row * lhs_strides[0] + inner * lhs_strides[1];
                    let rhs_index = inner * rhs_strides[0] + column * rhs_strides[1];
                    sum += params.lhs[lhs_index] * params.rhs[rhs_index];
                }
                output[row * params.n + column] = params.alpha * sum;
            }
        }

        Ok(output)
    }

    fn dot(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage> {
        let lhs = Self::to_host(lhs, lhs_layout)?;
        let rhs = Self::to_host(rhs, rhs_layout)?;
        Ok(vec![lhs.into_iter().zip(rhs).map(|(a, b)| a * b).sum()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_backend_supports_broadcasting() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![10.0, 20.0, 30.0];
        let lhs_layout = Layout::contiguous(Shape::new([2, 3]).unwrap());
        let rhs_layout = Layout::contiguous(Shape::new([3]).unwrap());
        let output_shape = Shape::new([2, 3]).unwrap();

        let result = Cpu::add(&lhs, &lhs_layout, &rhs, &rhs_layout, &output_shape).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}
