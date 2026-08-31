use std::{fmt::Debug, sync::Arc};

use anyhow::bail;
use num_traits::{FromPrimitive, Num};

use crate::backends::{Backend, GemmParams};
use anyhow::Result;

use super::{Layout, Shape, Tensor};

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + FromPrimitive + Clone + Copy + Debug + Send + Sync,
{
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let output_shape = self.shape().broadcast_with(rhs.shape())?;
        let storage = B::add(
            self.storage.as_ref(),
            &self.layout,
            rhs.storage.as_ref(),
            &rhs.layout,
            &output_shape,
        )?;
        Ok(Self::from_parts(storage, Layout::contiguous(output_shape)))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        let output_shape = self.shape().broadcast_with(rhs.shape())?;
        let storage = B::sub(
            self.storage.as_ref(),
            &self.layout,
            rhs.storage.as_ref(),
            &rhs.layout,
            &output_shape,
        )?;
        Ok(Self::from_parts(storage, Layout::contiguous(output_shape)))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let output_shape = self.shape().broadcast_with(rhs.shape())?;
        let storage = B::mul(
            self.storage.as_ref(),
            &self.layout,
            rhs.storage.as_ref(),
            &rhs.layout,
            &output_shape,
        )?;
        Ok(Self::from_parts(storage, Layout::contiguous(output_shape)))
    }

    pub fn dot(&self, rhs: &Self) -> Result<Self> {
        if self.rank() != 1 || rhs.rank() != 1 || self.shape() != rhs.shape() {
            bail!(
                "incompatible shapes for dot product: {:?} and {:?}",
                self.dims(),
                rhs.dims()
            );
        }

        let storage = B::dot(
            self.storage.as_ref(),
            &self.layout,
            rhs.storage.as_ref(),
            &rhs.layout,
        )?;
        Ok(Self::from_parts(
            storage,
            Layout::contiguous(Shape::new(Vec::<usize>::new())?),
        ))
    }

    pub fn relu(&self) -> Result<Self> {
        let storage = B::relu(self.storage.as_ref(), &self.layout)?;
        Ok(Self::from_parts(
            storage,
            Layout::contiguous(self.shape().clone()),
        ))
    }

    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let layout = self.layout.reshape(Shape::new(dims)?)?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout,
            _element: std::marker::PhantomData,
        })
    }

    pub fn flatten(&self, axis: usize) -> Result<Self> {
        if axis > self.rank() {
            bail!(
                "axis {axis} is invalid for a tensor of rank {}",
                self.rank()
            );
        }

        let outer = Shape::new(self.dims()[..axis].to_vec())?.num_elements();
        let inner = Shape::new(self.dims()[axis..].to_vec())?.num_elements();
        self.reshape([outer, inner])
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            layout,
            _element: std::marker::PhantomData,
        })
    }

    pub fn t(&self) -> Result<Self> {
        if self.rank() != 2 {
            bail!("axis 1 is invalid for a tensor of rank {}", self.rank());
        }
        self.transpose(0, 1)
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let storage = B::contiguous(self.storage.as_ref(), &self.layout)?;
        Ok(Self::from_parts(
            storage,
            Layout::contiguous(self.shape().clone()),
        ))
    }

    pub fn gemm(&self, rhs: &Self, alpha: Option<T>, beta: Option<T>) -> Result<Self> {
        if self.rank() == 1 && rhs.rank() == 1 {
            return self.dot(rhs);
        }

        let output_shape = self.shape().matmul_with(rhs.shape())?;
        let m = self.dims()[0];
        let k = self.dims()[1];
        let n = rhs.dims()[1];
        let storage = B::gemm(GemmParams {
            lhs: self.storage.as_ref(),
            lhs_layout: &self.layout,
            rhs: rhs.storage.as_ref(),
            rhs_layout: &rhs.layout,
            alpha: alpha.unwrap_or_else(T::one),
            beta: beta.unwrap_or_else(T::zero),
            m,
            n,
            k,
        })?;

        Ok(Self::from_parts(storage, Layout::contiguous(output_shape)))
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tensor, backends::Cpu};

    #[test]
    fn elementwise_operations_return_valid_tensors() {
        let lhs = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let rhs = Tensor::<Cpu>::new(&[5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();

        assert_eq!(
            lhs.add(&rhs).unwrap().data().unwrap(),
            [6.0, 8.0, 10.0, 12.0]
        );
        assert_eq!(
            lhs.sub(&rhs).unwrap().data().unwrap(),
            [-4.0, -4.0, -4.0, -4.0]
        );
        assert_eq!(
            lhs.mul(&rhs).unwrap().data().unwrap(),
            [5.0, 12.0, 21.0, 32.0]
        );
    }

    #[test]
    fn add_supports_broadcasting() {
        let lhs = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let rhs = Tensor::<Cpu>::new(&[10.0, 20.0, 30.0], [3]).unwrap();
        let output = lhs.add(&rhs).unwrap();

        assert_eq!(output.dims(), &[2, 3]);
        assert_eq!(output.data().unwrap(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn transpose_and_reshape_share_layout_semantics() {
        let tensor = Tensor::<Cpu>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.dims(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert_eq!(transposed.data().unwrap(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(transposed.reshape([6]).is_err());
    }

    #[test]
    fn gemm_uses_layout_strides() {
        let lhs = Tensor::<Cpu>::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2]).unwrap();
        let rhs = lhs.t().unwrap();
        let output = lhs.gemm(&rhs, None, None).unwrap();

        assert_eq!(output.dims(), &[3, 3]);
        assert_eq!(
            output.data().unwrap(),
            [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
        );
    }

    #[test]
    fn flatten_respects_axis() {
        let tensor = Tensor::<Cpu>::new(
            &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
            [2, 3, 4],
        )
        .unwrap();
        assert_eq!(tensor.flatten(1).unwrap().dims(), &[2, 12]);
        assert_eq!(tensor.flatten(2).unwrap().dims(), &[6, 4]);
    }
}
