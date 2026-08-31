use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use num_traits::{FromPrimitive, Num};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::{
    backends::{Backend, Cpu, Device},
    core::InfersResult,
};

use super::TensorError;

/// Logical dimensions of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    num_elements: usize,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let dims = dims.into();
        let num_elements = dims.iter().try_fold(1usize, |size, &dimension| {
            size.checked_mul(dimension)
                .ok_or_else(|| TensorError::ShapeOverflow {
                    shape: dims.clone(),
                })
        })?;

        Ok(Self { dims, num_elements })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    pub fn dimension(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    pub fn broadcast_with(&self, other: &Self) -> Result<Self, TensorError> {
        let output_rank = self.rank().max(other.rank());
        let mut output = Vec::with_capacity(output_rank);

        for offset in 0..output_rank {
            let lhs = self
                .rank()
                .checked_sub(offset + 1)
                .map_or(1, |axis| self.dims[axis]);
            let rhs = other
                .rank()
                .checked_sub(offset + 1)
                .map_or(1, |axis| other.dims[axis]);

            if lhs != rhs && lhs != 1 && rhs != 1 {
                return Err(TensorError::IncompatibleShapes {
                    operation: "broadcast",
                    lhs: self.dims.clone(),
                    rhs: other.dims.clone(),
                });
            }

            output.push(if lhs == 1 { rhs } else { lhs });
        }

        output.reverse();
        Self::new(output)
    }

    pub fn matmul_with(&self, other: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 || other.rank() != 2 || self.dims[1] != other.dims[0] {
            return Err(TensorError::IncompatibleShapes {
                operation: "matrix multiplication",
                lhs: self.dims.clone(),
                rhs: other.dims.clone(),
            });
        }

        Self::new(vec![self.dims[0], other.dims[1]])
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        self.dims()
    }
}

/// Describes how a tensor shape maps onto its physical storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Self {
        let strides = compute_strides(shape.dims());
        Self { shape, strides }
    }

    pub(crate) fn from_parts(shape: Shape, strides: Vec<usize>) -> Result<Self, TensorError> {
        if shape.rank() != strides.len() {
            return Err(TensorError::ShapeStrideRankMismatch {
                shape_rank: shape.rank(),
                strides_rank: strides.len(),
            });
        }

        Ok(Self { shape, strides })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(self.shape.dims())
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self, TensorError> {
        if self.shape.num_elements() != shape.num_elements() {
            return Err(TensorError::InvalidReshape {
                from: self.shape.dims.clone(),
                to: shape.dims,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguousReshape);
        }

        Ok(Self::contiguous(shape))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self, TensorError> {
        let rank = self.shape.rank();
        if axis_a >= rank {
            return Err(TensorError::InvalidAxis { axis: axis_a, rank });
        }
        if axis_b >= rank {
            return Err(TensorError::InvalidAxis { axis: axis_b, rank });
        }

        let mut dims = self.shape.dims.clone();
        let mut strides = self.strides.clone();
        dims.swap(axis_a, axis_b);
        strides.swap(axis_a, axis_b);

        Self::from_parts(Shape::new(dims)?, strides)
    }

    pub(crate) fn physical_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.shape.rank()
            || indices
                .iter()
                .zip(self.shape.dims())
                .any(|(&index, &dimension)| index >= dimension)
        {
            return Err(TensorError::InvalidIndex {
                indices: indices.to_vec(),
                shape: self.shape.dims.clone(),
            });
        }

        Ok(indices
            .iter()
            .zip(&self.strides)
            .map(|(&index, &stride)| index * stride)
            .sum())
    }

    pub(crate) fn physical_index_from_flat(
        &self,
        flat_index: usize,
        output_shape: &Shape,
    ) -> usize {
        if self.shape.rank() == 0 {
            return 0;
        }

        let output_strides = compute_strides(output_shape.dims());
        let rank_offset = output_shape.rank() - self.shape.rank();

        self.shape
            .dims()
            .iter()
            .zip(self.strides())
            .enumerate()
            .map(|(axis, (&dimension, &stride))| {
                let output_axis = axis + rank_offset;
                let coordinate =
                    (flat_index / output_strides[output_axis]) % output_shape.dims()[output_axis];
                if dimension == 1 {
                    0
                } else {
                    coordinate * stride
                }
            })
            .sum()
    }
}

pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut current_stride = 1;
    for axis in (0..shape.len()).rev() {
        strides[axis] = current_stride;
        current_stride *= shape[axis];
    }
    strides
}

/// Device-backed multidimensional array.
#[derive(Debug, Clone)]
pub struct Tensor<B = Cpu, T = f32>
where
    B: Backend<T>,
{
    pub(crate) storage: Arc<B::Storage>,
    pub(crate) layout: Layout,
    pub(crate) _element: PhantomData<T>,
}

impl Tensor<Cpu, f32> {
    pub fn rand(dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        let shape = Shape::new(dims)?;
        let data = (0..shape.num_elements())
            .map(|_| rand::random::<f32>())
            .collect();
        Self::from_vec(data, shape.dims)
    }

    pub fn randn(dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        let shape = Shape::new(dims)?;
        let mut rng = rand::rng();
        let data = (0..shape.num_elements())
            .map(|_| rng.sample(StandardNormal))
            .collect();
        Self::from_vec(data, shape.dims)
    }
}

impl<T> Tensor<Cpu, T>
where
    Cpu: Backend<T, Storage = Vec<T>>,
    T: Num + Clone + Copy + FromPrimitive + Debug + Send + Sync,
{
    pub fn new(data: &[T], dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        Self::from_vec(data.to_vec(), dims)
    }

    pub fn zeros(dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        let shape = Shape::new(dims)?;
        Self::from_vec(vec![T::zero(); shape.num_elements()], shape.dims)
    }

    pub fn ones(dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        let shape = Shape::new(dims)?;
        Self::from_vec(vec![T::one(); shape.num_elements()], shape.dims)
    }
}

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + FromPrimitive + Clone + Copy + Debug + Send + Sync,
{
    pub fn from_vec(data: Vec<T>, dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        let shape = Shape::new(dims)?;
        if data.len() != shape.num_elements() {
            return Err(TensorError::DataLengthMismatch {
                expected: shape.num_elements(),
                actual: data.len(),
                shape: shape.dims.clone(),
            }
            .into());
        }

        let storage = B::from_host(data)?;
        Ok(Self::from_parts(storage, Layout::contiguous(shape)))
    }

    pub fn from_data(data: &[T], dims: impl Into<Vec<usize>>) -> InfersResult<Self> {
        Self::from_vec(data.to_vec(), dims)
    }

    pub(crate) fn from_parts(storage: B::Storage, layout: Layout) -> Self {
        Self {
            storage: Arc::new(storage),
            layout,
            _element: PhantomData,
        }
    }

    pub fn data(&self) -> InfersResult<Vec<T>> {
        B::to_host(self.storage.as_ref(), &self.layout)
    }

    pub fn get(&self, indices: &[usize]) -> InfersResult<T> {
        let index = self.layout.physical_index(indices)?;
        B::read(self.storage.as_ref(), index)
    }

    pub fn device(&self) -> Device {
        B::device()
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.layout.shape().dims()
    }

    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    pub fn ndims(&self) -> usize {
        self.rank()
    }

    pub fn len(&self) -> usize {
        self.layout.shape().num_elements()
    }

    pub fn size(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layout.shape().is_empty()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn to<Destination>(&self) -> InfersResult<Tensor<Destination, T>>
    where
        Destination: Backend<T>,
    {
        Tensor::from_vec(self.data()?, self.dims().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_validates_and_caches_size() {
        let shape = Shape::new([2, 3, 4]).unwrap();
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.num_elements(), 24);
    }

    #[test]
    fn shape_broadcasts_from_the_right() {
        let lhs = Shape::new([2, 1, 3]).unwrap();
        let rhs = Shape::new([4, 3]).unwrap();
        assert_eq!(lhs.broadcast_with(&rhs).unwrap().dims(), &[2, 4, 3]);
    }

    #[test]
    fn layout_transpose_is_a_view() {
        let layout = Layout::contiguous(Shape::new([2, 3]).unwrap());
        let transposed = layout.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn tensor_construction_validates_data_length() {
        let error = Tensor::<Cpu, i32>::new(&[1, 2, 3], [2, 2]).unwrap_err();
        assert!(matches!(error, crate::core::InfersError::Tensor(_)));
    }

    #[test]
    fn tensor_cpu_roundtrip() {
        let tensor = Tensor::<Cpu, i32>::new(&[1, 2, 3, 4], [2, 2]).unwrap();
        assert_eq!(tensor.dims(), &[2, 2]);
        assert_eq!(tensor.strides(), &[2, 1]);
        assert_eq!(tensor.data().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(tensor.get(&[1, 0]).unwrap(), 3);
    }
}
