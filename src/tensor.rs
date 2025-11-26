use num_traits::{FromPrimitive, Num};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops;

use crate::backends::{Backend, Device};

#[derive(Debug, Clone)]
pub struct Tensor<B, T>
where
    B: Backend<T>,
{
    pub storage: B::Storage,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    _backend: PhantomData<B>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut current_stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    strides
}

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug,
{
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size mismatch");

        let strides = compute_strides(shape);
        let storage = B::init(data);

        Self {
            storage,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let strides = compute_strides(shape);
        let storage = B::zeros(size);

        Self {
            storage,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }

    fn get_physical_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut physical_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            physical_idx += idx * self.strides[i];
        }
        physical_idx
    }

    pub fn get(&self, indices: &[usize]) -> T {
        let idx = self.get_physical_index(indices);
        // Delegamos la lectura al backend (crucial para GPU)
        B::read(&self.storage, idx)
    }

    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.get_physical_index(indices);
        B::write(&mut self.storage, idx, value);
    }

    pub fn device(&self) -> Device {
        B::device()
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug + ops::AddAssign,
{
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape);

        let new_storage = B::add(&self.storage, &other.storage);

        Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B, T> ops::Add for &Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug + ops::AddAssign,
{
    type Output = Tensor<B, T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

const MAX_TENSOR_DISPLAY: usize = 10;

impl<B, T> Display for Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Debug + Clone + Copy + FromPrimitive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.len() > MAX_TENSOR_DISPLAY {
            return write!(
                f,
                "Tensor(data=[...], shape={:?}, device={}, dtype={})",
                self.shape,
                B::device(),
                std::any::type_name::<T>()
            );
        }

        write!(
            f,
            "Tensor(data={:?}, shape={:?}, device={}, dtype={})",
            self.storage,
            self.shape,
            B::device(),
            std::any::type_name::<T>()
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::Cpu;

    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::<Cpu, i32>::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.storage, &[1, 2, 3, 4]);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::<Cpu, i32>::zeros(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.storage.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_tensor_get() {
        let t = Tensor::<Cpu, i32>::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.get(&[0, 0]), 1);
    }

    #[test]
    fn test_tensor_set() {
        let mut t = Tensor::<Cpu, i32>::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.get(&[0, 0]), 1);
        t.set(&[0, 0], 10);
        assert_eq!(t.get(&[0, 0]), 10);
    }

    #[test]
    fn test_tensor_add() {
        let t1 = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        let t2 = Tensor::new(&[5, 6, 7, 8], &[2, 2]);
        let t3: Tensor<Cpu, i32> = t1.add(&t2);
        assert_eq!(t3.storage.as_slice(), &[6, 8, 10, 12]);
    }

    #[test]
    fn test_tensor_add_ref() {
        let t1 = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        let t2 = Tensor::new(&[5, 6, 7, 8], &[2, 2]);
        let t3: Tensor<Cpu, i32> = &t1 + &t2;
        assert_eq!(t3.storage.as_slice(), &[6, 8, 10, 12]);
    }
}
