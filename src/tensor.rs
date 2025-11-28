use num_traits::{FromPrimitive, Num};
use rand::Rng;
use rand_distr::StandardNormal;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops;

use crate::InfersResult;
use crate::backends::{Backend, Cpu, Device};

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut current_stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    strides
}

#[derive(Debug, Clone)]
pub struct Tensor<B, T>
where
    B: Backend<T>,
{
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    storage: B::Storage,
    _backend: PhantomData<B>,
}

impl<Cpu, T> Tensor<Cpu, T>
where
    Cpu: Backend<T, Storage = Vec<T>>,
    T: Num + Clone + Copy + FromPrimitive + Debug,
{
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        let size = shape.iter().product();

        assert_eq!(
            data.len(),
            size,
            "Data size mismatch: expected {} elements for shape {:?}, got {}",
            size,
            shape,
            data.len()
        );

        let strides = compute_strides(shape);
        let storage = data.to_vec();

        Self {
            storage,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();

        let strides = compute_strides(shape);
        let storage = vec![T::zero(); size];

        Self {
            storage,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }
}

impl Tensor<Cpu, f32> {
    pub fn rand(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let strides = compute_strides(shape);

        let data = (0..size)
            .map(|_| rand::random::<f32>())
            .collect::<Vec<f32>>();

        Self {
            storage: data,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }

    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let strides = compute_strides(shape);
        let mut rng = rand::rng();

        let data = (0..size)
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<f32>>();

        Tensor {
            storage: data,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        }
    }
}

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug,
{
    pub fn from_data(data: &[T], shape: &[usize]) -> InfersResult<Self> {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size mismatch");

        let strides = compute_strides(shape);
        let storage = B::init(data)?;

        Ok(Self {
            storage,
            shape: shape.to_vec(),
            strides,
            _backend: PhantomData,
        })
    }

    pub fn data(&self) -> InfersResult<Vec<T>> {
        B::copy_to_host(&self.storage)
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

    pub fn to<SrcB>(&self) -> InfersResult<Tensor<SrcB, T>>
    where
        SrcB: Backend<T>,
        T: Num + Clone + Copy + FromPrimitive + Debug,
    {
        let host_data = B::copy_to_host(&self.storage)?;
        Tensor::from_data(&host_data, &self.shape)
    }
}

impl<B, T> ops::Add for &Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug + ops::AddAssign,
{
    type Output = Tensor<B, T>;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        assert_eq!(
            self.device(),
            rhs.device(),
            "The two tensors must be on the same device."
        );

        let new_storage = B::add(&self.storage, &rhs.storage);

        Self::Output {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B, T> ops::AddAssign for Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Clone + Copy + FromPrimitive + Debug + ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        assert_eq!(
            self.device(),
            rhs.device(),
            "The two tensors must be on the same device."
        );

        self.storage = B::add(&self.storage, &rhs.storage);
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

        let data = match B::copy_to_host(&self.storage) {
            Ok(data) => data,
            Err(e) => return write!(f, "{:?}", e),
        };

        write!(
            f,
            "Tensor(data={:?}, shape={:?}, device={}, dtype={})",
            data,
            self.shape,
            B::device(),
            std::any::type_name::<T>()
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::{Cpu, Cuda};

    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::<Cpu, i32>::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_rand() {
        let t = Tensor::<Cpu, f32>::rand(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::<Cpu, i32>::zeros(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![0, 0, 0, 0]);
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
        let t3: Tensor<Cpu, i32> = &t1 + &t2;
        assert_eq!(t3.data().unwrap(), vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_tensor_add_ref() {
        let t1 = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        let t2 = Tensor::new(&[5, 6, 7, 8], &[2, 2]);
        let t3: Tensor<Cpu, i32> = &t1 + &t2;
        assert_eq!(t3.data().unwrap(), vec![6, 8, 10, 12]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_to_cuda() {
        let t_cpu = Tensor::<Cpu, i32>::zeros(&[2, 2]);
        let t_gpu = t_cpu.to::<Cuda>().unwrap();
        assert_eq!(t_gpu.device(), Device::Cuda);
        assert_eq!(t_gpu.shape, t_cpu.shape);
        assert_eq!(t_gpu.strides, t_cpu.strides);
        assert_eq!(t_gpu.data().unwrap(), t_cpu.data().unwrap());
    }
}
