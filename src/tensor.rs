use std::ops;

use num_traits::Num;

use crate::{InfersResult, backends::Backend};

#[derive(Debug)]
pub struct TensorData<B: Backend, T: Num + Copy + Clone> {
    pub shape: Vec<usize>,
    pub storage: B::Storage<T>,
}

impl<B: Backend, T: Num + Copy + Clone> TensorData<B, T> {
    pub fn new(shape: Vec<usize>, storage: B::Storage<T>) -> Self {
        Self { shape, storage }
    }
}

pub struct Tensor<B: Backend, T: Num + Copy + Clone> {
    data: TensorData<B, T>,
    backend: B,
}

impl<B, T> Tensor<B, T>
where
    B: Backend,
    T: Num + Copy + Clone,
{
    pub fn with_backend(shape: &[usize], data: &[T], backend: B) -> InfersResult<Self> {
        let data = backend.copy_from(data, shape)?;
        Ok(Self { data, backend })
    }

    pub fn new(shape: &[usize], data: &[T]) -> InfersResult<Self> {
        Self::with_backend(shape, data, B::instance())
    }

    pub fn to<TargetBackend: Backend>(&self) -> InfersResult<Tensor<TargetBackend, T>> {
        let target_backend = TargetBackend::instance();
        let new_data = target_backend.transfer_from(&self.backend, &self.data)?;
        Ok(Tensor {
            data: new_data,
            backend: target_backend,
        })
    }

    pub fn backend(&self) -> &str {
        self.backend.name()
    }

    pub fn shape(&self) -> &[usize] {
        &self.data.shape
    }

    pub fn storage(&self) -> &B::Storage<T> {
        &self.data.storage
    }
}

impl<B: Backend, T: Num + Copy + Clone> ops::Add for Tensor<B, T> {
    type Output = Tensor<B, T>;

    fn add(self, rhs: Self) -> Self::Output {
        let tensor_data = self.backend.add(&self.data, &rhs.data);
        Tensor {
            data: tensor_data,
            backend: self.backend,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{backends::CpuBackend, tensor::Tensor};

    #[test]
    fn test_tensor() {
        let t = Tensor::<CpuBackend, f32>::new(&[2], &[1., 2.]).unwrap();
        assert_eq!(t.data.shape, vec![2]);
        assert_eq!(t.data.storage.as_slice(), &[1., 2.]);
        assert_eq!(t.backend(), "cpu");
    }

    #[test]
    fn test_tenser_add() {
        let t1 = Tensor::<CpuBackend, f32>::new(&[2], &[1., 2.]).unwrap();
        let t2 = Tensor::<CpuBackend, f32>::new(&[2], &[3., 4.]).unwrap();
        let t3 = t1 + t2;
        assert_eq!(t3.data.shape, vec![2]);
        assert_eq!(t3.data.storage.as_slice(), &[4., 6.]);
    }

    // #[test]
    // fn test_tensor_to() {
    //     let t = Tensor::<CpuBackend, f32>::new(&[2], &[1., 2.]).unwrap();
    //     let t_cuda = t.to::<CudaBackend>().unwrap();
    //
    //     assert_eq!(t_cuda.data.shape, t.data.shape);
    //     assert_eq!(t_cuda.data.storage.as_slice(), t.data.storage.as_slice());
    //     assert_eq!(t_cuda.backend(), "cuda");
    // }
}
