use num_traits::{Float, FromPrimitive};
use std::ops;
use std::{fmt::Debug, marker::PhantomData};

use crate::{Tensor, backends::Backend};

impl<B, T> ops::Add for Tensor<B, T>
where
    B: Backend<T>,
    T: Float + Clone + Copy + FromPrimitive + Debug + Send + Sync,
{
    type Output = Tensor<B, T>;

    /// Performs element-wise addition between two tensors on the same device.
    ///
    /// The operation is delegated to the backend's efficient `add` method.
    /// Returns a new tensor containing the result.
    ///
    /// # Panics
    ///
    /// Panics if the tensors have different shapes or reside on different devices.
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        assert_eq!(
            self.device(),
            rhs.device(),
            "The two tensors must be on the same device."
        );

        let new_storage = B::add(&self.storage, &rhs.storage, self.len);

        Self::Output {
            storage: new_storage,
            shape: self.shape,
            strides: self.strides,
            len: self.len,
            _backend: PhantomData,
        }
    }
}

impl<B, T> ops::Sub for Tensor<B, T>
where
    B: Backend<T>,
    T: Float + Clone + Copy + FromPrimitive + Debug + Send + Sync,
{
    type Output = Tensor<B, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        assert_eq!(
            self.device(),
            rhs.device(),
            "The two tensors must be on the same device."
        );

        let new_storage = B::sub(&self.storage, &rhs.storage, self.len);

        Self::Output {
            storage: new_storage,
            shape: self.shape,
            strides: self.strides,
            len: self.len,
            _backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::Tensor;
    use crate::backends::{Cuda, Device};

    #[test]
    fn test_tensor_add_cpu() {
        let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t2 = Tensor::new(&[5., 6., 7., 8.], &[2, 2]);
        let t3 = t1 + t2;
        assert_eq!(t3.data().unwrap(), vec![6., 8., 10., 12.]);
    }

    #[test]
    fn test_tensor_sub_cpu() {
        let t1 = Tensor::new(&[5., 6., 7., 8.], &[2, 2]);
        let t2 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t3 = t1 - t2;
        assert_eq!(t3.data().unwrap(), vec![4., 4., 4., 4.]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_add_cuda() {
        let t1 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(&[5., 6., 7., 8.], &[2, 2]).unwrap();
        let t3 = t1 + t2;

        assert_eq!(t3.data().unwrap(), vec![6., 8., 10., 12.]);
        assert_eq!(t3.device(), Device::Cuda);
        assert_eq!(t3.shape, &[2, 2]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_sub_cuda() {
        let t1 = Tensor::<Cuda, f32>::from_data(&[5., 6., 7., 8.], &[2, 2]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();

        let t3 = t1 - t2;

        assert_eq!(t3.data().unwrap(), vec![4., 4., 4., 4.]);
        assert_eq!(t3.device(), Device::Cuda);
        assert_eq!(t3.shape, &[2, 2]);
    }
}
