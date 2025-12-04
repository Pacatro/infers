use num_traits::{Float, FromPrimitive, Num};
use std::ops;
use std::{fmt::Debug, marker::PhantomData};

use crate::tensor::base::compute_strides;
use crate::{Tensor, backends::Backend};

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + FromPrimitive + Clone + Copy + FromPrimitive + Debug,
{
    /// Applies the ReLU activation function to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` with the ReLU activation applied.
    pub fn relu(&self) -> Self {
        Self {
            storage: B::relu(&self.storage, self.len),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            len: self.len,
            _backend: PhantomData,
        }
    }

    /// Flattens the tensor into a 1D array.
    ///
    /// # Returns
    ///
    /// A new `Tensor` with the flattened data.
    pub fn flatten(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: vec![self.len],
            strides: vec![1],
            len: self.len,
            _backend: PhantomData,
        }
    }

    /// Performs matrix multiplication between two tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs`: The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing the result of the matrix multiplication.
    pub fn matmul(&self, rhs: &Self) -> Self {
        // TODO: Use batching to handle higher-dimensional tensors

        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];

        assert_eq!(
            k, rhs.shape[0],
            "mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})",
            m, k, k, n
        );

        let new_storage = B::gemm(&self.storage, &rhs.storage, T::one(), T::zero(), m, n, k);
        let shape = vec![m, n];
        let strides = compute_strides(&shape);

        Self {
            storage: new_storage,
            shape,
            strides,
            len: m * n,
            _backend: PhantomData,
        }
    }
}
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

impl<B, T> ops::Mul for Tensor<B, T>
where
    B: Backend<T>,
    T: Float + Clone + Copy + FromPrimitive + Debug + Send + Sync,
{
    type Output = Tensor<B, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        assert_eq!(
            self.device(),
            rhs.device(),
            "The two tensors must be on the same device."
        );

        let new_storage = B::mul(&self.storage, &rhs.storage, self.len);

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
    #[cfg(feature = "cuda")]
    use crate::backends::Cuda;
    use crate::{Tensor, backends::Device};

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
    fn test_tensor_mul_cpu() {
        let t1 = Tensor::new(&[5., 6., 7., 8.], &[2, 2]);
        let t2 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t3 = t1 * t2;
        assert_eq!(t3.data().unwrap(), vec![5., 12., 21., 32.]);
    }

    #[test]
    fn test_tensor_relu_cpu() {
        let t = Tensor::new(&[-1., -2., 3., 4.], &[2, 2]);
        let t = t.relu();
        assert_eq!(t.data().unwrap(), vec![0., 0., 3., 4.]);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.shape, &[2, 2]);
    }

    #[test]
    fn test_tensor_flatten_cpu() {
        let t = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t = t.flatten();
        assert_eq!(t.shape, &[4]);
        assert_eq!(t.strides, &[1]);
        assert_eq!(t.data().unwrap(), vec![1., 2., 3., 4.]);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_matmul_same_shapes_cpu() {
        let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t2 = Tensor::new(&[5., 6., 7., 8.], &[2, 2]);
        let t3 = t1.matmul(&t2);

        assert_eq!(t3.data().unwrap(), vec![19., 22., 43., 50.]);
        assert_eq!(t3.device(), Device::Cpu);
        assert_eq!(t3.shape, &[2, 2]);
    }

    #[test]
    fn test_tensor_matmul_diff_shapes_cpu() {
        let t1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let t2 = Tensor::new(
            &[5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
            &[3, 4],
        );
        let t3 = t1.matmul(&t2);

        assert_eq!(
            t3.data().unwrap(),
            vec![62., 68., 74., 80., 143., 158., 173., 188.]
        );
        assert_eq!(t3.device(), Device::Cpu);
        assert_eq!(t3.shape, &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "mat1 and mat2 shapes cannot be multiplied (2x2 and 2x2)")]
    fn test_tensor_matmul_bad_shapes_cpu() {
        let t1 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t2 = Tensor::new(&[5., 6.], &[1, 2]);
        let _ = t1.matmul(&t2);
    }

    #[test]
    #[ignore = "Not implemented"]
    fn test_tensor_matmul_ndims_cpu() {
        let t1 = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
        let t2 = Tensor::new(&[5., 6., 7., 8., 9., 10., 11., 12.], &[2, 2, 2]);

        let t3 = t1.matmul(&t2);

        assert_eq!(t3.shape, &[2, 2, 2]);
        assert_eq!(
            t3.data().unwrap(),
            vec![19., 22., 43., 50., 111., 122., 151., 166.]
        );
        assert_eq!(t3.device(), Device::Cpu);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_add_cuda() {
        use crate::backends::Device;

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
        use crate::backends::Device;

        let t1 = Tensor::<Cuda, f32>::from_data(&[5., 6., 7., 8.], &[2, 2]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();

        let t3 = t1 - t2;

        assert_eq!(t3.data().unwrap(), vec![4., 4., 4., 4.]);
        assert_eq!(t3.device(), Device::Cuda);
        assert_eq!(t3.shape, &[2, 2]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_mul_cuda() {
        let t1 = Tensor::new(&[5., 6., 7., 8.], &[2, 2]);
        let t2 = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
        let t3 = t1 * t2;
        assert_eq!(t3.data().unwrap(), vec![5., 12., 21., 32.]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_relu_cuda() {
        let t = Tensor::<Cuda, f32>::from_data(&[-1., -2., 3., 4.], &[2, 2]).unwrap();
        let t = t.relu();
        assert_eq!(t.data().unwrap(), vec![0., 0., 3., 4.]);
        assert_eq!(t.device(), Device::Cuda);
        assert_eq!(t.shape, &[2, 2]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_flatten_cuda() {
        let t = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let t = t.flatten();
        assert_eq!(t.shape, &[4]);
        assert_eq!(t.strides, &[1]);
        assert_eq!(t.data().unwrap(), vec![1., 2., 3., 4.]);
        assert_eq!(t.device(), Device::Cuda);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_matmul_same_shapes_cuda() {
        let t1 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(&[5., 6., 7., 8.], &[2, 2]).unwrap();
        let t3 = t1.matmul(&t2);

        assert_eq!(t3.data().unwrap(), vec![19., 22., 43., 50.]);
        assert_eq!(t3.device(), Device::Cuda);
        assert_eq!(t3.shape, &[2, 2]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_matmul_diff_shapes_cuda() {
        let t1 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(
            &[5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
            &[3, 4],
        )
        .unwrap();
        let t3 = t1.matmul(&t2);

        assert_eq!(
            t3.data().unwrap(),
            vec![62., 68., 74., 80., 143., 158., 173., 188.]
        );
        assert_eq!(t3.device(), Device::Cuda);
        assert_eq!(t3.shape, &[2, 4]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[should_panic(expected = "mat1 and mat2 shapes cannot be multiplied (2x2 and 2x2)")]
    fn test_tensor_matmul_bad_shapes_cuda() {
        let t1 = Tensor::<Cuda, f32>::from_data(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let t2 = Tensor::<Cuda, f32>::from_data(&[5., 6.], &[1, 2]).unwrap();
        let _ = t1.matmul(&t2);
    }

    #[test]
    #[ignore = "Not implemented"]
    fn test_tensor_matmul_ndims_cuda() {
        let t1 = Tensor::new(&[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
        let t2 = Tensor::new(&[5., 6., 7., 8., 9., 10., 11., 12.], &[2, 2, 2]);

        let t3 = t1.matmul(&t2);

        assert_eq!(t3.shape, &[2, 2, 2]);
        assert_eq!(
            t3.data().unwrap(),
            vec![19., 22., 43., 50., 111., 122., 151., 166.]
        );
        assert_eq!(t3.device(), Device::Cpu);
    }
}
