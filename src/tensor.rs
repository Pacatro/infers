use num_traits::{Float, FromPrimitive, Num};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops;

use crate::InfersResult;
use crate::backends::{Backend, Cpu, Device};

/// Calculates the strides (step size in linear memory) for a given tensor shape
/// assuming a row-major (C-style) memory layout.
///
/// # Arguments
///
/// * `shape`: A slice representing the dimensions of the tensor (e.g., `[2, 3]` for a 2x3 matrix).
///
/// # Returns
///
/// A vector of strides, where `strides[i]` is the step in the linear buffer
/// required to advance the index along the `i`-th dimension.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut current_stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i];
    }
    strides
}

/// The core data structure for numerical computation, representing a multi-dimensional
/// array (Tensor).
///
/// Tensors are device-agnostic, relying on the generic `Backend` trait to handle
/// storage and computation on different devices (CPU, GPU).
///
///
///
/// # Type Parameters
///
/// * `B`: The backend implementation (e.g., `Cpu`, `Cuda`). Defaults to `Cpu`.
/// * `T`: The element data type (e.g., `f32`, `i32`). Defaults to `f32`.
#[derive(Debug, Clone)]
pub(crate) struct Tensor<B = Cpu, T = f32>
where
    B: Backend<T>,
{
    /// The size of the tensor along each dimension (e.g., `[rows, columns]`).
    pub shape: Vec<usize>,
    /// The number of elements to skip in the linear storage to advance one unit along each dimension.
    pub strides: Vec<usize>,
    /// The total number of elements in the tensor.
    size: usize,
    /// The underlying device-specific storage for the tensor data.
    storage: B::Storage,
    /// Marker to hold the backend type without storing data.
    _backend: PhantomData<B>,
}

#[allow(dead_code)]
impl Tensor {
    /// Creates a new CPU tensor initialized with random numbers uniformly distributed
    /// between 0.0 and 1.0.
    ///
    /// # Arguments
    ///
    /// * `shape`: The multi-dimensional shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor of type `f32` with random values.
    pub fn rand(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let strides = compute_strides(shape);

        let data = (0..size)
            .into_par_iter()
            .map(|_| rand::random::<f32>())
            .collect::<Vec<f32>>();

        Self {
            storage: data,
            shape: shape.to_vec(),
            size,
            strides,
            _backend: PhantomData,
        }
    }

    /// Creates a new CPU tensor initialized with random numbers from a standard
    /// normal distribution (mean 0, variance 1).
    ///
    /// # Arguments
    ///
    /// * `shape`: The multi-dimensional shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor of type `f32` with normally distributed random values.
    pub fn randn(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let strides = compute_strides(shape);

        let data = (0..size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                rng.sample(StandardNormal)
            })
            .collect::<Vec<f32>>();

        Tensor {
            storage: data,
            shape: shape.to_vec(),
            size,
            strides,
            _backend: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<T> Tensor<Cpu, T>
where
    Cpu: Backend<T, Storage = Vec<T>>,
    T: Num + Clone + Copy + FromPrimitive + Debug,
{
    /// Creates a new CPU tensor from a linear data buffer and a shape.
    ///
    /// The data buffer is copied directly into the tensor's storage.
    ///
    /// # Arguments
    ///
    /// * `data`: The flat array of data elements.
    /// * `shape`: The multi-dimensional shape of the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` does not match the total size implied by `shape`.
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size mismatch for shape {:?}", shape);

        Self {
            storage: data.to_vec(),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            size,
            _backend: PhantomData,
        }
    }

    /// Creates a new CPU tensor initialized with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape`: The multi-dimensional shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor of the specified shape filled with the zero element of type `T`.
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            storage: vec![T::zero(); size],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            size,
            _backend: PhantomData,
        }
    }

    /// Creates a new CPU tensor initialized with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The multi-dimensional shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor of the specified shape filled with the one element of type `T`.
    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();

        let strides = compute_strides(shape);
        let storage = vec![T::one(); size];

        Self {
            storage,
            shape: shape.to_vec(),
            size,
            strides,
            _backend: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + FromPrimitive + Clone + Copy + FromPrimitive + Debug,
{
    /// Creates a tensor on the specified backend from host data.
    ///
    /// This method uses the backend's `init` function to move data from the host
    /// buffer to the device storage (e.g., copying to GPU memory for the CUDA backend).
    ///
    /// # Arguments
    ///
    /// * `data`: A slice of host data.
    /// * `shape`: The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `Tensor` or an error if backend initialization fails.
    pub fn from_data(data: &[T], shape: &[usize]) -> InfersResult<Self> {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size mismatch");
        Ok(Self {
            storage: B::init(data)?,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            size,
            _backend: PhantomData,
        })
    }

    /// Retrieves the tensor data from the device to a host (CPU) `Vec<T>`.
    ///
    /// This is an expensive synchronization operation for non-CPU backends.
    ///
    /// # Returns
    ///
    /// A `Result` containing the data as a linear `Vec<T>` on the host.
    pub fn data(&self) -> InfersResult<Vec<T>> {
        B::copy_to_host(&self.storage)
    }

    /// Converts a multi-dimensional index tuple into a single linear (physical) index
    /// in the underlying storage buffer, using the calculated strides.
    ///
    /// # Arguments
    ///
    /// * `indices`: The coordinate in the tensor (e.g., `[row, column]`).
    ///
    /// # Returns
    ///
    /// The flat index in the `storage` buffer.
    ///
    /// # Panics
    ///
    /// Panics if the number of indices does not match the tensor's rank (`shape.len()`).
    fn get_physical_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");
        let mut physical_idx = 0;
        for (i, &idx) in indices.iter().enumerate() {
            physical_idx += idx * self.strides[i];
        }
        physical_idx
    }

    /// Retrieves a single element from the tensor using multi-dimensional indices.
    ///
    /// This uses the backend's `read` method, which can be inefficient for GPU backends.
    ///
    /// # Arguments
    ///
    /// * `indices`: The coordinate of the element to retrieve.
    ///
    /// # Returns
    ///
    /// The value of the element at the specified indices.
    pub fn get(&self, indices: &[usize]) -> T {
        let idx = self.get_physical_index(indices);
        B::read(&self.storage, idx)
    }

    /// Sets a single element in the tensor using multi-dimensional indices.
    ///
    /// This uses the backend's `write` method, which can be inefficient for GPU backends.
    ///
    /// # Arguments
    ///
    /// * `indices`: The coordinate of the element to modify.
    /// * `value`: The new value to set.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.get_physical_index(indices);
        B::write(&mut self.storage, idx, value);
    }

    /// Returns the device this tensor resides on.
    pub fn device(&self) -> Device {
        B::device()
    }

    /// Returns the total number of elements in the tensor (the product of all dimensions).
    pub fn len(&self) -> usize {
        self.size
    }

    /// Converts the tensor from its current backend (`B`) to a new backend (`SrcB`).
    ///
    /// This involves copying the data from the current device to the host (CPU),
    /// and then from the host to the new target device.
    ///
    /// # Type Parameters
    ///
    /// * `SrcB`: The target backend type.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `Tensor` instance on the target backend.
    pub fn to<SrcB>(&self) -> InfersResult<Tensor<SrcB, T>>
    where
        SrcB: Backend<T>,
    {
        // Copy data from current device (B) to host (CPU)
        let host_data = B::copy_to_host(&self.storage)?;
        // Initialize new tensor on target device (SrcB) from host data
        Tensor::from_data(&host_data, &self.shape)
    }

    pub fn relu(&self) -> Self {
        Self {
            storage: B::relu(&self.storage, self.size),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
            _backend: PhantomData,
        }
    }

    pub fn flatten(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: vec![self.size],
            strides: vec![1],
            size: self.size,
            _backend: PhantomData,
        }
    }

    pub fn matmul(&self, rhs: Self) -> Self {
        B::gemm(self, &rhs, T::one(), T::zero())
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

        let new_storage = B::add(&self.storage, &rhs.storage, self.size);

        Self::Output {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
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

        let new_storage = B::sub(&self.storage, &rhs.storage, self.size);

        Self::Output {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            size: self.size,
            _backend: PhantomData,
        }
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
    use crate::backends::Cpu;

    #[cfg(feature = "cuda")]
    use crate::backends::Cuda;

    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.size, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::<Cpu, i32>::zeros(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.size, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::<Cpu, i32>::ones(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.size, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_tensor_rand() {
        let t = Tensor::<Cpu, f32>::rand(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.size, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_get() {
        let t = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.get(&[0, 0]), 1);
    }

    #[test]
    fn test_tensor_set() {
        let mut t = Tensor::new(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(t.get(&[0, 0]), 1);
        t.set(&[0, 0], 10);
        assert_eq!(t.get(&[0, 0]), 10);
    }

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
        let t3 = t1.matmul(t2);

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
        let t3 = t1.matmul(t2);

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
        let _ = t1.matmul(t2);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_to_cuda() {
        let t_cpu = Tensor::rand(&[2, 2]);
        let t_gpu = t_cpu.to::<Cuda>().unwrap();
        assert_eq!(t_gpu.device(), Device::Cuda);
        assert_eq!(t_gpu.shape, t_cpu.shape);
        assert_eq!(t_gpu.size, t_cpu.size);
        assert_eq!(t_gpu.strides, t_cpu.strides);
        assert_eq!(t_gpu.data().unwrap(), t_cpu.data().unwrap());
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
}
