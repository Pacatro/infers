use num_traits::{FromPrimitive, Num};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

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
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
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
/// storage and computation on different devices (CPU, cuda).
///
///
///
/// # Type Parameters
///
/// * `B`: The backend implementation (e.g., `Cpu`, `Cuda`). Defaults to `Cpu`.
/// * `T`: The element data type (e.g., `f32`, `i32`). Defaults to `f32`.
// TODO: Check https://huggingface.co/blog/KeighBee/tensors-from-scratch-in-rust-p1
// for better implementation
#[derive(Debug, Clone)]
pub struct Tensor<B = Cpu, T = f32>
where
    B: Backend<T>,
{
    /// The size of the tensor along each dimension (e.g., `[rows, columns]`).
    pub(crate) shape: Vec<usize>,
    /// The number of elements to skip in the linear storage to advance one unit along each dimension.
    pub(crate) strides: Vec<usize>,
    /// The underlying device-specific storage for the tensor data.
    pub(crate) storage: Rc<RefCell<B::Storage>>,
    /// The total number of elements in the tensor.
    pub(crate) len: usize,
    /// Marker to hold the backend type without storing data.
    pub(crate) _backend: PhantomData<B>,
}

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
        let len = shape.iter().product();

        let data = (0..len)
            .into_par_iter()
            .map(|_| rand::random::<f32>())
            .collect::<Vec<f32>>();

        Self {
            storage: Rc::new(RefCell::new(data)),
            shape: shape.to_vec(),
            len,
            strides: compute_strides(shape),
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
        let len: usize = shape.iter().product();

        let data = (0..len)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                rng.sample(StandardNormal)
            })
            .collect::<Vec<f32>>();

        Tensor {
            storage: Rc::new(RefCell::new(data)),
            shape: shape.to_vec(),
            len,
            strides: compute_strides(shape),
            _backend: PhantomData,
        }
    }
}

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
    /// Panics if the length of `data` does not match the total len implied by `shape`.
    pub fn new(data: &[T], shape: &[usize]) -> Self {
        let len = shape.iter().product();
        assert_eq!(
            data.len(),
            len,
            "Data length mismatch for shape {:?}",
            shape
        );

        Self {
            storage: Rc::new(RefCell::new(data.to_vec())),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            len,
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
        let len = shape.iter().product();
        Self {
            storage: Rc::new(RefCell::new(vec![T::zero(); len])),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            len,
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
        let len = shape.iter().product();

        let strides = compute_strides(shape);
        let storage = vec![T::one(); len];

        Self {
            storage: Rc::new(RefCell::new(storage)),
            shape: shape.to_vec(),
            len,
            strides,
            _backend: PhantomData,
        }
    }
}

impl<B, T> Tensor<B, T>
where
    B: Backend<T>,
    T: Num + FromPrimitive + Clone + Copy + FromPrimitive + Debug,
{
    /// Creates a tensor on the specified backend from host data.
    ///
    /// This method uses the backend's `init` function to move data from the host
    /// buffer to the device storage (e.g., copying to cuda memory for the CUDA backend).
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
        let len = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Ok(Self {
            storage: Rc::new(RefCell::new(B::init(data)?)),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            len,
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
        B::copy_to_host(&self.storage.borrow())
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
    /// This uses the backend's `read` method, which can be inefficient for cuda backends.
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
        B::read(&self.storage.borrow(), idx)
    }

    /// Sets a single element in the tensor using multi-dimensional indices.
    ///
    /// This uses the backend's `write` method, which can be inefficient for cuda backends.
    ///
    /// # Arguments
    ///
    /// * `indices`: The coordinate of the element to modify.
    /// * `value`: The new value to set.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.get_physical_index(indices);
        B::write(&mut self.storage.borrow_mut(), idx, value);
    }

    /// Returns the device this tensor resides on.
    pub fn device(&self) -> Device {
        B::device()
    }

    /// Returns the total number of elements in the tensor (the product of all dimensions).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tensor is empty (i.e., has zero length).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of dimensions of the tensor.
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
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
        let host_data = B::copy_to_host(&self.storage.borrow())?;
        // Initialize new tensor on target device (SrcB) from host data
        Tensor::from_data(&host_data, &self.shape)
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
        assert_eq!(t.len, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::<Cpu, i32>::zeros(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.len, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::<Cpu, i32>::ones(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.len, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.data().unwrap(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_tensor_rand() {
        let t = Tensor::rand(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.len, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.len, 4);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_randn() {
        let t = Tensor::randn(&[2, 2]);
        assert_eq!(t.shape, &[2, 2]);
        assert_eq!(t.len, 4);
        assert_eq!(t.strides, &[2, 1]);
        assert_eq!(t.len, 4);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_ndim() {
        let t = Tensor::rand(&[2, 2, 2]);
        assert_eq!(t.ndims(), 3);
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
    #[cfg(feature = "cuda")]
    fn test_tensor_to_cuda() {
        let t_cpu = Tensor::rand(&[2, 2]);
        let t_cuda = t_cpu.to::<Cuda>().unwrap();
        assert_eq!(t_cuda.device(), Device::Cuda);
        assert_eq!(t_cuda.shape, t_cpu.shape);
        assert_eq!(t_cuda.len, t_cpu.len);
        assert_eq!(t_cuda.strides, t_cpu.strides);
        assert_eq!(t_cuda.data().unwrap(), t_cpu.data().unwrap());
    }
}
