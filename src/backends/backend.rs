use std::fmt::{Debug, Display};

use crate::InfersResult;

/// Represents the physical device where the computation and storage will occur.
///
/// This enum allows the system to differentiate between standard CPU processing
/// and acceleration devices like CUDA-enabled GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Standard Central Processing Unit.
    #[default]
    Cpu,
    /// NVIDIA GPU using the CUDA framework.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

/// A trait defining the required interface for a computation and storage backend
/// on a specific device (e.g., CPU, GPU).
///
/// This trait abstracts the low-level memory management and device-specific
/// operations, allowing the rest of the framework to interact with data in a
/// device-agnostic manner.
///
/// The generic parameter `T` represents the element type stored by the backend
/// (e.g., f32, i32).
#[allow(dead_code)]
pub trait Backend<T>: Clone + Debug + Copy {
    /// The device-specific memory storage type.
    ///
    /// This might be a `Vec<T>` for the CPU backend, or a GPU buffer type
    /// (like a CUDA pointer or buffer wrapper) for a GPU backend.
    type Storage: Clone + Debug;

    /// Returns the specific device associated with this backend implementation.
    ///
    /// # Returns
    ///
    /// A `Device` enum variant corresponding to the backend (e.g., `Device::Cpu`).
    fn device() -> Device;

    /// Initializes the device-specific storage from a slice of host data.
    ///
    /// This involves allocating memory on the target device and copying the
    /// initial data from the host.
    ///
    /// # Arguments
    ///
    /// * `data`: A slice of data (`&[T]`) residing on the host (CPU).
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `Self::Storage` on success, or an error.
    fn init(data: &[T]) -> InfersResult<Self::Storage>;

    /// Reads a single element from the device storage at a given index.
    ///
    /// This operation might involve a slow device-to-host synchronization/transfer
    /// for GPU backends. It should primarily be used for debugging or small reads.
    ///
    /// # Arguments
    ///
    /// * `storage`: A reference to the device storage.
    /// * `index`: The zero-based index of the element to read.
    ///
    /// # Returns
    ///
    /// The value of the element at `index`.
    fn read(storage: &Self::Storage, index: usize) -> T;

    /// Writes a single element to the device storage at a given index.
    ///
    /// Similar to `read`, this operation might be slow for GPU backends as it
    /// involves host-to-device communication.
    ///
    /// # Arguments
    ///
    /// * `storage`: A mutable reference to the device storage.
    /// * `index`: The zero-based index where the value should be written.
    /// * `value`: The new value of type `T`.
    fn write(storage: &mut Self::Storage, index: usize, value: T);

    /// Copies the entire contents of the device storage back to a host (CPU) `Vec<T>`.
    ///
    /// This is typically a synchronization point where data is transferred from
    /// the device memory back to the CPU memory.
    ///
    /// # Arguments
    ///
    /// * `storage`: A reference to the device storage.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<T>` of the copied data on success, or an error.
    fn copy_to_host(storage: &Self::Storage) -> InfersResult<Vec<T>>;

    /// Performs an element-wise addition of two device storage blocks.
    ///
    /// The result is stored in a newly allocated device storage block. This operation
    /// should be optimized to run entirely on the target device (e.g., using CUDA kernels
    /// for a GPU backend).
    ///
    /// # Arguments
    ///
    /// * `lhs`: The left-hand side operand storage.
    /// * `rhs`: The right-hand side operand storage.
    /// * `size`: The number of elements in the storage
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the result of `lhs + rhs`.
    fn add(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage;

    /// Performs an element-wise subtraction of two device storage blocks.
    ///
    /// The result is stored in a newly allocated device storage block. This operation
    /// should be optimized to run entirely on the target device (e.g., using CUDA kernels
    /// for a GPU backend).
    ///
    /// # Arguments
    ///
    /// * `lhs`: The left-hand side operand storage.
    /// * `rhs`: The right-hand side operand storage.
    /// * `size`: The number of elements in the storage
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the result of `lhs - rhs`.
    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage;

    /// Performs an element-wise multiplication of two device storage blocks.
    ///
    /// The result is stored in a newly allocated device storage block. This operation
    /// should be optimized to run entirely on the target device (e.g., using CUDA kernels
    /// for a GPU backend).
    ///
    /// # Arguments
    ///
    /// * `lhs`: The left-hand side operand storage.
    /// * `rhs`: The right-hand side operand storage.
    /// * `size`: The number of elements in the storage
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the result of `lhs * rhs`.
    fn mul(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage;

    /// Applies the ReLU (Rectified Linear Unit) activation function to the input storage.
    ///
    /// ReLU sets all negative values to zero and leaves positive values unchanged.
    /// This operation should be optimized to run entirely on the target device.
    ///
    /// # Arguments
    ///
    /// * `input`: The input storage to apply ReLU to.
    /// * `size`: The number of elements in the storage.
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the result of applying ReLU to `input`.
    fn relu(input: &Self::Storage, size: usize) -> Self::Storage;

    /// Performs a General Matrix Multiply (GEMM) operation.
    ///
    /// Computes the matrix multiplication: C = alpha * A * B + beta * C
    /// where A is an m×k matrix, B is a k×n matrix, and C is an m×n matrix.
    /// This operation should be optimized to run entirely on the target device.
    ///
    /// # Arguments
    ///
    /// * `lhs`: The left-hand side matrix (A) with dimensions m×k.
    /// * `rhs`: The right-hand side matrix (B) with dimensions k×n.
    /// * `alpha`: Scalar multiplier for the matrix product.
    /// * `beta`: Scalar multiplier for the existing matrix C.
    /// * `m`: Number of rows in matrices A and C.
    /// * `n`: Number of columns in matrices B and C.
    /// * `k`: Number of columns in matrix A and rows in matrix B.
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the result matrix C.
    fn gemm(
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        alpha: T,
        beta: T,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self::Storage;

    /// Computes the dot product of two storage blocks.
    ///
    /// Calculates the sum of element-wise products: Σ(lhs[i] * rhs[i])
    /// This operation should be optimized to run entirely on the target device.
    ///
    /// # Arguments
    ///
    /// * `lhs`: The left-hand side operand storage.
    /// * `rhs`: The right-hand side operand storage.
    /// * `size`: The number of elements in the storage.
    ///
    /// # Returns
    ///
    /// A new `Self::Storage` containing the scalar dot product result.
    fn dot(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage;
}
