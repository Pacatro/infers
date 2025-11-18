use num_traits::Num;

use crate::{InfersResult, tensor::TensorData};

pub trait Backend: Clone {
    type Storage<T: Num + Copy + Clone>: Clone;

    /// Allocate memory for tensor data
    fn allocate<T: Num + Copy + Clone>(&self, shape: &[usize])
    -> InfersResult<TensorData<Self, T>>;

    /// Copy data from a slice to this backend
    fn copy_from<T: Num + Copy + Clone>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> InfersResult<TensorData<Self, T>>;

    /// Copy data from this backend to a Vec
    fn copy_to_vec<T: Num + Copy + Clone>(
        &self,
        data: &TensorData<Self, T>,
    ) -> InfersResult<Vec<T>>;

    /// Transfer data from another backend to this one
    fn transfer_from<SrcBackend: Backend, T: Num + Copy + Clone>(
        &self,
        src_backend: &SrcBackend,
        src_data: &TensorData<SrcBackend, T>,
    ) -> InfersResult<TensorData<Self, T>>;

    fn instance() -> Self;

    fn name(&self) -> &str;

    fn add<T: Num + Copy + Clone>(
        &self,
        a: &TensorData<Self, T>,
        b: &TensorData<Self, T>,
    ) -> TensorData<Self, T>;

    // TODO: Add more methods
}
