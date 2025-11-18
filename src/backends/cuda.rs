use num_traits::Num;

use crate::{InfersResult, backends::Backend, tensor::TensorData};

#[derive(Clone, Debug)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Storage<T: Num + Copy + Clone> = Vec<T>;

    fn allocate<T: Num + Copy + Clone>(
        &self,
        _shape: &[usize],
    ) -> InfersResult<TensorData<Self, T>> {
        todo!("This backend is not implemented yet");
    }

    fn copy_from<T: Num + Copy + Clone>(
        &self,
        _data: &[T],
        _shape: &[usize],
    ) -> InfersResult<TensorData<Self, T>> {
        todo!("This backend is not implemented yet");
    }

    fn copy_to_vec<T: Num + Copy + Clone>(
        &self,
        _data: &TensorData<Self, T>,
    ) -> InfersResult<Vec<T>> {
        todo!("This backend is not implemented yet");
    }

    fn transfer_from<SrcBackend: Backend, T: Num + Copy + Clone>(
        &self,
        _src_backend: &SrcBackend,
        _src_data: &TensorData<SrcBackend, T>,
    ) -> InfersResult<TensorData<Self, T>> {
        todo!("This backend is not implemented yet");
    }

    fn instance() -> Self {
        Self
    }

    fn name(&self) -> &str {
        "cuda"
    }

    fn add<T: Num + Copy + Clone>(
        &self,
        _a: &TensorData<Self, T>,
        _b: &TensorData<Self, T>,
    ) -> TensorData<Self, T> {
        todo!("This backend is not implemented yet");
    }
}
