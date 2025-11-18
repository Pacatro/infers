use num_traits::Num;

use crate::{InfersResult, backends::Backend, tensor::TensorData};

#[derive(Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Storage<T: Num + Copy + Clone> = Vec<T>;

    fn allocate<T: Num + Copy + Clone>(
        &self,
        shape: &[usize],
    ) -> InfersResult<TensorData<Self, T>> {
        let size = shape.iter().product::<usize>();
        Ok(TensorData::new(shape.to_vec(), Vec::with_capacity(size)))
    }

    fn copy_from<T: Num + Copy + Clone>(
        &self,
        data: &[T],
        shape: &[usize],
    ) -> InfersResult<TensorData<Self, T>> {
        Ok(TensorData::new(shape.to_vec(), data.to_vec()))
    }

    fn copy_to_vec<T: Num + Copy + Clone>(
        &self,
        data: &TensorData<Self, T>,
    ) -> InfersResult<Vec<T>> {
        Ok(data.storage.clone())
    }

    fn transfer_from<SrcBackend: Backend, T: Num + Copy + Clone>(
        &self,
        src_backend: &SrcBackend,
        src_data: &TensorData<SrcBackend, T>,
    ) -> InfersResult<TensorData<Self, T>> {
        let host_data = src_backend.copy_to_vec(src_data)?;
        self.copy_from(&host_data, &src_data.shape)
    }

    fn instance() -> Self {
        Self
    }

    fn name(&self) -> &str {
        "cpu"
    }

    fn add<T: Num + Copy + Clone>(
        &self,
        a: &TensorData<Self, T>,
        b: &TensorData<Self, T>,
    ) -> TensorData<Self, T> {
        assert_eq!(a.shape, b.shape);

        let result = a
            .storage
            .iter()
            .zip(b.storage.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        TensorData::new(a.shape.clone(), result)
    }
}
