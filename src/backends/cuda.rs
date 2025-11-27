use cudarc::driver::{CudaContext, CudaSlice, DeviceRepr, ValidAsZeroBits};
use num_traits::Num;
use std::{fmt::Debug, sync::Arc};

use crate::{
    InfersResult,
    backends::{Backend, Device},
};

#[derive(Debug, Clone)]
pub struct CudaStorage<T: DeviceRepr> {
    context: Arc<CudaContext>,
    buffer: CudaSlice<T>,
}

#[derive(Debug, Clone, Copy)]
pub struct Cuda;

impl<T> Backend<T> for Cuda
where
    T: Num + Clone + Copy + Debug + DeviceRepr + ValidAsZeroBits,
{
    type Storage = CudaStorage<T>;

    fn device() -> Device {
        Device::Cuda
    }

    fn init(data: &[T]) -> InfersResult<Self::Storage> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let slice = stream.clone_htod(data)?;

        Ok(CudaStorage {
            context: ctx,
            buffer: slice,
        })
    }

    fn zeros(size: usize) -> InfersResult<Self::Storage> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let slice: CudaSlice<T> = stream.alloc_zeros(size).unwrap();

        Ok(CudaStorage {
            context: ctx,
            buffer: slice,
        })
    }

    fn read(storage: &Self::Storage, index: usize) -> T {
        let stream = storage.context.default_stream();
        let host_buf = vec![T::zero(); storage.buffer.len()];
        stream.clone_dtoh(&storage.buffer).unwrap();
        host_buf[index]
    }

    fn write(storage: &mut Self::Storage, index: usize, value: T) {
        let stream = storage.context.default_stream();
        let mut host_buf = stream.clone_dtoh(&storage.buffer).expect("DTOH failed");
        host_buf[index] = value;
        storage.buffer = stream.clone_htod(&host_buf).expect("HTOD failed");
    }

    fn copy_to_host(storage: &Self::Storage) -> InfersResult<Vec<T>> {
        storage
            .context
            .default_stream()
            .clone_dtoh(&storage.buffer)
            .map_err(|e| e.into())
    }

    fn add(_lhs: &Self::Storage, _rhs: &Self::Storage) -> Self::Storage {
        todo!()
    }
}
