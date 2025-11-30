use cudarc::{
    driver::{CudaContext, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits},
    nvrtc::compile_ptx,
};
use num_traits::Num;
use std::{fmt::Debug, sync::Arc};

use crate::{
    InfersResult,
    backends::{Backend, Device},
};

/// Device-specific storage structure for the CUDA backend.
///
/// This wraps the necessary CUDA context and the actual device buffer.
///
/// # Type Parameters
///
/// * `T`: The element type, which must be representable on a CUDA device.
#[derive(Debug, Clone)]
pub struct CudaStorage<T: DeviceRepr> {
    /// The CUDA context, shared via `Arc` to manage device resources.
    context: Arc<CudaContext>,
    /// The actual memory buffer stored on the CUDA device.
    buffer: CudaSlice<T>,
}

/// The CUDA backend implementation.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on an NVIDIA GPU.
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

    fn add(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        let ptx = compile_ptx("../kernels/add.cu").unwrap();

        let ctx = lhs.context.clone();
        let stream = ctx.default_stream();

        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("add").unwrap();

        let mut out_device = stream.alloc_zeros::<f32>(size).unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        let launch = stream
            .launch_builder(&func)
            .arg(&lhs.buffer)
            .arg(&rhs.buffer)
            .arg(&mut out_device)
            .arg(&size);

        unsafe {
            launch.launch(config).unwrap();
        }

        CudaStorage {
            context: ctx,
            buffer: out_device,
        }
    }
}
