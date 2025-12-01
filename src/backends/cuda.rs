use cudarc::{
    driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

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
pub struct CudaStorage {
    /// The CUDA context, shared via `Arc` to manage device resources.
    context: Arc<CudaContext>,
    /// The actual memory buffer stored on the CUDA device.
    buffer: CudaSlice<f32>,
}

/// The CUDA backend implementation.
///
/// This struct implements the `Backend` trait, providing all the necessary
/// methods for managing data and performing operations on an NVIDIA GPU.
#[derive(Debug, Clone, Copy)]
pub struct Cuda;

impl Backend<f32> for Cuda {
    type Storage = CudaStorage;

    fn device() -> Device {
        Device::Cuda
    }

    fn init(data: &[f32]) -> InfersResult<Self::Storage> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let slice = stream.clone_htod(data)?;

        Ok(CudaStorage {
            context: ctx,
            buffer: slice,
        })
    }

    fn read(storage: &Self::Storage, index: usize) -> f32 {
        let stream = storage.context.default_stream();
        let host_buf = vec![0.0; storage.buffer.len()];
        stream.clone_dtoh(&storage.buffer).unwrap();
        host_buf[index]
    }

    fn write(storage: &mut Self::Storage, index: usize, value: f32) {
        let stream = storage.context.default_stream();
        let mut host_buf = stream.clone_dtoh(&storage.buffer).expect("DTOH failed");
        host_buf[index] = value;
        storage.buffer = stream.clone_htod(&host_buf).expect("HTOD failed");
    }

    fn copy_to_host(storage: &Self::Storage) -> InfersResult<Vec<f32>> {
        storage
            .context
            .default_stream()
            .clone_dtoh(&storage.buffer)
            .map_err(|e| e.into())
    }

    fn add(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        let ctx = lhs.context.clone();

        // FIXME: Can't compile to ptx because gpu arch is so fucking old.
        // After Cuda 13.x, nvcc stops supporting PASCAL architecture.
        // Which meand that my gtx 1060 is useless for this project.
        // My gpu is useless even for programming now :'D.
        let ptx = compile_ptx(include_str!("../../kernels/add.cu")).unwrap();

        let stream = ctx.default_stream();

        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("add").unwrap();

        let mut out_device = stream.alloc_zeros::<f32>(size).unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&lhs.buffer)
                .arg(&rhs.buffer)
                .arg(&mut out_device)
                .arg(&size)
                .launch(config)
                .unwrap();
        }

        CudaStorage {
            context: ctx,
            buffer: out_device,
        }
    }
}
