use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

use std::{fmt::Debug, sync::Arc};

use crate::{
    InfersResult,
    backends::{Backend, Cpu, Device},
    tensor::Tensor,
};

/// Compiles a CUDA kernel from a string source.
fn compile_kernel(
    src: &str,
    func_name: &str,
    ctx: &Arc<CudaContext>,
) -> InfersResult<CudaFunction> {
    let ptx = compile_ptx(src)?;
    let module = ctx.load_module(ptx)?;
    module.load_function(func_name).map_err(|e| e.into())
}

/// Device-specific storage structure for the CUDA backend.
///
/// This wraps the necessary CUDA context and the actual device buffer.
///
/// # Type Parameters
///
/// * `T`: The element type, which must be representable on a CUDA device.
#[derive(Debug, Clone)]
pub(crate) struct CudaStorage {
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
pub(crate) struct Cuda;

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
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("../../kernels/add.cu"), "add", &ctx).unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        let mut out_device = stream.alloc_zeros::<f32>(size).unwrap();
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

    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        let ctx = lhs.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("../../kernels/sub.cu"), "sub", &ctx).unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        let mut out_device = stream.alloc_zeros::<f32>(size).unwrap();
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

    fn relu(input: &Self::Storage, size: usize) -> Self::Storage {
        let ctx = input.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("../../kernels/relu.cu"), "relu", &ctx).unwrap();

        let config = LaunchConfig::for_num_elems(size as u32);
        let mut out_device = stream.alloc_zeros::<f32>(size).unwrap();

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&input.buffer)
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

    fn gemm(
        lhs: &Tensor<Self, f32>,
        rhs: &Tensor<Self, f32>,
        alpha: f32,
        beta: f32,
    ) -> Tensor<Self, f32> {
        let ctx = lhs.storage.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("../../kernels/gemm.cu"), "gemm", &ctx).unwrap();

        let m = lhs.shape[0] as i32;
        let k = lhs.shape[1] as i32;
        assert_eq!(rhs.shape[0] as i32, k);
        let n = rhs.shape[1] as i32;

        let mut c = Tensor::<Cpu, f32>::zeros(&[m as usize, n as usize])
            .to::<Self>()
            .unwrap();

        let block = (16, 16, 1);
        let grid = ((n + block.0 - 1) / block.0, (m + block.1 - 1) / block.1, 1);

        let config = LaunchConfig {
            grid_dim: (grid.0 as u32, grid.1 as u32, 1),
            block_dim: (block.0 as u32, block.1 as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&m)
                .arg(&n)
                .arg(&k)
                .arg(&alpha)
                .arg(&lhs.storage.buffer)
                .arg(&rhs.storage.buffer)
                .arg(&beta)
                .arg(&mut c.storage.buffer)
                .launch(config)
                .unwrap();
        }

        stream.synchronize().unwrap();

        c
    }
}
