use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

use std::{fmt::Debug, sync::Arc};

use crate::{
    InfersResult,
    backends::{Backend, Device},
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
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        alpha: f32,
        beta: f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self::Storage {
        let ctx = lhs.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("../../kernels/gemm.cu"), "gemm", &ctx).unwrap();

        let mut c = stream.alloc_zeros::<f32>(m * n).unwrap();

        let block = (16, 16, 1);
        let grid = (n.div_ceil(block.0), m.div_ceil(block.1), 1);

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
                .arg(&lhs.buffer)
                .arg(&rhs.buffer)
                .arg(&beta)
                .arg(&mut c)
                .launch(config)
                .unwrap();
        }

        stream.synchronize().unwrap();

        CudaStorage {
            context: ctx,
            buffer: c,
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_backend_init_and_copy_roundtrip_cuda() {
        let host = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = Cuda::init(&host).unwrap();
        let back = Cuda::copy_to_host(&storage).unwrap();
        assert_eq!(back, host);
    }

    #[test]
    fn test_backend_add_cuda() {
        let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
        let size = lhs.len();
        let s_lhs = Cuda::init(&lhs).unwrap();
        let s_rhs = Cuda::init(&rhs).unwrap();
        let s_out = Cuda::add(&s_lhs, &s_rhs, size);
        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_backend_sub_cuda() {
        let lhs = vec![5.0f32, 6.0, 7.0, 8.0];
        let rhs = vec![1.0f32, 2.0, 3.0, 4.0];
        let size = lhs.len();
        let s_lhs = Cuda::init(&lhs).unwrap();
        let s_rhs = Cuda::init(&rhs).unwrap();
        let s_out = Cuda::sub(&s_lhs, &s_rhs, size);
        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_backend_relu_cuda() {
        let input = vec![-1.0f32, -2.0, 3.0, 4.0];
        let s_input = Cuda::init(&input).unwrap();
        let s_out = Cuda::relu(&s_input, input.len());
        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_backend_gemm_cuda_2x2() {
        // A = [1 2; 3 4], B = [5 6; 7 8]
        let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs = vec![5.0f32, 6.0, 7.0, 8.0];
        let m = 2;
        let k = 2;
        let n = 2;

        let s_lhs = Cuda::init(&lhs).unwrap();
        let s_rhs = Cuda::init(&rhs).unwrap();

        let s_out = Cuda::gemm(&s_lhs, &s_rhs, 1.0, 0.0, m, n, k);

        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
