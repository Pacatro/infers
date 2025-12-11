use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

use std::{fmt::Debug, sync::Arc};

use crate::{
    backends::{Backend, Device, GemmParams},
    core::InfersResult,
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

/// Performs a general elementwise operation between two tensors
fn execute_elementwise_op(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    size: usize,
    src: &str,
    func_name: &str,
) -> InfersResult<CudaStorage> {
    let ctx = lhs.context.clone();
    let stream = ctx.default_stream();

    let func = compile_kernel(src, func_name, &ctx)?;

    let config = LaunchConfig::for_num_elems(size as u32);
    let mut out_device = stream.alloc_zeros::<f32>(size)?;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&lhs.buffer)
            .arg(&rhs.buffer)
            .arg(&mut out_device)
            .arg(&size)
            .launch(config)?;
    }

    stream.synchronize()?;

    Ok(CudaStorage {
        context: ctx,
        buffer: out_device,
    })
}

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

impl Backend for Cuda {
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
        execute_elementwise_op(lhs, rhs, size, include_str!("./kernels/add.cu"), "add").unwrap()
    }

    fn sub(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        execute_elementwise_op(lhs, rhs, size, include_str!("./kernels/sub.cu"), "sub").unwrap()
    }

    fn mul(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        execute_elementwise_op(lhs, rhs, size, include_str!("./kernels/mul.cu"), "mul").unwrap()
    }

    fn relu(input: &Self::Storage, size: usize) -> Self::Storage {
        let ctx = input.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("./kernels/relu.cu"), "relu", &ctx).unwrap();

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

    fn gemm(params: GemmParams<f32, Self::Storage>) -> Self::Storage {
        let ctx = params.lhs.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("./kernels/gemm.cu"), "gemm", &ctx).unwrap();

        let mut c = stream.alloc_zeros::<f32>(params.m * params.n).unwrap();
        let c_row_stride = params.n;
        let c_col_stride = 1;

        let block = (16, 16, 1);
        let grid = (params.n.div_ceil(block.0), params.m.div_ceil(block.1), 1);

        let config = LaunchConfig {
            grid_dim: (grid.0 as u32, grid.1 as u32, 1),
            block_dim: (block.0 as u32, block.1 as u32, 1),

            // TILE_SIZE * (TILE_SIZE + 1) * sizeof(float) * 2 matrices (As and Bs)
            // TILE_SIZE = 16, float size = 4 bytes
            // 16 * 17 * 4 * 2 = 2176 bytes
            shared_mem_bytes: (16 * 17 * 4 * 2),
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&params.m)
                .arg(&params.n)
                .arg(&params.k)
                .arg(&params.alpha)
                .arg(&params.lhs.buffer)
                .arg(&params.lhs_strides[0])
                .arg(&params.lhs_strides[1])
                .arg(&params.rhs.buffer)
                .arg(&params.rhs_strides[0])
                .arg(&params.rhs_strides[1])
                .arg(&params.beta)
                .arg(&mut c)
                .arg(&c_row_stride)
                .arg(&c_col_stride)
                .launch(config)
                .unwrap();
        }

        stream.synchronize().unwrap();

        CudaStorage {
            context: ctx,
            buffer: c,
        }
    }

    fn dot(lhs: &Self::Storage, rhs: &Self::Storage, size: usize) -> Self::Storage {
        let ctx = lhs.context.clone();
        let stream = ctx.default_stream();

        let func = compile_kernel(include_str!("./kernels/dot.cu"), "dot", &ctx).unwrap();

        let mut result = stream.alloc_zeros::<f32>(1).unwrap();

        let block_size = 256;
        let grid_size = size.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&lhs.buffer)
                .arg(&rhs.buffer)
                .arg(&mut result)
                .arg(&size)
                .launch(config)
                .unwrap();
        }

        stream.synchronize().unwrap();

        CudaStorage {
            context: ctx,
            buffer: result,
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
        let input = vec![-1.0, -2.0, 3.0, 4.0];
        let s_input = Cuda::init(&input).unwrap();
        let s_out = Cuda::relu(&s_input, input.len());
        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_backend_gemm_cuda_2x2() {
        // A = [1 2; 3 4], B = [5 6; 7 8]
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [5.0, 6.0, 7.0, 8.0];
        let m = 2;
        let k = 2;
        let n = 2;
        let lhs_strides = [2, 1];
        let rhs_strides = [2, 1];

        let s_lhs = Cuda::init(&lhs).unwrap();
        let s_rhs = Cuda::init(&rhs).unwrap();

        let s_out = Cuda::gemm(GemmParams {
            lhs: &s_lhs,
            rhs: &s_rhs,
            lhs_strides: lhs_strides.to_vec(),
            rhs_strides: rhs_strides.to_vec(),
            alpha: 1.0,
            beta: 0.0,
            m,
            n,
            k,
        });

        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_backend_dot_cuda() {
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![5.0, 6.0, 7.0, 8.0];
        let size = lhs.len();
        let s_lhs = Cuda::init(&lhs).unwrap();
        let s_rhs = Cuda::init(&rhs).unwrap();
        let s_out = Cuda::dot(&s_lhs, &s_rhs, size);
        let result = Cuda::copy_to_host(&s_out).unwrap();
        assert_eq!(result, vec![70.0]);
    }
}
