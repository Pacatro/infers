use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

use crate::{
    backends::{Backend, Device, GemmParams},
    core::InfersResult,
    tensor::{Layout, Shape},
};

fn compile_kernel(
    source: &str,
    function_name: &str,
    context: &Arc<CudaContext>,
) -> InfersResult<CudaFunction> {
    let ptx = compile_ptx(source)?;
    let module = context.load_module(ptx)?;
    module.load_function(function_name).map_err(Into::into)
}

fn raw_to_host(storage: &CudaStorage) -> InfersResult<Vec<f32>> {
    storage
        .context
        .default_stream()
        .clone_dtoh(&storage.buffer)
        .map_err(Into::into)
}

fn host_elementwise<F>(
    lhs: &CudaStorage,
    lhs_layout: &Layout,
    rhs: &CudaStorage,
    rhs_layout: &Layout,
    output_shape: &Shape,
    operation: F,
) -> InfersResult<CudaStorage>
where
    F: Fn(f32, f32) -> f32,
{
    let lhs_data = raw_to_host(lhs)?;
    let rhs_data = raw_to_host(rhs)?;
    let output = (0..output_shape.num_elements())
        .map(|index| {
            let lhs_index = lhs_layout.physical_index_from_flat(index, output_shape);
            let rhs_index = rhs_layout.physical_index_from_flat(index, output_shape);
            operation(lhs_data[lhs_index], rhs_data[rhs_index])
        })
        .collect::<Vec<_>>();
    Cuda::from_host(output)
}

fn execute_elementwise_kernel(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    size: usize,
    source: &str,
    function_name: &str,
) -> InfersResult<CudaStorage> {
    let context = Arc::clone(&lhs.context);
    let stream = context.default_stream();
    let function = compile_kernel(source, function_name, &context)?;
    let config = LaunchConfig::for_num_elems(size as u32);
    let mut output = stream.alloc_zeros::<f32>(size)?;

    // SAFETY: all buffers belong to the same CUDA context, contain at least `size`
    // f32 elements, and the launch configuration covers exactly that logical range.
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&lhs.buffer)
            .arg(&rhs.buffer)
            .arg(&mut output)
            .arg(&size)
            .launch(config)?;
    }
    stream.synchronize()?;

    Ok(CudaStorage {
        context,
        buffer: output,
    })
}

#[derive(Debug, Clone)]
pub struct CudaStorage {
    context: Arc<CudaContext>,
    buffer: CudaSlice<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Cuda;

impl Backend for Cuda {
    type Storage = CudaStorage;

    fn device() -> Device {
        Device::Cuda
    }

    fn from_host(data: Vec<f32>) -> InfersResult<Self::Storage> {
        let context = CudaContext::new(0)?;
        let buffer = context.default_stream().clone_htod(&data)?;
        Ok(CudaStorage { context, buffer })
    }

    fn read(storage: &Self::Storage, index: usize) -> InfersResult<f32> {
        let stream = storage.context.default_stream();
        let view = storage.buffer.try_slice(index..index + 1).ok_or_else(|| {
            crate::core::InfersError::Memory("CUDA scalar index is outside the buffer".to_string())
        })?;
        Ok(stream.clone_dtoh(&view)?[0])
    }

    fn to_host(storage: &Self::Storage, layout: &Layout) -> InfersResult<Vec<f32>> {
        let data = raw_to_host(storage)?;
        if layout.is_contiguous() {
            return Ok(data[..layout.shape().num_elements()].to_vec());
        }

        Ok((0..layout.shape().num_elements())
            .map(|index| data[layout.physical_index_from_flat(index, layout.shape())])
            .collect())
    }

    fn add(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage> {
        if Arc::ptr_eq(&lhs.context, &rhs.context)
            && lhs_layout.is_contiguous()
            && rhs_layout.is_contiguous()
            && lhs_layout.shape() == output_shape
            && rhs_layout.shape() == output_shape
        {
            return execute_elementwise_kernel(
                lhs,
                rhs,
                output_shape.num_elements(),
                include_str!("./kernels/add.cu"),
                "add",
            );
        }
        host_elementwise(lhs, lhs_layout, rhs, rhs_layout, output_shape, |a, b| a + b)
    }

    fn sub(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage> {
        if Arc::ptr_eq(&lhs.context, &rhs.context)
            && lhs_layout.is_contiguous()
            && rhs_layout.is_contiguous()
            && lhs_layout.shape() == output_shape
            && rhs_layout.shape() == output_shape
        {
            return execute_elementwise_kernel(
                lhs,
                rhs,
                output_shape.num_elements(),
                include_str!("./kernels/sub.cu"),
                "sub",
            );
        }
        host_elementwise(lhs, lhs_layout, rhs, rhs_layout, output_shape, |a, b| a - b)
    }

    fn mul(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage> {
        if Arc::ptr_eq(&lhs.context, &rhs.context)
            && lhs_layout.is_contiguous()
            && rhs_layout.is_contiguous()
            && lhs_layout.shape() == output_shape
            && rhs_layout.shape() == output_shape
        {
            return execute_elementwise_kernel(
                lhs,
                rhs,
                output_shape.num_elements(),
                include_str!("./kernels/mul.cu"),
                "mul",
            );
        }
        host_elementwise(lhs, lhs_layout, rhs, rhs_layout, output_shape, |a, b| a * b)
    }

    fn relu(input: &Self::Storage, layout: &Layout) -> InfersResult<Self::Storage> {
        if !layout.is_contiguous() {
            let data = Self::to_host(input, layout)?;
            return Self::from_host(
                data.into_iter()
                    .map(|value| value.max(0.0))
                    .collect::<Vec<_>>(),
            );
        }

        let context = Arc::clone(&input.context);
        let stream = context.default_stream();
        let function = compile_kernel(include_str!("./kernels/relu.cu"), "relu", &context)?;
        let size = layout.shape().num_elements();
        let mut output = stream.alloc_zeros::<f32>(size)?;

        // SAFETY: input and output are valid f32 buffers in the same context and both
        // contain at least `size` elements expected by the kernel.
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&input.buffer)
                .arg(&mut output)
                .arg(&size)
                .launch(LaunchConfig::for_num_elems(size as u32))?;
        }
        stream.synchronize()?;
        Ok(CudaStorage {
            context,
            buffer: output,
        })
    }

    fn gemm(params: GemmParams<f32, Self::Storage>) -> InfersResult<Self::Storage> {
        if !Arc::ptr_eq(&params.lhs.context, &params.rhs.context) {
            let lhs = raw_to_host(params.lhs)?;
            let rhs = raw_to_host(params.rhs)?;
            let lhs_strides = params.lhs_layout.strides();
            let rhs_strides = params.rhs_layout.strides();
            let mut output = vec![0.0; params.m * params.n];
            for row in 0..params.m {
                for column in 0..params.n {
                    let mut sum = 0.0;
                    for inner in 0..params.k {
                        sum += lhs[row * lhs_strides[0] + inner * lhs_strides[1]]
                            * rhs[inner * rhs_strides[0] + column * rhs_strides[1]];
                    }
                    output[row * params.n + column] = params.alpha * sum;
                }
            }
            return Self::from_host(output);
        }

        let context = Arc::clone(&params.lhs.context);
        let stream = context.default_stream();
        let function = compile_kernel(include_str!("./kernels/gemm.cu"), "gemm", &context)?;
        let mut output = stream.alloc_zeros::<f32>(params.m * params.n)?;
        let output_row_stride = params.n;
        let output_column_stride = 1;
        let lhs_strides = params.lhs_layout.strides();
        let rhs_strides = params.rhs_layout.strides();
        let block = (16, 16, 1);
        let config = LaunchConfig {
            grid_dim: (
                params.n.div_ceil(block.0) as u32,
                params.m.div_ceil(block.1) as u32,
                1,
            ),
            block_dim: (block.0 as u32, block.1 as u32, 1),
            shared_mem_bytes: 16 * 17 * 4 * 2,
        };

        // SAFETY: validated rank-2 layouts provide both strides, buffers share one
        // context, and the output allocation contains `m * n` f32 elements.
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&params.m)
                .arg(&params.n)
                .arg(&params.k)
                .arg(&params.alpha)
                .arg(&params.lhs.buffer)
                .arg(&lhs_strides[0])
                .arg(&lhs_strides[1])
                .arg(&params.rhs.buffer)
                .arg(&rhs_strides[0])
                .arg(&rhs_strides[1])
                .arg(&params.beta)
                .arg(&mut output)
                .arg(&output_row_stride)
                .arg(&output_column_stride)
                .launch(config)?;
        }
        stream.synchronize()?;
        Ok(CudaStorage {
            context,
            buffer: output,
        })
    }

    fn dot(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> InfersResult<Self::Storage> {
        let lhs = Self::to_host(lhs, lhs_layout)?;
        let rhs = Self::to_host(rhs, rhs_layout)?;
        let result = lhs.into_iter().zip(rhs).map(|(a, b)| a * b).sum::<f32>();
        Self::from_host(vec![result])
    }
}
