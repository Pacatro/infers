# InfeRS

A basic inference engine for ONNX models written in Rust.

> [!NOTE]
> This project is still under development, is not finished yet. There is no demo available at the moment, but I plan to add one in the future.

## Project Goal

`InfeRS` is an experimental project focused on building a small ONNX inference engine from scratch in Rust. The current codebase is centered around loading ONNX models, constructing an internal computation graph, and executing a limited set of operations on different backends.

## Current Features

- ONNX protobuf compilation during build time
- ONNX model loading from file
- Internal graph construction and topological sorting
- Tensor structure with basic operations
- CPU backend
- Optional CUDA backend behind the `cuda` feature
- Support for these operations: `Add`, `Gemm`, `Flatten`, `Relu`

## Current Status

The project already includes the core pieces needed to experiment with ONNX inference, but it is still incomplete.

Some parts are implemented and usable for development, while others are still being refined. In particular:

- the project is not finished yet
- there is no public demo at this moment
- the operation set is still limited
- the workflow is still evolving

I plan to improve this over time and add a proper demo in the future.

## Setup

Clone the repository and build it with Cargo:

```terminal
cargo build
```

If you want to enable CUDA support, build with the `cuda` feature:

```terminal
cargo build --features cuda
```

## Usage

The project currently exposes an `InfersSession` that loads an ONNX model and runs inference on an input tensor.

```rust
use infers::{
    Tensor,
    backends::Cpu,
    core::InfersSession,
};
use anyhow::Result;

fn main() -> Result<()> {
    let mut session = InfersSession::<Cpu>::new("model.onnx")?;
    let input = Tensor::new(&[0.3545, -0.5851, 0.5578, 0.0222], &[1, 4]).to::<Cpu>()?;
    let output = session.run(input)?;

    println!("{}", output);
    Ok(())
}
```

## Project Structure

- `src/tensor`: tensor implementation and operations
- `src/graph`: graph, nodes, attributes, and ONNX operation mapping
- `src/core`: session and error handling
- `src/backends`: CPU backend and optional CUDA backend
- `utils/`: helper files, ONNX protobufs, and training utilities

## Inspiration

- <https://michalpitr.substack.com/p/build-your-own-inference-engine-from>
- <https://github.com/MichalPitr/inference_engine/tree/main>
- <https://dev.to/cemonix/building-a-cuda-accelerated-neural-network-library-in-rust-b90>

## License

[MIT](https://opensource.org/license/mit/) - Created by [**Paco Algar**](https://github.com/Pacatro).
