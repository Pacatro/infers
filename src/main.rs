use anyhow::Result;
use std::time::Instant;

use infers::{
    Tensor,
    backends::{Backend, Cpu},
    core::InfersSession,
};

const MODEL_PATH: &str = "onnx_models/iris_model.onnx";

fn run_inference<B: Backend>() -> Result<Tensor<B>> {
    let mut session = InfersSession::new(MODEL_PATH)?;
    let input = Tensor::new(&[0.3545, -0.5851, 0.5578, 0.0222], [1, 4])?.to::<B>()?;
    println!("Input:\n{}", input);
    let output = session.run(input)?;

    Ok(output)
}

fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use infers::backends::Cuda;
        let now = Instant::now();
        let output = run_inference::<Cuda>()?;
        println!("Time taken: {:?}", now.elapsed());
        println!("Output: {}", output);
    }

    let now = Instant::now();
    let output = run_inference::<Cpu>()?;
    println!("Time taken: {:?}", now.elapsed());
    println!("Output: {}", output);
    Ok(())
}
