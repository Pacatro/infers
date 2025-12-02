mod backends;
mod tensor;

use crate::backends::Cuda;
use std::time::Instant;
use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> InfersResult<()> {
    let t1 = Tensor::rand(&[500, 500, 500]);
    let t2 = Tensor::rand(&[500, 500, 500]);
    let t1_gpu = t1.to::<Cuda>()?;
    let t2_gpu = t2.to::<Cuda>()?;

    println!("t1: {}", t1);
    println!("t1: {}", t1);

    println!("Running on CPU");
    let start = Instant::now();
    let t3 = t1 + t2;
    let duration = start.elapsed();
    println!("Duration: {:?}", duration);
    println!("{t3}\n");

    #[cfg(feature = "cuda")]
    {
        println!("Running on CUDA");
        let start = Instant::now();
        let t3 = t1_gpu + t2_gpu;
        let duration = start.elapsed();
        println!("Duration: {:?}", duration);
        println!("{t3}");
    }

    Ok(())
}
