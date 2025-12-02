mod backends;
mod tensor;

use crate::backends::Cuda;
use std::time::Instant;
use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> InfersResult<()> {
    let start = Instant::now();
    let t1 = Tensor::rand(&[700, 700, 700]);
    let t2 = Tensor::rand(&[700, 700, 700]);
    println!("Duration: {:?}", start.elapsed());
    let t1_gpu = t1.to::<Cuda>()?;
    let t2_gpu = t2.to::<Cuda>()?;

    println!("Running on CPU");
    let start = Instant::now();
    let _ = t1 + t2;
    println!("Duration: {:?}", start.elapsed());

    #[cfg(feature = "cuda")]
    {
        println!("Running on CUDA");
        let start = Instant::now();
        let _ = t1_gpu + t2_gpu;
        println!("Duration: {:?}", start.elapsed());
        // println!("{t3}");
    }

    Ok(())
}
