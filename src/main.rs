mod backends;
mod tensor;

use std::time::Instant;
use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn test_time() -> InfersResult<()> {
    let start = Instant::now();
    let t1 = Tensor::rand(&[700, 700, 700]);
    let t2 = Tensor::rand(&[700, 700, 700]);
    println!("Rand duration: {:?}", start.elapsed());

    println!("Running on CPU");
    let start = Instant::now();
    let _ = t1 + t2;
    println!("Add duration: {:?}", start.elapsed());

    #[cfg(feature = "cuda")]
    {
        use crate::backends::Cuda;
        let t1 = Tensor::rand(&[700, 700, 700]).to::<Cuda>()?;
        let t2 = Tensor::rand(&[700, 700, 700]).to::<Cuda>()?;
        println!("Running on CUDA");
        let start = Instant::now();
        let _ = t1 + t2;
        println!("Duration: {:?}", start.elapsed());
        // println!("{t3}");
    }

    Ok(())
}
fn main() -> InfersResult<()> {
    test_time()?;
    #[cfg(feature = "cuda")]
    {
        use crate::backends::Cuda;
        let t = Tensor::rand(&[2, 2]).to::<Cuda>()?;
        let t = t.relu();
        println!("{t}");
    }

    Ok(())
}
