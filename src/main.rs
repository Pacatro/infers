mod backends;
mod tensor;

use std::time::Instant;
use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn test_time() -> InfersResult<()> {
    let t1 = Tensor::rand(&[3000, 3000]);
    let t2 = Tensor::rand(&[3000, 3000]);

    let start = Instant::now();
    let _ = t1.matmul(t2);
    println!("Matmul (CPU) duration: {:?}", start.elapsed());

    #[cfg(feature = "cuda")]
    {
        use crate::backends::Cuda;
        let t1 = Tensor::rand(&[3000, 3000]).to::<Cuda>()?;
        let t2 = Tensor::rand(&[3000, 3000]).to::<Cuda>()?;
        let start = Instant::now();
        let _ = t1.matmul(t2);
        println!("Matmul (CUDA) duration: {:?}", start.elapsed());
        // println!("{t3}");
    }

    Ok(())
}
fn main() -> InfersResult<()> {
    test_time()
}
