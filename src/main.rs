use std::time::Instant;

use infers::InfersResult;
use infers::Tensor;

fn test_time() -> InfersResult<()> {
    let t1 = Tensor::rand(&[1000, 1000]);
    let t2 = Tensor::rand(&[1000, 1000]);

    let start = Instant::now();
    let _ = t1.matmul(&t2);
    println!("Matmul (CPU) duration: {:?}", start.elapsed());

    #[cfg(feature = "cuda")]
    {
        use infers::backends::Cuda;
        let t1 = t1.to::<Cuda>()?;
        let t2 = t2.to::<Cuda>()?;
        let start = Instant::now();
        let _ = t1.matmul(&t2);
        println!("Matmul (CUDA) duration: {:?}", start.elapsed());
        // println!("{t3}");
    }

    Ok(())
}

fn main() -> InfersResult<()> {
    test_time()
}
