mod backends;
mod tensor;

use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> InfersResult<()> {
    #[cfg(feature = "cuda")]
    {
        use crate::backends::Cuda;
        println!("Running on CUDA");
        let t1 = Tensor::rand(&[2, 2]).to::<Cuda>()?;
        let t2 = Tensor::rand(&[2, 2]).to::<Cuda>()?;
        println!("{t1}");
        println!("{t2}");
        let t3 = t1 + t2;
        println!("{t3}");
    }

    println!("Running on CPU");
    let t1 = Tensor::rand(&[2, 2]);
    let t2 = Tensor::rand(&[2, 2]);
    println!("t1: {}", t1);
    println!("t1: {}", t1);

    let t3 = t1 + t2;
    println!("{t3}");

    Ok(())
}
