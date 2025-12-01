mod backends;
mod tensor;

use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    #[cfg(feature = "cuda")]
    {
        use crate::backends::Cuda;
        let t1 = Tensor::rand(&[2, 2]).to::<Cuda>().unwrap();
        let t2 = Tensor::rand(&[2, 2]).to::<Cuda>().unwrap();

        let t3 = t1 + t2;
        println!("{t3}");
    }

    let t1 = Tensor::rand(&[2, 2]);
    let t2 = Tensor::rand(&[2, 2]);
    println!("t1: {}", t1);
    println!("t1: {}", t1);

    let t3 = t1 + t2;
    println!("{t3}");
    println!("Tensor 3 shape: {:?}", t3.shape);
}
