mod backends;
mod tensor;

use tensor::Tensor;

use crate::backends::CpuBackend;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    let t1 = Tensor::<CpuBackend, f32>::new(&[2], &[1., 2.]).unwrap();
    let t2 = Tensor::<CpuBackend, f32>::new(&[2], &[3., 4.]).unwrap();
    let t3 = t1 + t2;
    println!("{:?}", t3.storage());
}
