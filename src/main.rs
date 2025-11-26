mod backends;
mod tensor;

use tensor::Tensor;

use crate::backends::Cpu;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    let t1 = Tensor::<Cpu, f32>::zeros(&[2, 2]);
    println!("{t1}");
}
