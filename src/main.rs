mod backends;
mod tensor;

use tensor::Tensor;

use crate::backends::{Cpu, Cuda};

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    let t_cpu = Tensor::<Cpu, i32>::zeros(&[2, 2]);
    let t_gpu = t_cpu.to::<Cuda>().unwrap();
    println!("{}", t_gpu);
    println!("{}", t_cpu);
}
