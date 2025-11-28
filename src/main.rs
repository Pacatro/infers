mod backends;
mod tensor;

use tensor::Tensor;

use crate::backends::Cuda;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    let t_cpu = Tensor::rand(&[2, 2]);
    let t_gpu = t_cpu.to::<Cuda>().unwrap();
    println!("{}", t_cpu);
    println!("{}", t_gpu);

    let mut t1 = Tensor::rand(&[2, 2]);
    println!("t1: {}", t1);
    let t2 = Tensor::rand(&[2, 2]);
    println!("t2: {}", t2);
    t1 += t2;
    println!("t1: {}", t1);
}
