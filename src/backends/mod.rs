mod backend;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

pub use backend::{Backend, Device};
pub use cpu::Cpu;
#[cfg(feature = "cuda")]
pub use cuda::Cuda;
