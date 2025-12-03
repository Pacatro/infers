pub mod backends;
mod tensor;

pub use tensor::Tensor;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
