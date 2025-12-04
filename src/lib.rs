pub mod backends;
pub mod graph;
mod tensor;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use tensor::Tensor;
// pub use graph::Graph;

pub type InfersResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
