pub mod backends;
pub mod core;
pub mod graph;
mod tensor;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use tensor::{Layout, Shape, Tensor};
