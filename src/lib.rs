pub mod backends;
pub mod error;
pub mod graph;
mod session;
mod tensor;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use session::InfersSession;
pub use tensor::Tensor;
// pub use graph::Graph;

pub use error::{InfersError, InfersResult};
