use infers::{InferenceSession, InfersResult, backends::Cpu};

const MODEL_PATH: &str = "onnx_models/mnist_fc_model.onnx";

fn main() -> InfersResult<()> {
    let session = InferenceSession::<Cpu>::new(MODEL_PATH)?;
    for n in session.graph.iter() {
        dbg!(n);
    }
    Ok(())
}
