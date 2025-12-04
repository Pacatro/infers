use prost::Message;
use std::{fs::File, io::Read};

use infers::{InfersResult, onnx::ModelProto};

const MODEL_PATH: &str = "onnx_models/mnist_fc_model.onnx";

fn main() -> InfersResult<()> {
    let mut file = File::open(MODEL_PATH)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let model = ModelProto::decode(&*buffer)?;

    if let Some(graph) = model.graph.as_ref() {
        for init in graph.initializer.iter() {
            println!("Node name: {:?}", init.name);
            if !init.float_data.is_empty() {
                println!("Found float data");
                println!("{:?}", init.float_data);
            } else if let Some(data) = init.raw_data.as_ref() {
                // The raw data is a sequence of bytes, each representing a float.
                // If we want to convert it to a vector of f32, we need to
                // get the bytes in chunks of 4 (4*8 = 32) and then convert them using little-endian.
                let floats = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<f32>>();
                println!("Found raw data");
                println!("{:?}", floats);
            }
        }
    }

    Ok(())
}
