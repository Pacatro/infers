use prost::Message;
use std::{fs::File, io::Read};

use infers::{InfersResult, onnx::ModelProto};

fn main() -> InfersResult<()> {
    let mut file = File::open("onnx_models/mnist_fc_model.onnx")?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Decodificar el buffer protobuf
    let model = ModelProto::decode(&*buffer)?;

    // Por ejemplo: obtener el grafo, número de nodos, tensores inicializadores (pesos)
    if let Some(graph) = model.graph.as_ref() {
        println!("Graph name: {:?}", graph.name);
        println!("Number of nodes: {}", graph.node.len());
        println!(
            "Number of initializers (weights): {}",
            graph.initializer.len()
        );
    } else {
        println!("No graph in the model");
    }

    Ok(())
}
