fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(prost_build::compile_protos(
        &["protos/onnx-ml.proto"],
        &["protos/"],
    )?)
}
