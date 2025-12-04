fn main() -> Result<(), Box<dyn std::error::Error>> {
    Ok(prost_build::compile_protos(
        &["./utils/protos/onnx-ml.proto"],
        &["./utils/protos"],
    )?)
}
