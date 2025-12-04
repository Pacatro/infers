# TODOs

Inspirations:

- <https://michalpitr.substack.com/p/build-your-own-inference-engine-from>
- <https://github.com/MichalPitr/inference_engine/tree/main>
- <https://dev.to/cemonix/building-a-cuda-accelerated-neural-network-library-in-rust-b90>

## Main workflow

- [x] Tensor structure
- [x] CUDA support
- [x] Basic Operations
  - [x] Add
  - [x] Relu
  - [x] Flatten
  - [x] Gemm
- [x] Compile ONNX protobufs
  - [x] Load ONNX model
- [ ] Construct computational graph (Parsing protos from ONNX to own implementations)
  - [ ] Attributes (Ops parameters)
  - [ ] Nodes (Operations)
  - [ ] Tensors
- [ ] Run inference with user inputs
  - [ ] Session
- [ ] Display results

## Additional features

- [ ] HTTP server
