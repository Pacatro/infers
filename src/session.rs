use prost::Message;
use std::{collections::HashMap, fs::File, io::Read};

use crate::{
    InfersResult, Tensor,
    backends::{Backend, Device},
    graph::{Graph, Node, OpType},
    onnx::{ModelProto, TensorProto},
};

#[derive(Debug, Clone)]
pub struct InferenceSession<B>
where
    B: Backend<f32>,
{
    pub model_path: String,
    pub graph: Graph,
    pub weights: HashMap<String, Tensor<B, f32>>,
    pub device: Device,
}

impl<B> InferenceSession<B>
where
    B: Backend<f32>,
{
    pub fn new(model_path: &str) -> InfersResult<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = ModelProto::decode(&*buffer)?;

        let Some(graph_proto) = model.graph.as_ref() else {
            return Err("Invalid ONNX format, model does not have a graph".into());
        };

        if graph_proto.node.is_empty() {
            return Err("Invalid ONNX format, graph has no nodes".into());
        }

        let weights = Self::load_weights(&graph_proto.initializer)?;
        let graph = Graph::try_from(graph_proto)?;

        Ok(Self {
            model_path: model_path.to_string(),
            graph,
            weights,
            device: B::device(),
        })
    }

    fn load_weights(initializer: &[TensorProto]) -> InfersResult<HashMap<String, Tensor<B, f32>>> {
        let mut weights = HashMap::new();

        for init in initializer.iter() {
            let dims: Vec<usize> = init.dims.iter().map(|x| *x as usize).collect();

            if !init.float_data.is_empty() {
                weights.insert(
                    init.name().to_string(),
                    Tensor::<B, f32>::from_data(&init.float_data, dims.as_slice())?,
                );
            } else if let Some(data) = init.raw_data.as_ref() {
                // The raw data is a sequence of bytes, each representing a float.
                // If we want to convert it to a vector of f32, we need to
                // get the bytes in chunks of 4 (4*8 = 32) and then convert them using little-endian.
                let data: InfersResult<Vec<f32>> = data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let bytes = chunk.try_into()?;
                        Ok(f32::from_le_bytes(bytes))
                    })
                    .collect();

                let expected_len: usize = dims.iter().product();
                let data = data?;

                // FIXME: I don't know if this is correct, the problema is that for some reason,
                // the graph proto form onnx has two global inputs (x and val_3)
                // maybe the second input refers to the validation input (i don't know)
                let shape = if expected_len == data.len() {
                    dims
                } else {
                    vec![data.len()]
                };

                weights.insert(
                    init.name().to_string(),
                    Tensor::<B, f32>::from_data(&data, shape.as_slice())?,
                );
            }
        }

        Ok(weights)
    }

    pub fn run(&mut self, input: Tensor<B, f32>) -> InfersResult<()> {
        self.weights.insert(self.graph.inputs[0].to_string(), input);
        for node in self.graph.iter() {
            let inputs = self.prepare_inputs(node);
            let output = self.evaluate_node(node, inputs)?;

            if output.is_empty() {
                return Err("Output tensor is empty".into());
            }

            self.weights.insert(node.output[0].to_string(), output);
        }

        Ok(())
    }

    fn evaluate_node(
        &self,
        node: &Node,
        inputs: Vec<Tensor<B, f32>>,
    ) -> InfersResult<Tensor<B, f32>> {
        match node.op_type {
            OpType::Add => {
                if inputs.len() != 2 {
                    return Err("Invalid number of inputs for Add operation".into());
                }

                let lhs = &inputs[0];
                let rhs = &inputs[1];

                Ok(lhs.add(rhs))
            }
            OpType::Gemm => {
                if inputs.len() != 3 {
                    return Err("Invalid number of inputs for Gemm operation".into());
                }

                let lhs = &inputs[0];
                let rhs = &inputs[1];
                let bias = &inputs[2];

                Ok(lhs.matmul(rhs).add(bias))
            }
            OpType::Flatten => {
                if inputs.len() != 1 {
                    return Err("Invalid number of inputs for Flatten operation".into());
                }

                Ok(inputs[0].flatten())
            }
            OpType::Relu => {
                if inputs.len() != 1 {
                    return Err("Invalid number of inputs for Relu operation".into());
                }

                Ok(inputs[0].relu())
            }
            _ => Err("Invalid operation".into()),
        }
    }

    fn prepare_inputs(&self, node: &Node) -> Vec<Tensor<B, f32>> {
        let mut inputs = vec![];

        for input_name in &node.input {
            if let Some(tensor) = self.weights.get(input_name) {
                inputs.push(tensor.clone());
            }
        }

        inputs
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::Cpu;
    #[cfg(feature = "cuda")]
    use crate::backends::Cuda;

    use super::*;

    // TODO: Create test for method `new`

    #[test]
    fn test_load_weights_float_data_cpu() {
        let tensor_proto = TensorProto {
            name: Some("tensor".into()),
            dims: vec![2, 2],
            float_data: vec![1., 2., 3., 4.],
            ..Default::default()
        };

        let weights = InferenceSession::<Cpu>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.strides, &[2, 1]);
        assert_eq!(tensor.data().unwrap(), vec![1., 2., 3., 4.]);
        assert!(tensor.device() == Device::Cpu);
    }

    #[test]
    fn test_load_weights_raw_data_cpu() {
        let orig: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let raw_data = orig
            .iter()
            .flat_map(|&x| x.to_le_bytes().to_vec())
            .collect::<Vec<u8>>();

        let tensor_proto = TensorProto {
            name: Some("tensor".into()),
            dims: vec![2, 2],
            raw_data: Some(raw_data),
            ..Default::default()
        };

        let weights = InferenceSession::<Cpu>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.strides, &[2, 1]);
        assert_eq!(tensor.data().unwrap(), orig);
        assert!(tensor.device() == Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_load_weights_float_data_cuda() {
        let tensor_proto = TensorProto {
            name: Some("tensor".into()),
            dims: vec![2, 2],
            float_data: vec![1., 2., 3., 4.],
            ..Default::default()
        };

        let weights = InferenceSession::<Cuda>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.strides, &[2, 1]);
        assert_eq!(tensor.data().unwrap(), vec![1., 2., 3., 4.]);
        assert!(tensor.device() == Device::Cuda);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_load_weights_raw_data_cuda() {
        let orig: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let raw_data = orig
            .iter()
            .flat_map(|&x| x.to_le_bytes().to_vec())
            .collect::<Vec<u8>>();

        let tensor_proto = TensorProto {
            name: Some("tensor".into()),
            dims: vec![2, 2],
            raw_data: Some(raw_data),
            ..Default::default()
        };

        let weights = InferenceSession::<Cuda>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.strides, &[2, 1]);
        assert_eq!(tensor.data().unwrap(), orig);
        assert_eq!(tensor.device(), Device::Cuda);
    }
}
