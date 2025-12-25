use prost::Message;
use std::{collections::HashMap, fs::File, io::Read};

use crate::{
    Tensor,
    backends::{Backend, Device},
    core::InfersError,
    core::InfersResult,
    graph::{AttributeValue, Graph, Node, OpType},
    onnx::{ModelProto, TensorProto},
};

#[derive(Debug, Clone)]
pub struct InfersSession<B: Backend> {
    pub model_path: String,
    pub graph: Graph,
    pub weights: HashMap<String, Tensor<B>>,
    pub device: Device,
}

impl<B> InfersSession<B>
where
    B: Backend,
{
    pub fn new(model_path: &str) -> InfersResult<Self> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = ModelProto::decode(&*buffer)?;

        let Some(graph_proto) = model.graph.as_ref() else {
            return Err(InfersError::OnnxFormat(
                "model does not have a graph".to_string(),
            ));
        };

        if graph_proto.node.is_empty() {
            return Err(InfersError::OnnxFormat("graph has no nodes".to_string()));
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

    pub fn run(&mut self, input: Tensor<B>) -> InfersResult<Tensor<B>> {
        self.weights.insert(self.graph.inputs[0].to_string(), input);

        for node in self.graph.iter() {
            let inputs = self.prepare_inputs(node);
            let output = self.evaluate_node(node, inputs)?;

            if output.is_empty() {
                return Err(InfersError::Tensor("Output tensor is empty".to_string()));
            }

            self.weights.insert(node.output[0].to_string(), output);
        }

        let Some(output) = self.weights.get(&self.graph.outputs[0]) else {
            return Err(InfersError::Tensor("Output tensor is empty".to_string()));
        };

        Ok(output.clone())
    }

    fn load_weights(initializer: &[TensorProto]) -> InfersResult<HashMap<String, Tensor<B, f32>>> {
        let mut weights = HashMap::new();

        for init in initializer.iter() {
            let dims: Vec<usize> = init.dims.iter().map(|x| *x as usize).collect();

            if !init.float_data.is_empty() {
                weights.insert(
                    init.name().to_string(),
                    Tensor::<B>::from_data(&init.float_data, dims.as_slice())?,
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

                weights.insert(
                    init.name().to_string(),
                    Tensor::<B>::from_data(&data?, &dims)?,
                );
            }
        }

        Ok(weights)
    }

    fn evaluate_node(&self, node: &Node, inputs: Vec<Tensor<B>>) -> InfersResult<Tensor<B>> {
        match node.op_type {
            OpType::Add => {
                if inputs.len() != 2 {
                    return Err(InfersError::Operation(
                        "Invalid number of inputs for Add operation".to_string(),
                    ));
                }

                let lhs = &inputs[0];
                let rhs = &inputs[1];

                Ok(lhs.add(rhs))
            }
            OpType::Gemm => {
                if inputs.len() > 3 || inputs.len() < 2 {
                    return Err(InfersError::Operation(
                        "Invalid number of inputs for Gemm operation".to_string(),
                    ));
                }

                // Checks if an attribute with the given name is set to 1
                let is_transposed = |attr_name: &str| -> bool {
                    matches!(node.get_attribute(attr_name), Some(attr) if matches!(attr.value, AttributeValue::Int64(x) if x != 0))
                };

                // Get the attribute value
                let get_attr = |attr_name: &str| -> Option<f32> {
                    match node.get_attribute(attr_name) {
                        Some(attr) => match &attr.value {
                            AttributeValue::Float(alpha) => Some(*alpha),
                            _ => None,
                        },
                        _ => None,
                    }
                };

                let alpha = get_attr("alpha");
                let beta = get_attr("beta");

                let trans_a = is_transposed("transA");
                let trans_b = is_transposed("transB");

                // SAFETY: We have already checked that inputs.len() > 2
                let lhs = &inputs[0];
                let rhs = &inputs[1];

                let mm = match (trans_a, trans_b) {
                    (false, false) => lhs.gemm(rhs, alpha, beta),
                    (false, true) => lhs.gemm(&rhs.t(), alpha, beta),
                    (true, false) => lhs.t().gemm(rhs, alpha, beta),
                    (true, true) => lhs.t().gemm(&rhs.t(), alpha, beta),
                };

                // Add bias if it exists
                let bias = inputs.get(2);
                match bias {
                    Some(b) => Ok(mm.add(b)),
                    None => Ok(mm),
                }
            }
            OpType::Flatten => {
                if inputs.is_empty() || inputs.len() != 1 {
                    return Err(InfersError::Operation(
                        "Invalid number of inputs for Flatten operation".to_string(),
                    ));
                }

                // SAFETY: We have already checked that the input is a single tensor
                Ok(inputs[0].flatten())
            }
            OpType::Relu => {
                if inputs.is_empty() || inputs.len() != 1 {
                    return Err(InfersError::Operation(
                        "Invalid number of inputs for Relu operation".to_string(),
                    ));
                }

                Ok(inputs[0].relu())
            }
            _ => Err(InfersError::Operation("Invalid operation".to_string())),
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

        let weights = InfersSession::<Cpu>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size(), 4);
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

        let weights = InfersSession::<Cpu>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size(), 4);
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

        let weights = InfersSession::<Cuda>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size(), 4);
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

        let weights = InfersSession::<Cuda>::load_weights(&[tensor_proto]).unwrap();

        assert_eq!(weights.len(), 1);
        let tensor = &weights["tensor"];
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.strides, &[2, 1]);
        assert_eq!(tensor.data().unwrap(), orig);
        assert_eq!(tensor.device(), Device::Cuda);
    }
}
