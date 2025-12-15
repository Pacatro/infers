use std::{collections::HashMap, str::FromStr};

use crate::{
    core::{InfersError, InfersResult},
    graph::{Attribute, AttributeValue, OpType},
    onnx::NodeProto,
};

/// Represents a node in a computation graph.
#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    /// Unique name of the node.
    pub name: String,
    /// Type of operation this node performs.
    pub op_type: OpType,
    /// List of input tensor names connected to this node.
    pub input: Vec<String>,
    /// List of output tensor names produced by this node.
    pub output: Vec<String>,
    /// Key-value pairs of attributes that configure the node's operation.
    attributes: HashMap<String, Attribute>,
}

impl Node {
    /// Returns the value of the attribute with the given name, if it exists.
    pub fn get_attribute(&self, name: &str) -> Option<AttributeValue> {
        self.attributes.get(name).map(|attr| attr.value.clone())
    }
}

impl TryFrom<&NodeProto> for Node {
    type Error = InfersError;

    /// Creates a new `Node` from the given `NodeProto`.
    ///
    /// # Errors
    ///
    /// Returns an error if the `NodeProto` is invalid.
    fn try_from(node_proto: &NodeProto) -> InfersResult<Node> {
        let name = node_proto.name().to_string();
        let op_type = OpType::from_str(node_proto.op_type())?;
        let input = node_proto.input.clone();
        let output = node_proto.output.clone();

        let mut attributes = HashMap::new();
        for attr_proto in node_proto.attribute.iter() {
            attributes.insert(
                attr_proto.name().to_string(),
                Attribute::try_from(attr_proto)?,
            );
        }

        Ok(Self {
            name,
            op_type,
            input,
            output,
            attributes,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::AttributeValue,
        onnx::{AttributeProto, NodeProto, attribute_proto::AttributeType},
    };

    use super::*;

    #[test]
    fn test_node_from_proto() {
        let attr_proto = AttributeProto {
            name: Some("test".to_string()),
            r#type: Some(AttributeType::Float.into()),
            f: Some(1.0),
            ..Default::default()
        };

        let node_proto = NodeProto {
            name: Some("test".to_string()),
            op_type: Some("Add".to_string()),
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![attr_proto],
            ..Default::default()
        };

        let node = Node::try_from(&node_proto).unwrap();

        assert_eq!(node.name, "test");
        assert_eq!(node.op_type, OpType::Add);
        assert_eq!(node.input, vec!["input".to_string()]);
        assert_eq!(node.output, vec!["output".to_string()]);
        assert_eq!(node.attributes.len(), 1);
        assert_eq!(
            node.get_attribute("test").unwrap(),
            AttributeValue::Float(1.0)
        );
    }
}
