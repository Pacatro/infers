use std::collections::{HashMap, HashSet};

use crate::{InfersResult, graph::Node, onnx::GraphProto};

/// Representation of a node inside the computational graph along with
/// its connectivity information.
///
/// `NodeInfo` holds the `Node` value and lists of the node names for
/// its immediate children (nodes that consume this node's outputs)
/// and parents (nodes that produce inputs for this node).
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// The underlying node (operator + metadata).
    pub node: Node,
    /// Names of nodes that depend on this node (outgoing edges).
    pub children: Vec<String>,
    /// Names of nodes that this node depends on (incoming edges).
    pub parents: Vec<String>,
}

/// In-memory representation of a computational graph.
#[derive(Default, Debug, Clone)]
pub struct Graph {
    /// Global input tensor names of the graph (graph-level inputs).
    pub inputs: Vec<String>,
    /// Global output tensor names of the graph (graph-level outputs).
    pub outputs: Vec<String>,
    /// Mapping from node name to node information (connectivity + node).
    pub nodes: HashMap<String, NodeInfo>,
    /// Nodes ordered in topological order (producers before consumers).
    pub sorted_nodes: Vec<Node>,
}

impl Graph {
    /// Adds a node to the graph.
    pub fn add_node(&mut self, node: &Node) {
        let info = NodeInfo {
            node: node.clone(),
            children: vec![],
            parents: vec![],
        };

        self.nodes.insert(node.name.to_string(), info);
    }

    /// Checks if a node is an input node.
    fn is_input_node(&self, node: &Node) -> bool {
        node.input
            .iter()
            .any(|input_name| self.inputs.contains(input_name))
    }

    /// Topological sort of the graph.
    ///
    /// The topological sort is a linear ordering of the nodes such that
    /// all producers of a node appear before the node itself.
    ///
    /// ## Example
    ///
    /// ```text
    ///     A
    ///    / \
    ///   B   C
    ///    \ /
    ///     D
    /// ```
    ///
    /// The topological sort of this graph can be `[A, B, C, D]` or `[A, C, B, D]`
    fn topological_sort(&mut self) {
        let mut visited = HashSet::new();
        let mut stack = vec![];

        for (name, info) in self.nodes.iter() {
            if (info.parents.is_empty() || self.is_input_node(&info.node))
                && !visited.contains(name)
            {
                self.visit(name, &mut visited, &mut stack);
            }
        }

        self.sorted_nodes = stack
    }

    /// Check if a node has been visited.
    ///
    /// If not, we visit the children of the node recursively and add them to the stack.
    fn visit(&self, node_name: &str, visited: &mut HashSet<String>, stack: &mut Vec<Node>) {
        visited.insert(node_name.to_string());

        if let Some(node_info) = self.nodes.get(node_name) {
            for child_name in node_info.children.iter() {
                if !visited.contains(child_name) {
                    self.visit(child_name, visited, stack);
                }
            }
            stack.insert(0, node_info.node.clone());
        }
    }
}

impl TryFrom<&GraphProto> for Graph {
    type Error = Box<dyn std::error::Error>;

    fn try_from(graph_proto: &GraphProto) -> InfersResult<Self> {
        let mut graph = Self::default();

        // Load inputs and outputs
        for input in graph_proto.input.iter() {
            graph.inputs.push(input.name().to_string());
        }

        for output in graph_proto.output.iter() {
            graph.outputs.push(output.name().to_string());
        }

        // Load nodes to graph
        let mut tensor_to_producer = HashMap::<String, String>::new();

        for node_proto in graph_proto.node.iter() {
            let node = Node::try_from(node_proto)?;

            for output in node.output.iter() {
                tensor_to_producer.insert(output.to_string(), node.name.to_string());
            }

            graph.add_node(&node);
        }

        // Connect edges
        let mut updates: Vec<(String, String)> = vec![];

        for (child_name, info) in graph.nodes.iter_mut() {
            for input_name in &info.node.input {
                if let Some(producer_name) = tensor_to_producer.get(input_name) {
                    updates.push((producer_name.to_string(), child_name.to_string()));
                }
            }
        }

        for (parent, child) in updates {
            if let Some(p_info) = graph.nodes.get_mut(&parent) {
                p_info.children.push(child.clone());
            }

            if let Some(c_info) = graph.nodes.get_mut(&child) {
                c_info.parents.push(parent);
            }
        }

        graph.topological_sort();

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::OpType,
        onnx::{NodeProto, ValueInfoProto},
    };

    use super::*;

    fn create_node(
        name: &str,
        op_type: &str,
        input: &[&str],
        output: &[&str],
    ) -> (Node, NodeProto) {
        let node_proto = NodeProto {
            name: Some(name.to_string()),
            op_type: Some(op_type.to_string()),
            input: input.iter().map(|s| s.to_string()).collect(),
            output: output.iter().map(|s| s.to_string()).collect(),
            attribute: vec![],
            ..Default::default()
        };

        (Node::try_from(&node_proto).unwrap(), node_proto)
    }

    fn build_graph_from_nodes(nodes: Vec<Node>, global_inputs: &[&str]) -> Graph {
        let mut graph = Graph {
            inputs: global_inputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };

        let mut tensor_to_producer: HashMap<String, String> = HashMap::new();
        for node in &nodes {
            graph.add_node(node);
            for out in &node.output {
                tensor_to_producer.insert(out.clone(), node.name.clone());
            }
        }

        let mut updates: Vec<(String, String)> = Vec::new();
        for (child_name, info) in &graph.nodes {
            for input_tensor in &info.node.input {
                if let Some(parent_name) = tensor_to_producer.get(input_tensor) {
                    updates.push((parent_name.clone(), child_name.clone()));
                }
            }
        }

        for (parent, child) in updates {
            if let Some(p_info) = graph.nodes.get_mut(&parent) {
                p_info.children.push(child.clone());
            }
            if let Some(c_info) = graph.nodes.get_mut(&child) {
                c_info.parents.push(parent.clone());
            }
        }

        graph.topological_sort();
        graph
    }

    #[test]
    fn test_linear_graph_and_linking() {
        // A(in: global_in, out: x) -> B(in: x, out: global_out)
        let (node_a, _) = create_node(
            "A",
            &OpType::Input.to_string(),
            &["input_global"],
            &["tensor_x"],
        );

        let (node_b, _) = create_node(
            "B",
            &OpType::Add.to_string(),
            &["tensor_x"],
            &["output_global"],
        );

        let graph = build_graph_from_nodes(vec![node_a, node_b], &["input_global"]);

        let info_a = graph.nodes.get("A").unwrap();
        assert_eq!(info_a.node.op_type.to_string(), "Input");

        assert!(info_a.children.contains(&String::from("B")),);
        let info_b = graph.nodes.get("B").unwrap();
        assert!(info_b.parents.contains(&String::from("A")),);

        let sorted_names: Vec<&String> = graph.sorted_nodes.iter().map(|n| &n.name).collect();
        assert_eq!(sorted_names, vec![&"A".to_string(), &"B".to_string()],);

        assert_eq!(graph.inputs, vec!["input_global"]);
    }

    #[test]
    fn test_topological_sort_diamond_shape() {
        //      A
        //     / \
        //    B   C
        //     \ /
        //      D
        let (a, _) = create_node("A", &OpType::Add.to_string(), &[], &["t1", "t2"]);
        let (b, _) = create_node("B", &OpType::Relu.to_string(), &["t1"], &["t3"]);
        let (c, _) = create_node("C", &OpType::Gemm.to_string(), &["t2"], &["t4"]);
        let (d, _) = create_node("D", &OpType::Flatten.to_string(), &["t3", "t4"], &[]);

        let graph = build_graph_from_nodes(vec![d, a, b, c], &[]);

        assert_eq!(graph.sorted_nodes.len(), 4);
        let indices: HashMap<String, usize> = graph
            .sorted_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.to_string(), i))
            .collect();

        assert!(indices["A"] < indices["B"]);
        assert!(indices["C"] < indices["D"]);
    }

    #[test]
    fn test_graph_from_proto() {
        let (_, node_a) = create_node(
            "A",
            &OpType::Input.to_string(),
            &["input_global"],
            &["tensor_x"],
        );

        let (_, node_b) = create_node(
            "B",
            &OpType::Add.to_string(),
            &["tensor_x"],
            &["output_global"],
        );

        let graph_proto = GraphProto {
            input: vec![ValueInfoProto {
                name: Some("input_global".into()),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: Some("output_global".into()),
                ..Default::default()
            }],
            node: vec![node_a, node_b],
            ..Default::default()
        };

        let graph = Graph::try_from(&graph_proto).unwrap();

        let info_a = graph.nodes.get("A").unwrap();
        let info_b = graph.nodes.get("B").unwrap();

        assert_eq!(info_a.node.op_type.to_string(), "Input");
        assert_eq!(info_b.node.op_type.to_string(), "Add");
        assert!(info_a.children.contains(&String::from("B")),);
        assert!(info_b.parents.contains(&String::from("A")),);
        assert_eq!(graph.inputs, vec!["input_global"]);
        assert_eq!(graph.outputs, vec!["output_global"]);
    }
}
