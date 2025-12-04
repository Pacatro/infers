use std::fmt;
use std::str::FromStr;

use crate::InfersResult;

/// The type of an operation in the graph.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OpType {
    /// Input to the graph.
    Input,
    /// Output of the graph.
    Output,
    /// Add two tensors.
    Add,
    /// General Matrix multiplication.
    Gemm,
    /// Flatten a tensor.
    Flatten,
    /// Rectified Linear Unit.
    Relu,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OpType::Input => "Input",
            OpType::Output => "Output",
            OpType::Add => "Add",
            OpType::Gemm => "Gemm",
            OpType::Flatten => "Reshape",
            OpType::Relu => "Relu",
        };
        write!(f, "{s}")
    }
}

impl FromStr for OpType {
    type Err = Box<dyn std::error::Error>;

    fn from_str(value: &str) -> InfersResult<OpType> {
        match value {
            "Input" => Ok(OpType::Input),
            "Output" => Ok(OpType::Output),
            "Add" => Ok(OpType::Add),
            "Gemm" => Ok(OpType::Gemm),
            // ONNX uses "Reshape" for flatten:
            "Reshape" | "Flatten" => Ok(OpType::Flatten),
            "Relu" => Ok(OpType::Relu),
            other => Err(format!("Unknown op type: {}", other).into()),
        }
    }
}

impl TryFrom<&str> for OpType {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: &str) -> InfersResult<OpType> {
        OpType::from_str(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_from_str_ok() {
        assert_eq!(OpType::from_str("Add").unwrap(), OpType::Add);
        assert_eq!(OpType::from_str("Reshape").unwrap(), OpType::Flatten);
    }

    #[test]
    fn test_from_str_err() {
        assert!(OpType::from_str("UnknownOp").is_err());
    }

    #[test]
    fn test_to_string() {
        assert_eq!(OpType::Add.to_string(), "Add");
        assert_eq!(OpType::Flatten.to_string(), "Reshape");
    }

    #[test]
    fn test_try_from() {
        let op: OpType = "Gemm".try_into().unwrap();
        assert_eq!(op, OpType::Gemm);
    }
}
