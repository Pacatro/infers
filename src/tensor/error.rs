use std::fmt;

/// Errors produced while constructing or manipulating tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    DataLengthMismatch {
        expected: usize,
        actual: usize,
        shape: Vec<usize>,
    },
    ShapeOverflow {
        shape: Vec<usize>,
    },
    ShapeStrideRankMismatch {
        shape_rank: usize,
        strides_rank: usize,
    },
    InvalidAxis {
        axis: usize,
        rank: usize,
    },
    InvalidIndex {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },
    InvalidReshape {
        from: Vec<usize>,
        to: Vec<usize>,
    },
    IncompatibleShapes {
        operation: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    NonContiguousReshape,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataLengthMismatch {
                expected,
                actual,
                shape,
            } => write!(
                f,
                "data length {actual} does not match shape {shape:?} ({expected} elements)"
            ),
            Self::ShapeOverflow { shape } => {
                write!(f, "shape {shape:?} exceeds the addressable tensor size")
            }
            Self::ShapeStrideRankMismatch {
                shape_rank,
                strides_rank,
            } => write!(
                f,
                "shape rank {shape_rank} does not match strides rank {strides_rank}"
            ),
            Self::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is invalid for a tensor of rank {rank}")
            }
            Self::InvalidIndex { indices, shape } => {
                write!(f, "index {indices:?} is outside tensor shape {shape:?}")
            }
            Self::InvalidReshape { from, to } => {
                write!(f, "cannot reshape tensor from {from:?} to {to:?}")
            }
            Self::IncompatibleShapes {
                operation,
                lhs,
                rhs,
            } => write!(
                f,
                "incompatible shapes for {operation}: {lhs:?} and {rhs:?}"
            ),
            Self::NonContiguousReshape => {
                write!(f, "cannot reshape a non-contiguous tensor without copying")
            }
        }
    }
}

impl std::error::Error for TensorError {}
