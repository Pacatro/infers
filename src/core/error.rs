use std::fmt;

/// Result type alias for convenience throughout the Infers library
pub type InfersResult<T> = Result<T, InfersError>;

/// Error type for the Infers inference engine.
#[derive(Debug)]
pub enum InfersError {
    /// I/O related errors (file reading, writing, etc.)
    Io(std::io::Error),

    /// Errors that occur during ONNX protobuf decoding
    OnnxDecode(prost::DecodeError),

    /// Errors related to invalid ONNX model format or structure
    OnnxFormat(String),

    /// Errors that occur during operation execution (invalid inputs, unsupported ops, etc.)
    Operation(String),

    /// Tensor-related errors (invalid dimensions, empty tensors, etc.)
    Tensor(String),

    /// Backend-specific errors (CPU/CUDA backend failures)
    Backend(String),

    /// CUDA-specific errors (driver errors, compilation failures, etc.)
    Cuda(String),

    /// Shape-related errors (dimension mismatches, invalid shapes, etc.)
    Shape(String),

    /// Type-related errors (unsupported data types, type mismatches, etc.)
    Type(String),

    /// Device-related errors (device not available, invalid device selection, etc.)
    Device(String),

    /// Memory-related errors (allocation failures, out of memory, etc.)
    Memory(String),

    /// Parsing errors (string to enum conversion, attribute parsing, etc.)
    Parse(String),

    /// Validation errors (input validation, parameter validation, etc.)
    Validation(String),
}

impl fmt::Display for InfersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfersError::Io(err) => write!(f, "IO error: {}", err),
            InfersError::OnnxDecode(err) => write!(f, "ONNX decode error: {}", err),
            InfersError::OnnxFormat(msg) => write!(f, "Invalid ONNX format: {}", msg),
            InfersError::Operation(msg) => write!(f, "Operation error: {}", msg),
            InfersError::Tensor(msg) => write!(f, "Tensor error: {}", msg),
            InfersError::Backend(msg) => write!(f, "Backend error: {}", msg),
            InfersError::Cuda(msg) => write!(f, "CUDA error: {}", msg),
            InfersError::Shape(msg) => write!(f, "Shape error: {}", msg),
            InfersError::Type(msg) => write!(f, "Type error: {}", msg),
            InfersError::Device(msg) => write!(f, "Device error: {}", msg),
            InfersError::Memory(msg) => write!(f, "Memory error: {}", msg),
            InfersError::Parse(msg) => write!(f, "Parse error: {}", msg),
            InfersError::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for InfersError {}

impl From<crate::tensor::TensorError> for InfersError {
    fn from(err: crate::tensor::TensorError) -> Self {
        Self::Tensor(err.to_string())
    }
}

impl From<std::io::Error> for InfersError {
    fn from(err: std::io::Error) -> Self {
        InfersError::Io(err)
    }
}

impl From<prost::DecodeError> for InfersError {
    fn from(err: prost::DecodeError) -> Self {
        InfersError::OnnxDecode(err)
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for InfersError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        InfersError::Cuda(err.to_string())
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::nvrtc::CompileError> for InfersError {
    fn from(err: cudarc::nvrtc::CompileError) -> Self {
        InfersError::Cuda(format!("NVRTC compilation error: {}", err))
    }
}

impl From<&str> for InfersError {
    fn from(err: &str) -> Self {
        InfersError::Validation(err.to_string())
    }
}

impl From<String> for InfersError {
    fn from(err: String) -> Self {
        InfersError::Validation(err)
    }
}

impl From<Box<dyn std::error::Error>> for InfersError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        InfersError::Operation(err.to_string())
    }
}

impl From<std::array::TryFromSliceError> for InfersError {
    fn from(err: std::array::TryFromSliceError) -> Self {
        InfersError::Parse(format!("Slice conversion error: {}", err))
    }
}
