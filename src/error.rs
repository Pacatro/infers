use std::fmt;

#[derive(Debug)]
pub enum InfersError {
    Io(std::io::Error),
    OnnxDecode(prost::DecodeError),
    OnnxFormat(String),
    Operation(String),
    Tensor(String),
    Backend(String),
    Cuda(String),
    Shape(String),
    Type(String),
    Device(String),
    Memory(String),
    Parse(String),
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

pub type InfersResult<T> = Result<T, InfersError>;
