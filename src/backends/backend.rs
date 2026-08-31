use std::fmt::{Debug, Display};

use crate::{
    core::InfersResult,
    tensor::{Layout, Shape},
};

#[derive(Debug, Clone)]
pub struct GemmParams<'a, T, S> {
    pub lhs: &'a S,
    pub lhs_layout: &'a Layout,
    pub rhs: &'a S,
    pub rhs_layout: &'a Layout,
    pub alpha: T,
    pub beta: T,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

/// Storage and computation primitives for a physical device.
pub trait Backend<T = f32>: Clone + Debug + Copy {
    type Storage: Clone + Debug;

    fn device() -> Device;

    fn from_host(data: Vec<T>) -> InfersResult<Self::Storage>;

    fn read(storage: &Self::Storage, index: usize) -> InfersResult<T>;

    /// Materializes the logical tensor represented by `layout` on the host.
    fn to_host(storage: &Self::Storage, layout: &Layout) -> InfersResult<Vec<T>>;

    fn contiguous(storage: &Self::Storage, layout: &Layout) -> InfersResult<Self::Storage> {
        Self::from_host(Self::to_host(storage, layout)?)
    }

    fn add(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage>;

    fn sub(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage>;

    fn mul(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
        output_shape: &Shape,
    ) -> InfersResult<Self::Storage>;

    fn relu(input: &Self::Storage, layout: &Layout) -> InfersResult<Self::Storage>;

    fn gemm(params: GemmParams<T, Self::Storage>) -> InfersResult<Self::Storage>;

    fn dot(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> InfersResult<Self::Storage>;
}
