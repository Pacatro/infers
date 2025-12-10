use std::fmt::{Debug, Display};

use num_traits::{FromPrimitive, Num};

use crate::{Tensor, backends::Backend};

const MAX_TENSOR_DISPLAY: usize = 1000;

/// Displays a tensor in a human-readable format.
///
/// # Arguments
///
/// * `f` - The formatter to use for display.
/// * `data` - The tensor data to display.
/// * `shape` - The tensor shape.
/// * `strides` - The tensor strides.
/// * `dim` - The dimension to display.
/// * `offset` - The offset to start displaying from.
fn fmt_tensor_data<T>(
    f: &mut std::fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
    offset: usize,
) -> std::fmt::Result
where
    T: Num + Debug + Clone + Copy + FromPrimitive + Display,
{
    let indent_level = dim * 2 + 2;
    let indent_str = " ".repeat(indent_level);

    // Last dimension
    if dim == shape.len() - 1 {
        write!(f, "[")?;
        for i in 0..shape[dim] {
            let flat_idx = offset + i * strides[dim];
            write!(f, "{:.4}", data[flat_idx])?;
            if i < shape[dim] - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;
        return Ok(());
    }

    write!(f, "[")?;
    for i in 0..shape[dim] {
        let new_offset = offset + i * strides[dim];

        if i > 0 {
            write!(f, ",\n{}", indent_str)?;
        }

        fmt_tensor_data(f, data, shape, strides, dim + 1, new_offset)?;
    }

    write!(f, "]")
}
impl<B, T> Display for Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Debug + Clone + Copy + FromPrimitive + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.len == 0 {
            return write!(
                f,
                "Tensor([], shape: {:?}, device: {}, dtype: {})",
                self.shape,
                B::device(),
                std::any::type_name::<T>()
            );
        }

        if self.len > MAX_TENSOR_DISPLAY {
            return write!(
                f,
                "Tensor([...], shape: {:?}, device: {}, dtype: {})",
                self.shape,
                B::device(),
                std::any::type_name::<T>()
            );
        }

        // Copy data from device to host
        let data = match B::copy_to_host(&self.storage.borrow()) {
            Ok(data) => data,
            Err(e) => return write!(f, "{:?}", e),
        };

        writeln!(f, "Tensor(")?;
        write!(f, "  ")?;

        fmt_tensor_data(f, &data, &self.shape, &self.strides, 0, 0)?;

        write!(
            f,
            ",\n  shape: {:?},\n  device: {},\n  dtype: {}\n)",
            self.shape,
            B::device(),
            std::any::type_name::<T>()
        )
    }
}
