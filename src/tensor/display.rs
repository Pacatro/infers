use std::fmt::{Debug, Display};

use num_traits::{FromPrimitive, Num};

use crate::backends::Backend;

use super::{Tensor, base::compute_strides};

const MAX_TENSOR_DISPLAY: usize = 1000;

fn fmt_tensor_data<T>(
    f: &mut std::fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    offset: usize,
) -> std::fmt::Result
where
    T: Display,
{
    if shape.is_empty() {
        return write!(f, "{:.4}", data[0]);
    }

    if axis == shape.len() - 1 {
        write!(f, "[")?;
        for index in 0..shape[axis] {
            write!(f, "{:.4}", data[offset + index * strides[axis]])?;
            if index + 1 < shape[axis] {
                write!(f, ", ")?;
            }
        }
        return write!(f, "]");
    }

    write!(f, "[")?;
    for index in 0..shape[axis] {
        if index > 0 {
            write!(f, ",\n{}", " ".repeat(axis * 2 + 2))?;
        }
        fmt_tensor_data(
            f,
            data,
            shape,
            strides,
            axis + 1,
            offset + index * strides[axis],
        )?;
    }
    write!(f, "]")
}

impl<B, T> Display for Tensor<B, T>
where
    B: Backend<T>,
    T: Num + Debug + Clone + Copy + FromPrimitive + Display + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(
                f,
                "Tensor([], shape: {:?}, device: {}, dtype: {})",
                self.dims(),
                self.device(),
                std::any::type_name::<T>()
            );
        }

        if self.len() > MAX_TENSOR_DISPLAY {
            return write!(
                f,
                "Tensor([...], shape: {:?}, device: {}, dtype: {})",
                self.dims(),
                self.device(),
                std::any::type_name::<T>()
            );
        }

        let data = match self.data() {
            Ok(data) => data,
            Err(error) => return write!(f, "{error}"),
        };
        let display_strides = compute_strides(self.dims());

        writeln!(f, "Tensor(")?;
        write!(f, "  ")?;
        fmt_tensor_data(f, &data, self.dims(), &display_strides, 0, 0)?;
        write!(
            f,
            ",\n  shape: {:?},\n  device: {},\n  dtype: {}\n)",
            self.dims(),
            self.device(),
            std::any::type_name::<T>()
        )
    }
}
