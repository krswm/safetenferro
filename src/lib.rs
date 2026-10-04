// Safetensors Loader for tenferro
// Copyright (C) 2026  Kurosawa Mutsumi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// The specification of the Safetensors file format:
// https://github.com/safetensors/safetensors#format

//! This crate loads tensors from a [Safetensors](https://github.com/safetensors/safetensors) format file as tenferro’s `Tensor`s.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use tenferro_runtime::{Tensor, TensorScalar};

trait FromLeByteSlice {
    fn from_le_byte_slice(bytes: &[u8]) -> Self;
}

impl FromLeByteSlice for f32 {
    fn from_le_byte_slice(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl FromLeByteSlice for f64 {
    fn from_le_byte_slice(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl FromLeByteSlice for i32 {
    fn from_le_byte_slice(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().unwrap())
    }
}
impl FromLeByteSlice for i64 {
    fn from_le_byte_slice(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().unwrap())
    }
}

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

fn get_tensor_permuted<T: FromLeByteSlice + TensorScalar>(
    info: Info,
    byte_buffer: &[u8],
) -> Result<Tensor, Box<dyn Error>> {
    let size = std::mem::size_of::<T>();
    let begin = info.data_offsets.0;
    let end = begin + size * info.shape.iter().product::<usize>();
    if end < info.data_offsets.1 {
        return Err("tensor data smaller than tensor shape suggests".into());
    }

    // Feed the tensor data in the file (row-major) to `from_vec_col_major`.
    // As a result, we get the *permuted* tensor.
    let colmaj: Vec<_> = byte_buffer[begin..end]
        .chunks_exact(size)
        .map(|chunk| T::from_le_byte_slice(chunk))
        .collect();

    // Revert the shape because we need a *permuted* tensor.
    let shape = info.shape.into_iter().rev().collect::<Vec<_>>();

    let tensor = Tensor::from_vec_col_major(shape, colmaj)?;
    Ok(tensor)
}

/// Load a Safetensors file from `path`.
pub fn load_safetensors_permuted<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, Tensor>, Box<dyn Error>> {
    let mut file = File::open(path)?;

    let header: HashMap<String, Value> = {
        let header_size = {
            let mut buffer = [0; 8];
            file.read_exact(&mut buffer)?;
            usize::from_le_bytes(buffer)
        };

        let mut buffer = vec![0; header_size];
        file.read_exact(&mut buffer)?;
        serde_json::from_slice(&buffer)?
    };

    let byte_buffer = {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        buffer
    };

    let mut tensors = HashMap::new();

    for (key, value) in header.into_iter() {
        if key == "__metadata__" {
            continue;
        }

        let info: Info = serde_json::from_value(value)?;

        match info.dtype.as_str() {
            "F32" => {
                let tensor = get_tensor_permuted::<f32>(info, &byte_buffer)?;
                tensors.insert(key, tensor);
            }
            "F64" => {
                let tensor = get_tensor_permuted::<f64>(info, &byte_buffer)?;
                tensors.insert(key, tensor);
            }
            "I32" => {
                let tensor = get_tensor_permuted::<i32>(info, &byte_buffer)?;
                tensors.insert(key, tensor);
            }
            "I64" => {
                let tensor = get_tensor_permuted::<i64>(info, &byte_buffer)?;
                tensors.insert(key, tensor);
            }
            _ => {}
        };
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use tenferro_runtime::DType;

    use super::*;

    #[test]
    fn test_load_safetensors_permuted() -> Result<(), Box<dyn Error>> {
        let tensors = {
            let out_dir = env!("OUT_DIR");
            let path = PathBuf::from(&out_dir).join("tensors.safetensors");
            load_safetensors_permuted(path)?
        };

        let expected_tensors = {
            let mut tensors = HashMap::new();
            tensors.insert(
                String::from("F32_tensor"),
                Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as f32).collect())?,
            );
            tensors.insert(
                String::from("F64_tensor"),
                Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as f64).collect())?,
            );
            tensors.insert(
                String::from("I32_tensor"),
                Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as i32).collect())?,
            );
            tensors.insert(
                String::from("I64_tensor"),
                Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as i64).collect())?,
            );
            tensors
        };

        assert_eq!(
            tensors.keys().collect::<HashSet<_>>(),
            expected_tensors.keys().collect::<HashSet<_>>()
        );

        for key in expected_tensors.keys() {
            let tensor = &tensors[key];
            let expected_tensor = &expected_tensors[key];

            assert_eq!(tensor.dtype(), expected_tensor.dtype());
            assert_eq!(tensor.shape(), expected_tensor.shape());

            match expected_tensor.dtype() {
                DType::F32 => {
                    assert_eq!(
                        tensor.as_slice::<f32>()?,
                        expected_tensor.as_slice::<f32>()?
                    );
                }
                DType::F64 => {
                    assert_eq!(
                        tensor.as_slice::<f64>()?,
                        expected_tensor.as_slice::<f64>()?
                    );
                }
                DType::I32 => {
                    assert_eq!(
                        tensor.as_slice::<i32>()?,
                        expected_tensor.as_slice::<i32>()?
                    );
                }
                DType::I64 => {
                    assert_eq!(
                        tensor.as_slice::<i64>()?,
                        expected_tensor.as_slice::<i64>()?
                    );
                }
                _ => {}
            }
        }

        Ok(())
    }
}
