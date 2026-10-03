// GPT-2 Inference with tenferro
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

macro_rules! impl_from_le_byte_slice {
    ($T:ty) => {
        impl FromLeByteSlice for $T {
            fn from_le_byte_slice(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}
impl_from_le_byte_slice!(f32);
impl_from_le_byte_slice!(f64);
impl_from_le_byte_slice!(i32);
impl_from_le_byte_slice!(i64);

// Now I learned generics, traits, and macros!
// Rust is so much fun!

impl PartialEq<Self> for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.dtype() == other.dtype()
        && self.shape() == other.shape()
        // && self.as_slice().unwrap() == other.as_slice().unwrap()
    }
}

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

fn get_tensor_permuted<T: FromLeByteSlice + TensorScalar>(
    byte_buffer: &[u8],
    shape: Vec<usize>,
    data_offsets: (usize, usize),
) -> Result<Tensor, Box<dyn Error>> {
    let size: usize = std::mem::size_of::<T>();

    let begin = data_offsets.0;
    let end = begin + 4 * shape.iter().product::<usize>();
    if end < data_offsets.1 {
        return Err("tensor data smaller than tensor shape suggests".into());
    }

    let colmaj: Vec<T> = byte_buffer[begin..end]
        .chunks_exact(size)
        .map(|chunk| T::from_le_byte_slice(chunk))
        .collect();

    let tensor = Tensor::from_vec_col_major(shape, colmaj)?;

    Ok(tensor)
}

pub fn load_safetensors_permuted<P: AsRef<Path>>(
    safetensors_path: P,
) -> Result<HashMap<String, Tensor>, Box<dyn Error>> {
    let mut file = File::open(safetensors_path)?;

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

        let tensor = match info.dtype.as_str() {
            "F32" => get_tensor_permuted::<f32>(&byte_buffer, info.shape, info.data_offsets)?,
            "F64" => get_tensor_permuted::<f64>(&byte_buffer, info.shape, info.data_offsets)?,
            "I32" => get_tensor_permuted::<i32>(&byte_buffer, info.shape, info.data_offsets)?,
            "I64" => get_tensor_permuted::<i64>(&byte_buffer, info.shape, info.data_offsets)?,
            _ => Tensor::from_vec_col_major(vec![], vec![0])?,
        };

        tensors.insert(key, tensor);
    }

    Ok(tensors)
}

/*
pub fn load_safetensors(
    path_to_model: &str,
) -> Result<HashMap<String, TypedTensor<f32>>, Box<dyn Error>> {
    let mut file = File::open(path_to_model)?;

    let size_of_header = {
        let mut buffer = [0; 8];
        file.read_exact(&mut buffer)?;
        usize::from_le_bytes(buffer)
    };

    let header: HashMap<String, Value> = {
        let mut buffer = vec![0; size_of_header];
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
        // I do not use metadata in my inferenece engine.
        if key == "__metadata__" {
            continue;
        }

        let name = key;
        let info: Info = serde_json::from_value(value)?;

        // My inference engine uses only f32 tensors.
        if info.dtype != "F32" {
            continue;
        }

        // f32 is 4 bytes long.
        let size = 4 * info.shape.iter().product::<usize>();
        if size < info.data_offsets.1 - info.data_offsets.0 {
            return Err("tensor data smaller than tensor shape suggests".into());
        }
        let begin = info.data_offsets.0;
        let end = begin + size;

        // ┏━━━━━┯━━━━━┯━━━━━┓
        // ┃ a₁₁ │ a₁₂ │ a₁₃ ┃
        // ┠─────┼─────┼─────┨
        // ┃ a₂₁ │ a₂₂ │ a₂₃ ┃
        // ┗━━━━━┷━━━━━┷━━━━━┛
        //
        // Safetensors uses row-major: a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃.
        // https://github.com/safetensors/safetensors#format
        //
        // tenferro uses column-major: a₁₁ a₂₁ a₁₂ a₂₂ a₁₃ a₂₃.
        // https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html#column-major-storage

        let rowmaj: Vec<f32> = byte_buffer[begin..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(*chunk.as_array::<4>().unwrap()))
            .collect();

        let (rowmaj_factors, colmaj_factors) = {
            let mut rowmaj_factors: Vec<usize> = Vec::with_capacity(info.shape.len());
            let mut colmaj_factors: Vec<usize> = Vec::with_capacity(info.shape.len());

            for i in 0..info.shape.len() {
                let rowmaj_factor = info.shape[i + 1..info.shape.len()].iter().product();
                rowmaj_factors.push(rowmaj_factor);

                let colmaj_factor = info.shape[0..i].iter().product();
                colmaj_factors.push(colmaj_factor);
            }

            (rowmaj_factors, colmaj_factors)
        };

        let colmaj = {
            let capacity: usize = info.shape.iter().product();
            let mut colmaj = Vec::with_capacity(capacity);
            for index_for_colmaj in 0..capacity {
                let mut index_for_rowmaj = 0usize;
                for (size, (colmaj_factor, rowmaj_factor)) in
                    zip(&info.shape, zip(&colmaj_factors, &rowmaj_factors))
                {
                    let index = index_for_colmaj / colmaj_factor % size;
                    index_for_rowmaj += index * rowmaj_factor;
                }
                colmaj.push(rowmaj[index_for_rowmaj]);
            }
            colmaj
        };

        let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}
*/

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_load_safetensors_permuted() -> Result<(), Box<dyn Error>> {
        let out_dir = env!("OUT_DIR");
        let path = PathBuf::from(&out_dir).join("tensors.safetensors");

        let tensors = load_safetensors_permuted(path)?;

        let mut expected = HashMap::new();
        expected.insert(
            String::from("F32_tensor"),
            Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as f32).collect())?,
        );
        expected.insert(
            String::from("F64_tensor"),
            Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as f64).collect())?,
        );
        expected.insert(
            String::from("I32_tensor"),
            Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as i32).collect())?,
        );
        expected.insert(
            String::from("I64_tensor"),
            Tensor::from_vec_col_major(vec![4, 3, 2], (0..24).map(|x| x as i64).collect())?,
        );

        assert_eq!(tensors, expected);

        Ok(())
    }
}
