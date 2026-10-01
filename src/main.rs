/*
use std::error::Error;

use tenferro_runtime::{TensorScalar, TypedTensor};

trait FromVecColMajor2: Sized {
    fn from_vec_col_major_2<T>(shape: Vec<usize>, data: Vec<T>) -> Result<Self, Box<dyn Error>>
    where
        T: TensorScalar;
}

impl<T> FromVecColMajor2 for TypedTensor<T> {
    fn from_vec_col_major_2<T>(shape: Vec<usize>, data: Vec<T>) -> Result<Self, Box<dyn Error>> {
        self::from_vec_col_major(shape, data)
    }
}

/*
fn foo<T: TensorScalar>() -> Result<T, Box<dyn Error>> {
    let tensor = T::from_vec_col_major(vec![2, 2], vec![0.0, 1.0, 2.0, 3.0])?;
    Ok(tensor)
}
*/

fn main() -> Result<(), Box<dyn Error>> {
    /*
    let tensor = TypedTensor::<f32>::from_vec_row_major(vec![2, 2], vec![0.0, 1.0, 2.0, 3.0])?;
    println!("{:?}", tensor.host_data());
    */

    Ok(())
}
*/

// I just now noticed that tenferro has `from_vec_row_major`!
// From when has it been there???
//
// - On tensor4all.org's docs' search, it HITS.
//   https://tensor4all.org/tenferro-rs/api/tenferro_runtime/?search=from_vec_row_major
// - On docs.rs's search, NO HIT.
//   https://docs.rs/tenferro-runtime/0.7.1/tenferro_runtime/index.html?search=from_vec_row_major
//
// I noticed it on tensor4all.org's docs, maybe it's tenferro's nightly feature?
// Today is 2026-10-01 by the way.
//
// I CAN'T use `from_vec_row_major` just now. (Cargo.toml has tenferro_runtime = "0.7").
// It's indeed a nightly.
// I'm really looking forward for it to be released,
// since it makes my Safetensors loading implementation easier!
//
// This commit introduced `from_vec_row_major`.
// It's just three days ago!
// https://github.com/tensor4all/tenferro-rs/commit/689f5f28fbe5bcf172f119be8438f92c74fc2a3e
// tenferro is actively developed!
