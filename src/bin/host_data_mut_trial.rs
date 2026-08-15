use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

fn main() -> Result<(), Box<dyn Error>> {
    // I want to extend columns of a tensor
    // because KV-cache needs that operation.

    let mut a = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
    ])?;
    println!("{a:?}");

    a.host_data_mut()?[0] = 0.5;
    println!("{a:?}");

    // host_data is a &[f32]
    // host_data_mut is a &mut [f32]

    // a.host_data_mut()?.extend(&[5.0, 6.0]);
    // println!("{a:?}");

    // What are "strides"?
    // What is "offset"?

    Ok(())
}
