use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

fn main() -> Result<(), Box<dyn Error>> {
    let mut be = CpuBackend::new();
    
    let a = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
    ]).unwrap();
    let b = [&a].einsum("ij->ij", &mut be)?;
    println!("{b:?}");
    let b = [&a].einsum("ij->ji", &mut be)?;
    println!("{b:?}");
    let b = [&a].einsum("ji->ji", &mut be)?;
    println!("{b:?}");

    println!("----");

    let a = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
    ]).unwrap();
    let b = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![
        5.0, 6.0,
        7.0, 8.0,
    ]).unwrap();
    let c = a.matmul(&b, &mut be)?;
    println!("{c:?}");
    let c = [&a, &b].einsum("ij,jk->ik", &mut be)?;
    println!("{c:?}");
    let c = [&a, &b].einsum("ki,ij->kj", &mut be)?;
    println!("{c:?}");
    let c = a.transpose(&[1, 0], &mut be)?.matmul(&b, &mut be)?;
    println!("{c:?}");
    let c = [&a, &b].einsum("ji,jk->ik", &mut be)?;
    println!("{c:?}");

    println!("----");
    let a = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![
        1.0, 2.0,
        3.0, 4.0,
    ]).unwrap();
    let b = [&a].einsum("ij->i", &mut be)?;
    println!("{b:?}");
 
    Ok(())
}
