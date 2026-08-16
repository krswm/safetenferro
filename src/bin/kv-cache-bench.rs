use std::error::Error;
use std::time::Instant;

use tenferro_runtime::TypedTensor;

fn main() -> Result<(), Box<dyn Error>> {
    let dummy: &[f32] = &vec![0.0; 64][..];

    for _ in 0..10 {
        let ins = Instant::now();
        let mut vs = vec![Vec::<f32>::new(); 12];
        for p in 0..1024 {
            // Simulation of pos loop
            for v in &mut vs {
                // Simulation of layer loop
                v.extend(dummy);
                let _ = TypedTensor::<f32>::from_vec_col_major(vec![64, p + 1], v.to_vec());
            }
        }
        let time = ins.elapsed().as_nanos();
        println!("A: {time} ns");
    }

    println!("~~~~");

    for _ in 0..10 {
        let ins = Instant::now();
        let mut vs = vec![Vec::<f32>::new(); 12];
        for p in 0..1024 {
            // Simulation of pos loop
            for v in &mut vs {
                // Simulation of layer loop
                v.extend(dummy);
                let _ = TypedTensor::<f32>::from_vec_col_major(vec![64, p + 1], v.clone());
            }
        }
        let time = ins.elapsed().as_nanos();
        println!("B: {time} ns");
    }

    println!("~~~~");

    for _ in 0..10 {
        let ins = Instant::now();
        let mut vs = vec![Vec::<f32>::new(); 12];
        for p in 0..1024 {
            // Simulation of pos loop
            for v in &mut vs {
                // Simulation of layer loop
                v.extend(dummy);
                let tensor = TypedTensor::<f32>::from_vec_col_major(vec![64, p + 1], v);
            }
        }
        let time = ins.elapsed().as_nanos();
        println!("B: {time} ns");
    }

    // Hmm...

    Ok(())
}
