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

use std::collections::HashMap;
use std::error::Error;
use std::f32::consts::PI;
use std::iter::zip;

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub struct Layer {
    pub g1: TypedTensor<f32>,
    pub t1: TypedTensor<f32>,
    pub w11: TypedTensor<f32>,
    pub b11: TypedTensor<f32>,
    pub w12: TypedTensor<f32>,
    pub b12: TypedTensor<f32>,
    pub g2: TypedTensor<f32>,
    pub t2: TypedTensor<f32>,
    pub w21: TypedTensor<f32>,
    pub b21: TypedTensor<f32>,
    pub w22: TypedTensor<f32>,
    pub b22: TypedTensor<f32>,
}

pub struct Model {
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub e: TypedTensor<f32>,
    pub n_embd_as_tensor: TypedTensor<f32>,
    pub head_size_as_tensor: TypedTensor<f32>,
    pub c1: TypedTensor<f32>,
    pub c2: TypedTensor<f32>,
    pub c3: TypedTensor<f32>,
    pub c4: TypedTensor<f32>,
    pub id_embd_vecs: Vec<TypedTensor<f32>>,
    pub pos_embd_vecs: Vec<TypedTensor<f32>>,
    pub layers: Vec<Layer>,
    pub gf: TypedTensor<f32>,
    pub tf: TypedTensor<f32>,
    pub wte_transposed: TypedTensor<f32>,
}

fn validate_shape(tensor: &TypedTensor<f32>, expected: [usize; 2]) -> Result<(), Box<dyn Error>> {
    let shape = tensor.shape();
    if shape != expected {
        let message = format!("shape of tensor {shape:?} differs from expected {expected:?}");
        return Err(message.into());
    }
    Ok(())
}

pub fn get_model(
    tensors: HashMap<String, TypedTensor<f32>>,
    config: HashMap<String, Value>,
) -> Result<Model, Box<dyn Error>> {
    let n_ctx = config["n_ctx"].as_u64().unwrap() as usize;
    let n_embd = config["n_embd"].as_u64().unwrap() as usize;
    let n_head = config["n_head"].as_u64().unwrap() as usize;
    let n_layer = config["n_layer"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;

    let e = {
        let value = config["layer_norm_epsilon"].as_f64().unwrap() as f32;
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
    };

    let n_embd_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![], vec![n_embd as f32])?;

    let head_size_as_tensor = TypedTensor::<f32>::from_vec_col_major(
        vec![],
        vec![1.0 / ((n_embd as f32) / (n_head as f32)).sqrt()],
    )?;

    let c1 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.044715])?;

    let c2 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![(2.0 / PI).sqrt()])?;

    let c3 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0])?;

    let c4 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.5])?;

    let mut backend = CpuBackend::new();

    // It feels more natural for me
    // to perform "matrix * vector -> vector"
    // than to perform "row vector * matrix -> row vector."
    // Therefore, I apply `transpose` to the all matrices.

    // This may also simplify the KV cache code
    // since tenferro is col-major
    // so adding a new column vector to the cache matrices
    // is just appending the column vector to its internal Vector.

    // wte and wpe are used just to obtain vectors at the input embedding phase
    // (wte is also used for output embedding though, it is the transposed one)
    // And my previous implementation in tenferro/Rust construct vector every time!
    // So I'll obtain vectors beforehand here!
    let id_embd_vecs = {
        let mut embd_vecs = Vec::with_capacity(vocab_size);
        for row in 0..vocab_size {
            let mut colmaj = Vec::with_capacity(n_embd);
            for col in 0..n_embd {
                colmaj.push(*tensors["wte.weight"].get(&[row, col])?);
            }
            let embd_vec = TypedTensor::<f32>::from_vec_col_major(vec![n_embd, 1], colmaj)?;
            embd_vecs.push(embd_vec);
        }
        embd_vecs
    };

    let pos_embd_vecs = {
        let mut embd_vecs = Vec::with_capacity(n_ctx);
        for row in 0..n_ctx {
            let mut colmaj = Vec::with_capacity(n_embd);
            for col in 0..n_embd {
                colmaj.push(*tensors["wpe.weight"].get(&[row, col])?);
            }
            let embd_vec = TypedTensor::<f32>::from_vec_col_major(vec![n_embd, 1], colmaj)?;
            embd_vecs.push(embd_vec);
        }
        embd_vecs
    };

    let layers = {
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let g1 = tensors[&format!("h.{i}.ln_1.weight")].reshape(&[n_embd, 1], &mut backend)?;

            let t1 = tensors[&format!("h.{i}.ln_1.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let w11 =
                tensors[&format!("h.{i}.attn.c_attn.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w11, [n_embd * 3, n_embd])?;

            let b11 = tensors[&format!("h.{i}.attn.c_attn.bias")]
                .reshape(&[n_embd * 3, 1], &mut backend)?;

            let w12 =
                tensors[&format!("h.{i}.attn.c_proj.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w12, [n_embd, n_embd])?;

            let b12 =
                tensors[&format!("h.{i}.attn.c_proj.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let g2 = tensors[&format!("h.{i}.ln_2.weight")].reshape(&[n_embd, 1], &mut backend)?;

            let t2 = tensors[&format!("h.{i}.ln_2.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let w21 =
                tensors[&format!("h.{i}.mlp.c_fc.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w21, [n_embd * 4, n_embd])?;

            let b21 =
                tensors[&format!("h.{i}.mlp.c_fc.bias")].reshape(&[n_embd * 4, 1], &mut backend)?;

            let w22 =
                tensors[&format!("h.{i}.mlp.c_proj.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w22, [n_embd, n_embd * 4])?;

            let b22 =
                tensors[&format!("h.{i}.mlp.c_proj.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let layer = Layer {
                g1,
                t1,
                w11,
                b11,
                w12,
                b12,
                g2,
                t2,
                w21,
                b21,
                w22,
                b22,
            };
            layers.push(layer);
        }
        layers
    };

    // If I remember correctly
    // tenferro raises RankMismatch
    // when I try (TypedTensor with shape [a, b]) matmul (TypedTensor with shape [a]).
    // If I need a matrix * vector operation
    // I have to use (TypedTensor with shape [a, b]) matmul (TypedTensor with shape [a, 1]) instead.
    // That means, I have to reshape the vectors to have the second index whose size is 1.
    let gf = tensors["ln_f.weight"].reshape(&[n_embd, 1], &mut backend)?;

    let tf = tensors["ln_f.bias"].reshape(&[n_embd, 1], &mut backend)?;

    let wte_transposed = tensors["wte.weight"].clone();

    let model = Model {
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        vocab_size,
        e,
        n_embd_as_tensor,
        head_size_as_tensor,
        c1,
        c2,
        c3,
        c4,
        id_embd_vecs,
        pos_embd_vecs,
        layers,
        gf,
        tf,
        wte_transposed,
    };
    Ok(model)
}

/// The transformer for the GPT-2 architecture.
pub fn transform(
    cached_k: &mut [Vec<f32>],
    cached_v: &mut [Vec<f32>],
    model: &Model,
    id: usize,
    pos: usize,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // ==== Embedding ====

    let mut x = model.id_embd_vecs[id].add(&model.pos_embd_vecs[pos], backend)?;

    for (i, layer) in zip(0..model.n_layer, &model.layers) {
        // ==== Masked Multi-Head Attention ====

        let y = layer_norm(&x, &layer.g1, &layer.t1, model, backend)?;

        let y = layer.w11.matmul(&y, backend)?.add(&layer.b11, backend)?;

        //   /\|
        //   \/|
        //     | ih

        let host_data = y.host_data()?;
        let mut j = 0;
        let q = TypedTensor::<f32>::from_vec_col_major(
            vec![model.n_embd / model.n_head, model.n_head],
            host_data[j..(j + model.n_embd)].to_vec(),
        )?;
        j += model.n_embd;
        cached_k[i].extend_from_slice(&host_data[j..(j + model.n_embd)]);
        let k = TypedTensor::<f32>::from_vec_col_major(
            vec![model.n_embd / model.n_head, model.n_head, pos + 1],
            cached_k[i].clone(),
        )?;
        j += model.n_embd;
        cached_v[i].extend_from_slice(&host_data[j..(j + model.n_embd)]);
        let v = TypedTensor::<f32>::from_vec_col_major(
            vec![model.n_embd / model.n_head, model.n_head, pos + 1],
            cached_v[i].clone(),
        )?;

        // z_c = [k^T q]_{c}
        //     = [k^T]_{ci} q_{i}
        //     = k_{ic} q_{i}

        // z_{hc} = [k^T q]_{hc}
        //        = [k^T]_{chi} q_{ih}   <- (?)
        //        = k_{ihc} q_{ih}       <- (?)

        let z = [&k, &q].einsum("ihc,ih->ch", backend)?;

        // _{ch}
        let z = z.mul(&model.head_size_as_tensor, backend)?;

        // _{h}
        let max_z = {
            let mut colmaj = Vec::with_capacity(model.n_head);
            for h in 0..model.n_head {
                let max: f32 = *z.host_data()?[(h * (pos + 1))..((h + 1) * (pos + 1))]
                    .iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();
                colmaj.push(max);
            }
            TypedTensor::<f32>::from_vec_col_major(vec![1, model.n_head], colmaj)?
        };

        // _{ch}
        let z = z.sub(&max_z, backend)?;

        // _{ch}
        let z = z.exp(backend)?;

        // _{h}
        let x12 = z
            .reduce_sum(&[0], backend)?
            .reshape(&[1, model.n_head], backend)?;

        // _{ch}
        let x13 = z.div(&x12, backend)?;

        let z = [&v, &x13].einsum("ihc,ch->ih", backend)?;

        let y = z.reshape(&[model.n_embd, 1], backend)?;

        let y = layer.w12.matmul(&y, backend)?.add(&layer.b12, backend)?;

        x = x.add(&y, backend)?;

        // ==== Feed Forward ====

        let y = layer_norm(&x, &layer.g2, &layer.t2, model, backend)?;

        let y = layer.w21.matmul(&y, backend)?.add(&layer.b21, backend)?;

        let y = y
            .mul(&y, backend)?
            .mul(&y, backend)?
            .mul(&model.c1, backend)?
            .add(&y, backend)?
            .mul(&model.c2, backend)?
            .tanh(backend)?
            .add(&model.c3, backend)?
            .mul(&y, backend)?
            .mul(&model.c4, backend)?;

        let y = layer.w22.matmul(&y, backend)?.add(&layer.b22, backend)?;

        x = x.add(&y, backend)?;
    }

    x = layer_norm(&x, &model.gf, &model.tf, model, backend)?;

    x = model.wte_transposed.matmul(&x, backend)?;

    Ok(x)
}

fn layer_norm(
    x: &TypedTensor<f32>,
    g: &TypedTensor<f32>,
    t: &TypedTensor<f32>,
    model: &Model,
    backend: &mut tenferro_cpu::CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // mean(x) = sum(x) / n_embd
    let mean_x = x
        .reduce_sum(&[0], backend)?
        .div(&model.n_embd_as_tensor, backend)?;

    // diff(x) = x .- mean(x)
    let diff_x = x.sub(&mean_x, backend)?;

    // var(x, corrected = false) = sum(diff(x) .^ 2) / n_embd
    let var_x = diff_x
        .mul(&diff_x, backend)?
        .reduce_sum(&[0], backend)?
        .div(&model.n_embd_as_tensor, backend)?;

    // √(var(x, corrected = false) + e)
    let denom = var_x.add(&model.e, backend)?.sqrt(backend)?;

    // (x .- mean(x)) ./ √(var(x, corrected = false) + e) .* g + t
    let x = diff_x
        .div(&denom, backend)?
        .mul(g, backend)?
        .add(t, backend)?;

    Ok(x)
}

/// Pretty-print a 2D tensor for debug.
#[allow(dead_code)]
fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    if tensor.shape().len() != 2 {
        return Err("not 2D tensor".into());
    }

    let num_rows = tensor.shape()[0];
    let num_cols = tensor.shape()[1];

    if num_rows == 0 {
        return Err("num_rows is 0".into());
    }
    if num_cols == 0 {
        return Err("num_cols is 0".into());
    }

    println!("┌{:─^29}┐", "");

    println!(
        "│ {:<+12.6e} ⋯ {:<+12.6e} │",
        tensor.get(&[0, 0]).unwrap(),
        tensor.get(&[0, num_cols - 1]).unwrap(),
    );

    println!("│ {:^12}   {:^12} {num_rows}", "⋮", "⋮");

    println!(
        "│ {:<+12.6e} ⋯ {:<+12.6e} │",
        tensor.get(&[num_rows - 1, 0]).unwrap(),
        tensor.get(&[num_rows - 1, num_cols - 1]).unwrap(),
    );

    println!("└{num_cols:─^29}┘");

    Ok(())
}
