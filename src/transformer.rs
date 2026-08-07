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
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

/*
pub struct Config {
    pub layer_norm_epsilon: f32,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
}
*/

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

    let head_size_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![], vec![(n_embd / n_head) as f32])?;

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

    // &[a, b] means axis 0 becomes axis a and axis 1 becomes axis b maybe.
    // So &[1, 0] is just a normal matrix transposition.
    /*
    let wte = tensors["wte.weight"].transpose(&[1, 0], &mut backend)?;
    validate_shape(&wte, [n_embd, vocab_size])?;

    let wpe = tensors["wpe.weight"].transpose(&[1, 0], &mut backend)?;
    validate_shape(&wpe, [n_embd, n_ctx])?;
    */

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
            // Btw, g stands for gamma and t stands for beta (b and e are already taken!)
            // Pytorch etc. uses these greek letters
            // The original paper of layernorm uses latin g and b maybe gain and bias
            // The greek letters corresponds to the latin letters g and b.
            
            let g1 = tensors[&format!("h.{i}.ln_1.weight")].reshape(&[n_embd, 1], &mut backend)?;

            let t1 = tensors[&format!("h.{i}.ln_1.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let w11 = tensors[&format!("h.{i}.attn.c_attn.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w11, [n_embd * 3, n_embd])?;

            let b11 = tensors[&format!("h.{i}.attn.c_attn.bias")].reshape(&[n_embd * 3, 1], &mut backend)?;

            let w12 = tensors[&format!("h.{i}.attn.c_proj.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w12, [n_embd, n_embd])?;

            let b12 = tensors[&format!("h.{i}.attn.c_proj.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let g2 = tensors[&format!("h.{i}.ln_2.weight")].reshape(&[n_embd, 1], &mut backend)?;

            let t2 = tensors[&format!("h.{i}.ln_2.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let w21 = tensors[&format!("h.{i}.mlp.c_fc.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w21, [n_embd * 4, n_embd])?;

            let b21 = tensors[&format!("h.{i}.mlp.c_fc.bias")].reshape(&[n_embd * 4, 1], &mut backend)?;

            let w22 = tensors[&format!("h.{i}.mlp.c_proj.weight")].transpose(&[1, 0], &mut backend)?;
            validate_shape(&w22, [n_embd, n_embd * 4])?;

            let b22 = tensors[&format!("h.{i}.mlp.c_proj.bias")].reshape(&[n_embd, 1], &mut backend)?;

            let layer = Layer { g1, t1, w11, b11, w12, b12, g2, t2, w21, b21, w22, b22 };
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

    // There was a typo!!
    let tf = tensors["ln_f.bias"].reshape(&[n_embd, 1], &mut backend)?;
    // Now fixed!

    // Matrix transposed transposed is just original matrix!
    let wte_transposed = tensors["wte.weight"].clone();

    // Rust's field init shorthand is elegant!
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
    cached_k: &mut Vec<Vec<Vec<f32>>>,
    cached_v: &mut Vec<Vec<Vec<f32>>>,
    model: &Model,
    id: usize,
    pos: usize,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // ==== Embedding ====
    println!("[I]");

    let mut x = model.id_embd_vecs[id].add(&model.pos_embd_vecs[pos], backend)?;

    for (i, (layer, (k_matrices, v_matrices))) in zip(&model.layers, zip(cached_k, cached_v)).enumerate() {
        println!("[J]");
        // ==== Masked Multi-Head Attention ====

        println!("[M]");
        let y = layer_norm(&x, &layer.g1, &layer.t1, model, backend)?;

        println!("[N]");
        // Rust's variable shadowing is nice!
        let y = layer.w11.matmul(&y, backend)?.add(&layer.b11, backend)?;

        println!("[O]");
        let host_data = y.host_data()?;
        let mut j = 0;
        let q_vectors = {
            let mut vectors = Vec::with_capacity(model.n_head);
            for i_head in 0..model.n_head {
                vectors.push(&host_data[j..(j + model.n_embd / model.n_head)]);
                j += model.n_embd / model.n_head;
            }
            vectors
        };
        let k_vectors = {
            let mut vectors = Vec::with_capacity(model.n_head);
            for i_head in 0..model.n_head {
                vectors.push(&host_data[j..(j + model.n_embd / model.n_head)]);
                j += model.n_embd / model.n_head;
            }
            vectors
        };
        let v_vectors = {
            let mut vectors = Vec::with_capacity(model.n_head);
            for i_head in 0..model.n_head {
                vectors.push(&host_data[j..(j + model.n_embd / model.n_head)]);
                j += model.n_embd / model.n_head;
            }
            vectors
        };

        println!("[P]");
        let qs = {
            let mut qs = Vec::with_capacity(model.n_head);
            for q_vector in q_vectors {
                let q = TypedTensor::<f32>::from_vec_col_major(vec![model.n_embd / model.n_head, 1], q_vector.to_vec())?;
                qs.push(q);
            }
            qs
        };
        println!("[a]");

        // Utilize the fact that tenferro is col major
        // so extending a new column is just appending to col major.
        for (k_matrix, k_vector) in zip(&mut *k_matrices, k_vectors) {
            k_matrix.extend_from_slice(k_vector);
        }
        println!("[b]");
        let ks = {
            let mut ks = Vec::with_capacity(model.n_head);
            for k_matrix in k_matrices {
                let k = TypedTensor::<f32>::from_vec_col_major(vec![model.n_embd / model.n_head, pos + 1], k_matrix.to_vec())?;
                ks.push(k);
            }
            ks
        };

        println!("[c]");
        for (v_matrix, v_vector) in zip(&mut *v_matrices, v_vectors) {
            v_matrix.extend_from_slice(v_vector);
        }
        println!("[d]");
        let vs = {
            let mut vs = Vec::with_capacity(model.n_head);
            for v_matrix in v_matrices {
                let v = TypedTensor::<f32>::from_vec_col_major(vec![model.n_embd / model.n_head, pos + 1], v_matrix.to_vec())?;
                vs.push(v);
            }
            vs
        };

        println!("[Q]");
        let y = {
            let mut attention = Vec::with_capacity(model.n_embd);
            for (q, (k, v)) in zip(qs, zip(ks, vs)) {
                let z = k
                    .transpose(&[1, 0], backend)?
                    .matmul(&q, backend)?
                    .div(&model.head_size_as_tensor, backend)?;
                let mut z_max = -1.0e12;
                for data in z.host_data()?.into_iter() {
                    if *data > z_max {
                        z_max = *data;
                    }
                }
                let z_max = TypedTensor::<f32>::from_vec_col_major(vec![], vec![z_max])?;
                let z = z.sub(&z_max, backend)?.exp(backend)?;
                let mut z = v.matmul(&z, backend)?;
                attention.extend(&*z.host_data_mut()?);
            }
            TypedTensor::<f32>::from_vec_col_major(vec![model.n_embd, 1], attention)?
        };


        println!("[R]");
        let y = layer.w12.matmul(&y, backend)?.add(&layer.b12, backend)?;

        println!("[S]");
        x = x.add(&y, backend)?;

        println!("[K]");
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

    println!("[L]");

    x = layer_norm(&x, &model.gf, &model.tf, model, backend)?;

    x = model.wte_transposed.matmul(&x, backend)?;


    Ok(x)
}


/*
/// The transformer for the GPT-2 architecture.
pub fn transform(
    tensors: &HashMap<String, TypedTensor<f32>>,
    config: &Config,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // ==== Embedding ====

    let wte_weight = &tensors["wte.weight"];
    if wte_weight.shape() != [config.vocab_size, config.n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    // wte_weight[ids]
    let x0 = {
        let mut colmaj = Vec::with_capacity(ids.len() * config.n_embd);
        for col in 0..config.n_embd {
            for row in ids {
                colmaj.push(*wte_weight.get(&[*row, col])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), config.n_embd], colmaj)?
    };

    let wpe_weight = &tensors["wpe.weight"];
    if wpe_weight.shape() != [config.n_ctx, config.n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    // wpe_weight[0:length(ids)]
    let x1 = {
        let mut colmaj = Vec::with_capacity(ids.len() * config.n_embd);
        for col in 0..config.n_embd {
            for row in 0..ids.len() {
                colmaj.push(*wpe_weight.get(&[row, col])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), config.n_embd], colmaj)?
    };

    // wte_weight[ids] + wpe_weight[0:length(ids)]
    let mut x2 = x0.add(&x1, backend)?;

    for i_layer in 0..config.n_layer {
        // ==== Masked Multi-Head Attention ====

        let ln_1_weight = &tensors[&format!("h.{i_layer}.ln_1.weight")];
        if ln_1_weight.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let ln_1_bias = &tensors[&format!("h.{i_layer}.ln_1.bias")];
        if ln_1_bias.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let x3 = layer_norm(&x2, ln_1_weight, ln_1_bias, config, backend)?;

        let attn_c_attn_weight = &tensors[&format!("h.{i_layer}.attn.c_attn.weight")];
        if attn_c_attn_weight.shape() != [config.n_embd, 3 * config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let attn_c_attn_bias = &tensors[&format!("h.{i_layer}.attn.c_attn.bias")];
        if attn_c_attn_bias.shape() != [3 * config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x3 * attn_c_attn_weight .+ attn_c_attn_bias
        let x4 = x3
            .matmul(attn_c_attn_weight, backend)?
            .add(attn_c_attn_bias, backend)?;

        let x15 = {
            let size_of_head = config.n_embd / config.n_head;
            let mut stacked_colmaj: Vec<f32> = Vec::with_capacity(ids.len() * config.n_embd);

            //                config.n_head sub-matrices
            //      ┌─────────┴─────────┐
            //      ┏━━━┯━━━┯   ┯━━━┯━━━┳━━━┯━━━┯   ┯━━━┯━━━┳━━━┯━━━┯   ┯━━━┯━━━┓ ┐
            // x4 = ┃ q │ q │ ⋯ │ q │ q ┃ k │ k │ ⋯ │ k │ k ┃ v │ v │ ⋯ │ v │ v ┃ ├ ids.len() rows
            //      ┗━━━┷━━━┷   ┷━━━┷━━━┻━━━┷━━━┷   ┷━━━┷━━━┻━━━┷━━━┷   ┷━━━┷━━━┛ ┘
            //      └─┬─┘
            //        size_of_head columns

            for i_head in 0..config.n_head {
                let (q, k, v) = {
                    let mut q_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut k_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut v_colmaj = Vec::with_capacity(ids.len() * size_of_head);

                    for subcol in 0..size_of_head {
                        for row in 0..ids.len() {
                            let col = size_of_head * i_head + subcol;
                            q_colmaj.push(*x4.get(&[row, col])?);
                            k_colmaj.push(*x4.get(&[row, config.n_embd + col])?);
                            v_colmaj.push(*x4.get(&[row, 2 * config.n_embd + col])?);
                        }
                    }

                    let q = TypedTensor::<f32>::from_vec_col_major(
                        vec![ids.len(), size_of_head],
                        q_colmaj,
                    )?;
                    let k = TypedTensor::<f32>::from_vec_col_major(
                        vec![ids.len(), size_of_head],
                        k_colmaj,
                    )?;
                    let v = TypedTensor::<f32>::from_vec_col_major(
                        vec![ids.len(), size_of_head],
                        v_colmaj,
                    )?;

                    (q, k, v)
                };

                // transpose(k)
                let x5 = k.transpose(&[1, 0], backend)?;

                // √size_of_head
                let x6 = TypedTensor::<f32>::from_vec_col_major(
                    vec![],
                    vec![(size_of_head as f32).sqrt()],
                )?;

                // q * transpose(k) / √size_of_head
                let x7 = q.matmul(&x5, backend)?.div(&x6, backend)?;

                //      ⎛ a₁₁ a₁₂ a₁₃  ⋯  a₁ₙ ⎞        ⎛ a₁₁ −∞  −∞   ⋯  −∞  ⎞
                //      ⎜ a₂₁ a₂₂ a₂₃  ⋯  a₂ₙ ⎟        ⎜ a₂₁ a₂₂ −∞   ⋯  −∞  ⎟
                // x7 = ⎜ a₃₁ a₃₂ a₃₃  ⋯  a₃ₙ ⎟ → x8 = ⎜ a₃₁ a₃₂ a₃₃  ⋯  −∞  ⎟
                //      ⎜  ⋮   ⋮   ⋮   ⋱   ⋮  ⎟        ⎜  ⋮   ⋮   ⋮   ⋱   ⋮  ⎟
                //      ⎝ aₙ₁ aₙ₂ aₙ₃  ⋯  aₙₙ ⎠        ⎝ aₙ₁ aₙ₂ aₙ₃  ⋯  aₙₙ ⎠
                let x8 = {
                    let mut colmaj = Vec::with_capacity(ids.len() * ids.len());
                    for col in 0..ids.len() {
                        for row in 0..ids.len() {
                            colmaj.push(if col <= row {
                                *x7.get(&[row, col])?
                            } else {
                                f32::NEG_INFINITY
                            })
                        }
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), ids.len()], colmaj)?
                };

                // maximum(x8)
                let x9 = {
                    let mut colmaj = Vec::with_capacity(ids.len());
                    for row in 0..ids.len() {
                        let mut max = f32::NEG_INFINITY;
                        for col in 0..ids.len() {
                            let value = *x8.get(&[row, col])?;
                            if value > max {
                                max = value;
                            }
                        }
                        colmaj.push(max);
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), 1], colmaj)?
                };

                // x8 .- maximum(x8)
                let x10 = x8.sub(&x9, backend)?;

                // exp.(x8 .- maximum(x8))
                let x11 = x10.exp(backend)?;

                // sum(exp.(x8 .- maximum(x8))))
                let x12 = x11
                    .reduce_sum(&[1], backend)?
                    .reshape(&[ids.len(), 1], backend)?;

                // softmax(x8) = exp.(x8 .- maximum(x8)) ./ sum(exp.(x8 .- max(x8)))
                let x13 = x11.div(&x12, backend)?;

                // attention = softmax(x8) * v
                let x14 = x13.matmul(&v, backend)?;

                stacked_colmaj.extend(x14.as_slice()?);
            }
            TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), config.n_embd], stacked_colmaj)?
        };

        let attn_c_proj_weight = &tensors[&format!("h.{i_layer}.attn.c_proj.weight")];
        if attn_c_proj_weight.shape() != [config.n_embd, config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let attn_c_proj_bias = &tensors[&format!("h.{i_layer}.attn.c_proj.bias")];
        if attn_c_proj_bias.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x15 * attn_c_proj_weight .+ attn_c_proj_bias
        let x16 = x15
            .matmul(attn_c_proj_weight, backend)?
            .add(attn_c_proj_bias, backend)?;

        // x2 + x16
        let x17 = x2.add(&x16, backend).unwrap();

        // ==== Feed Forward ====

        let ln_2_weight = &tensors[&format!("h.{i_layer}.ln_2.weight")];
        if ln_2_weight.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let ln_2_bias = &tensors[&format!("h.{i_layer}.ln_2.bias")];
        if ln_2_bias.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let x18 = layer_norm(&x17, ln_2_weight, ln_2_bias, config, backend)?;

        let mlp_c_fc_weight = &tensors[&format!("h.{i_layer}.mlp.c_fc.weight")];
        if mlp_c_fc_weight.shape() != [config.n_embd, 4 * config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let mlp_c_fc_bias = &tensors[&format!("h.{i_layer}.mlp.c_fc.bias")];
        if mlp_c_fc_bias.shape() != [4 * config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x18 * mlp_c_fc_weight .+ mlp_c_fc_bias
        let x19 = x18
            .matmul(mlp_c_fc_weight, backend)?
            .add(mlp_c_fc_bias, backend)?;

        // The formula is according to the original paper of GELU.
        // https://arxiv.org/pdf/1606.08415

        // 0.044715
        let x20 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.044715])?;

        // √(2.0 / π)
        let x21 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![(2.0 / PI).sqrt()])?;

        // 1.0
        let x22 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0])?;

        // 0.5
        let x23 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.5])?;

        // GELU(x19) = (tanh.((x19 .^ 3 * 0.044715 + x19) * √(2.0 / π)) .+ 1.0) .* x19 * 0.5
        let x24 = x19
            .mul(&x19, backend)?
            .mul(&x19, backend)?
            .mul(&x20, backend)?
            .add(&x19, backend)?
            .mul(&x21, backend)?
            .tanh(backend)?
            .add(&x22, backend)?
            .mul(&x19, backend)?
            .mul(&x23, backend)?;

        let mlp_c_proj_weight = &tensors[&format!("h.{i_layer}.mlp.c_proj.weight")];
        if mlp_c_proj_weight.shape() != [4 * config.n_embd, config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let mlp_c_proj_bias = &tensors[&format!("h.{i_layer}.mlp.c_proj.bias")];
        if mlp_c_proj_bias.shape() != [config.n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x24 * mlp_c_proj_weight .+ mlp_c_proj_bias
        let x25 = x24
            .matmul(mlp_c_proj_weight, backend)?
            .add(mlp_c_proj_bias, backend)?;

        // x17 + x25
        x2 = x17.add(&x25, backend).unwrap();
    }

    // ==== Projection ====

    // x2[end]
    let x26 = {
        let mut colmaj = Vec::with_capacity(config.n_embd);
        for col in 0..config.n_embd {
            colmaj.push(*x2.get(&[ids.len() - 1, col])?);
        }
        TypedTensor::<f32>::from_vec_col_major(vec![1, config.n_embd], colmaj)?
    };

    let ln_f_weight = &tensors["ln_f.weight"];
    if ln_f_weight.shape() != [config.n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    let ln_f_bias = &tensors["ln_f.bias"];
    if ln_f_bias.shape() != [config.n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    let x27 = layer_norm(&x26, ln_f_weight, ln_f_bias, config, backend)?;

    // transpose(x2)
    let x28 = x27.reshape(&[config.n_embd, 1], backend)?;

    // wte_weight * transpose(x2)
    let x29 = wte_weight.matmul(&x28, backend)?;

    Ok(x29)
}
*/

fn layer_norm(
    x: &TypedTensor<f32>,
    g: &TypedTensor<f32>,
    t: &TypedTensor<f32>,
    model: &Model,
    backend: &mut tenferro_cpu::CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // mean(x) = sum(x) / n_embd
    let mean_x = x.reduce_sum(&[0], backend)?.div(&model.n_embd_as_tensor, backend)?;

    // diff(x) = x .- mean(x)
    let diff_x = x.sub(&mean_x, backend)?;

    // var(x, corrected = false) = sum(diff(x) .^ 2) / n_embd
    let var_x = diff_x.mul(&diff_x, backend)?.reduce_sum(&[0], backend)?.div(&model.n_embd_as_tensor, backend)?;

    // √(var(x, corrected = false) + e)
    let denom = var_x.add(&model.e, backend)?.sqrt(backend)?;

    // (x .- mean(x)) ./ √(var(x, corrected = false) + e) .* g + t
    let x = diff_x.div(&denom, backend)?.mul(&g, backend)?.add(&t, backend)?;

    Ok(x)

    /*
    // n_embd
    let x0 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![config.n_embd as f32])?;

    // mean(tensor) = sum(tensor / n_embd)
    let x1 = tensor
        .reduce_sum(&[1], backend)?
        .reshape(&[tensor.shape()[0], 1], backend)?
        .div(&x0, backend)?;

    // tensor .- mean(tensor)
    let x2 = tensor.sub(&x1, backend)?;

    // The original paper of linear normalization uses
    // H (n_embd) instead of H - 1 (n_embd - 1) for the denominator of variance.
    // https://arxiv.org/pdf/1607.06450

    // var(tensor) = sum((tensor .- mean(tensor) .^ 2) / n_embd
    let x3 = x2
        .mul(&x2, backend)?
        .reduce_sum(&[1], backend)?
        .reshape(&[tensor.shape()[0], 1], backend)?
        .div(&x0, backend)?;

    // layer_norm_epsilon
    let x4 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![config.layer_norm_epsilon])?;

    // .√(var(tensor) + layer_norm_epsilon)
    let x5 = x3.add(&x4, backend)?.sqrt(backend)?;

    // (tensor .- mean(tensor)) ./ .√(var(tensor) + layer_norm_epsilon) .* weight .+ bias
    let x6 = x2
        .div(&x5, backend)?
        .mul(weight, backend)?
        .add(bias, backend)?;

    Ok(x6)
    */
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
