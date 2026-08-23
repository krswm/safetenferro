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

use std::error::Error;
use std::iter::zip;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

use crate::model::{Layer, Model};

fn layer_norm(
    x: &TypedTensor<f32>,
    g: &TypedTensor<f32>,
    t: &TypedTensor<f32>,
    model: &Model,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // mean(x) = sum(x) / n_embd
    let mean = x.reduce_sum(&[0], backend)?.mul(&model.c0, backend)?;

    // x .- mean(x)
    let numerator = x.sub(&mean, backend)?;

    // The paper that introduced layer norm uses uncorrected variance.
    // https://arxiv.org/abs/1607.06450
    // var(x, corrected = false) = sum((x .- mean(x)) .^ 2) / n_embd
    let var = numerator
        .mul(&numerator, backend)?
        .reduce_sum(&[0], backend)?
        .mul(&model.c0, backend)?;

    // √(var(x, corrected = false) + model.e)
    let denominator = var.add(&model.e, backend)?.sqrt(backend)?;

    // g .* (x .- mean(x)) ./ √(var(x, corrected = false) + model.e) + t
    let x = g
        .mul(&numerator, backend)?
        .div(&denominator, backend)?
        .add(t, backend)?;

    Ok(x)
}

fn multi_head_attention(
    x: &TypedTensor<f32>,
    layer: &Layer,
    k_colmaj: &mut Vec<f32>,
    v_colmaj: &mut Vec<f32>,
    pos: usize,
    model: &Model,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // x = layer.w11 * x + layer.b11
    let x = layer.w11.matmul(x, backend)?.add(&layer.b11, backend)?;
    // `x` is a 1D `TypedTensor`:
    // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    // ┃ q[i=0               h=0             ] ┃
    // ┠───────────────────────────────────────┨
    // ┃ q[i=1               h=0             ] ┃
    // ┠───────────────────────────────────────┨
    // ┃                   ⋮                   ┃
    // ┠───────────────────────────────────────┨
    // ┃ q[i=n_embd/n_head-1 h=0             ] ┃
    // ┠───────────────────────────────────────┨
    // ┃ q[i=0               h=1             ] ┃
    // ┠───────────────────────────────────────┨
    // ┃                   ⋮                   ┃
    // ┠───────────────────────────────────────┨
    // ┃ q[i=n_embd/n_head-1 h=n_head-1      ] ┃
    // ┠───────────────────────────────────────┨
    // ┃ k[i=0               h=0        p=pos] ┃
    // ┠───────────────────────────────────────┨
    // ┃                   ⋮                   ┃
    // ┠───────────────────────────────────────┨
    // ┃ k[i=n_embd/n_head-1 h=n_head-1 p=pos] ┃
    // ┠───────────────────────────────────────┨
    // ┃ v[i=0               h=0        p=pos] ┃
    // ┠───────────────────────────────────────┨
    // ┃                   ⋮                   ┃
    // ┠───────────────────────────────────────┨
    // ┃ v[i=n_embd/n_head-1 h=n_head-1 p=pos] ┃
    // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    let mut chunks = x.host_data()?.chunks(model.n_embd);

    let q = TypedTensor::<f32>::from_vec_col_major(
        vec![model.n_embd / model.n_head, model.n_head],
        chunks.next().unwrap().to_vec(),
    )?;
    // `q` is a 2D `TypedTensor`:
    //    ┏━━━━━━━━━━━━━━━━━━┓
    //    ┃ q[i=0    h=last] ┃
    //            ⋰         ─┨
    // ┏━━━━━━━━━━━━━━━━━━┓  ┃
    // ┃ q[i=0    h=0   ] ┃ ─┨
    // ┠──────────────────┨  ┃
    // ┃        ⋮         ┃ ━┛
    // ┠──────────────────┨
    // ┃ q[i=last h=0   ] ┃
    // ┗━━━━━━━━━━━━━━━━━━┛

    k_colmaj.extend_from_slice(chunks.next().unwrap());
    v_colmaj.extend_from_slice(chunks.next().unwrap());
    // `k_colmaj` is a `Vec` (same for `v_colmaj`):
    // ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    // ┃ k[i=0    h=0    p=0    ] ┃
    // ┠──────────────────────────┨
    // ┃ k[i=1    h=0    p=0    ] ┃
    // ┠──────────────────────────┨
    // ┃            ⋮             ┃
    // ┠──────────────────────────┨
    // ┃ k[i=last h=0    p=0    ] ┃
    // ┠──────────────────────────┨
    // ┃ k[i=0    h=1    p=0    ] ┃
    // ┠──────────────────────────┨
    // ┃            ⋮             ┃
    // ┠──────────────────────────┨
    // ┃ k[i=last h=last p=0    ] ┃
    // ┠──────────────────────────┨
    // ┃ k[i=0    h=0    p=1    ] ┃
    // ┠──────────────────────────┨
    // ┃            ⋮             ┃
    // ┠──────────────────────────┨
    // ┃ k[i=last h=last p=pos-1] ┃
    // ┠──────────────────────────┨ ╮
    // ┃ k[i=0    h=0    p=pos  ] ┃ │
    // ┠──────────────────────────┨ │
    // ┃            ⋮             ┃ ├ Newly extended values
    // ┠──────────────────────────┨ │
    // ┃ k[i=last h=last p=pos  ] ┃ │
    // ┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛ ╯

    let k = TypedTensor::<f32>::from_vec_col_major(
        vec![model.n_embd / model.n_head, model.n_head, pos + 1],
        k_colmaj.to_vec(),
    )?;
    let v = TypedTensor::<f32>::from_vec_col_major(
        vec![model.n_embd / model.n_head, model.n_head, pos + 1],
        v_colmaj.to_vec(),
    )?;
    // `k` is a 3D `TypedTensor` (same for `v`):
    //    ┏━━━━━━━━━━━━━━━━━━━━━━━━┯━━━┯━━━━━━━━━━━━━━━━━━━━━━━━┓
    //    ┃ k[i=0    h=last p=0  ] │ ⋯ │ k[i=0    h=last p=pos] ┃
    //                              ⋰                          ─┨
    // ┏━━━━━━━━━━━━━━━━━━━━━━━━┯━━━┯━━━━━━━━━━━━━━━━━━━━━━━━┓  ┃
    // ┃ k[i=0    h=0    p=0  ] │ ⋯ │ k[i=0    h=0    p=pos] ┃ ─┨
    // ┠────────────────────────┼───┼────────────────────────┨  ┃
    // ┃            ⋮           │   │            ⋮           ┃ ━┛
    // ┠────────────────────────┼───┼────────────────────────┨
    // ┃ k[i=last h=0    p=0  ] │ ⋯ │ k[i=last h=0    p=pos] ┃
    // ┗━━━━━━━━━━━━━━━━━━━━━━━━┷━━━┷━━━━━━━━━━━━━━━━━━━━━━━━┛

    // y = transpose.(k) .* q ./ √Float32(model.n_embd ÷ model.n_head)
    let y = [&k, &q]
        .einsum("ihp,ih->ph", backend)?
        .mul(&model.c1, backend)?;
    // 2D `TypedTensor`:
    //    ┏━━━━━━━━━━━━━━┓
    //    ┃ p=0   h=last ┃
    //          ⋰       ─┨
    // ┏━━━━━━━━━━━━━━┓  ┃
    // ┃ p=0   h=0    ┃ ─┨
    // ┠──────────────┨  ┃
    // ┃      ⋮       ┃ ━┛
    // ┠──────────────┨
    // ┃ p=pos h=0    ┃
    // ┗━━━━━━━━━━━━━━┛

    // maximum(y)
    let maximum = {
        let colmaj: Vec<f32> = y
            .host_data()?
            .chunks(pos + 1)
            .map(|chunk| {
                *chunk
                    .iter()
                    .max_by(|value0, value1| value0.total_cmp(value1))
                    .unwrap()
            })
            .collect();
        TypedTensor::<f32>::from_vec_col_major(vec![1, model.n_head], colmaj)?
    };
    // 1D `TypedTensor`:
    //    ┏━━━━━━━━┓
    //    ┃ h=last ┃
    //       ⋰    ━┛
    // ┏━━━━━━━━┓
    // ┃ h=0    ┃
    // ┗━━━━━━━━┛

    // y .- maximum(y)
    let numerator = y.sub(&maximum, backend)?.exp(backend)?;
    // 2D `TypedTensor`:
    //    ┏━━━━━━━━━━━━━━┓
    //    ┃ p=0   h=last ┃
    //          ⋰       ─┨
    // ┏━━━━━━━━━━━━━━┓  ┃
    // ┃ p=0   h=0    ┃ ─┨
    // ┠──────────────┨  ┃
    // ┃      ⋮       ┃ ━┛
    // ┠──────────────┨
    // ┃ p=pos h=0    ┃
    // ┗━━━━━━━━━━━━━━┛

    // sum(y .- maximum(y))
    let denominator = numerator
        .reduce_sum(&[0], backend)?
        .reshape(&[1, model.n_head], backend)?;
    // 1D `TypedTensor`:
    //    ┏━━━━━━━━┓
    //    ┃ h=last ┃
    //       ⋰    ━┛
    // ┏━━━━━━━━┓
    // ┃ h=0    ┃
    // ┗━━━━━━━━┛

    // Numerically stable softmax
    // softmax.(y) = (y .- maximum(y)) ./ sum(y .- maximum(y))
    let softmax = numerator.div(&denominator, backend)?;
    // 2D `TypedTensor`:
    //    ┏━━━━━━━━━━━━━━┓
    //    ┃ p=0   h=last ┃
    //          ⋰       ─┨
    // ┏━━━━━━━━━━━━━━┓  ┃
    // ┃ p=0   h=0    ┃ ─┨
    // ┠──────────────┨  ┃
    // ┃      ⋮       ┃ ━┛
    // ┠──────────────┨
    // ┃ p=pos h=0    ┃
    // ┗━━━━━━━━━━━━━━┛

    // Scaled dot-product attention
    // a = v .* softmax.(y)
    let a = [&v, &softmax].einsum("ihp,ph->ih", backend)?;
    // `a` is a 2D `TypedTensor`:
    //    ┏━━━━━━━━━━━━━━━━━━┓
    //    ┃ a[i=0    h=last] ┃
    //            ⋰         ─┨
    // ┏━━━━━━━━━━━━━━━━━━┓  ┃
    // ┃ a[i=0    h=0   ] ┃ ─┨
    // ┠──────────────────┨  ┃
    // ┃        ⋮         ┃ ━┛
    // ┠──────────────────┨
    // ┃ a[i=last h=0   ] ┃
    // ┗━━━━━━━━━━━━━━━━━━┛

    // x = vcat(a...)
    let x = a.reshape(&[model.n_embd, 1], backend)?;
    // `x` is a 1D `TypedTensor`:
    // ┏━━━━━━━━━━━━━━━━━━┓
    // ┃ a[i=0    h=0   ] ┃
    // ┠──────────────────┨
    // ┃ a[i=1    h=0   ] ┃
    // ┠──────────────────┨
    // ┃        ⋮         ┃
    // ┠──────────────────┨
    // ┃ a[i=last h=0   ] ┃
    // ┠──────────────────┨
    // ┃ a[i=0    h=1   ] ┃
    // ┠──────────────────┨
    // ┃        ⋮         ┃
    // ┠──────────────────┨
    // ┃ a[i=last h=last] ┃
    // ┗━━━━━━━━━━━━━━━━━━┛

    // x = layer.w12 * x + layer.b12
    let x = layer.w12.matmul(&x, backend)?.add(&layer.b12, backend)?;

    Ok(x)
}

fn feed_forward(
    x: &TypedTensor<f32>,
    layer: &Layer,
    model: &Model,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // x = layer.w21 * x + layer.b21
    let x = layer.w21.matmul(x, backend)?.add(&layer.b21, backend)?;

    // This formula is based on the paper that introduced GELU.
    // https://arxiv.org/abs/1606.08415
    // x = (tanh.((x .^ 3 * 0.044715f0 + x) * √(2.0f0 / π)) .+ 1.0f0) .* x * 0.5f0
    let x = x
        .mul(&x, backend)?
        .mul(&x, backend)?
        .mul(&model.c2, backend)?
        .add(&x, backend)?
        .mul(&model.c3, backend)?
        .tanh(backend)?
        .add(&model.c4, backend)?
        .mul(&x, backend)?
        .mul(&model.c5, backend)?;

    // x = layer.w22 * x + layer.b22
    let x = layer.w22.matmul(&x, backend)?.add(&layer.b22, backend)?;

    Ok(x)
}

// The transformer of the GPT-2 architecture.
pub fn transformer(
    id: usize,
    pos: usize,
    model: &Model,
    k_colmaj_caches: &mut [Vec<f32>],
    v_colmaj_caches: &mut [Vec<f32>],
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // x = model.wte[:, id+1] + model.wpe[:, pos]
    let mut x = model.id_embd_vecs[id].add(&model.pos_embd_vecs[pos], backend)?;

    for (layer, (k_colmaj, v_colmaj)) in zip(
        &model.layers,
        zip(k_colmaj_caches.iter_mut(), v_colmaj_caches.iter_mut()),
    ) {
        // y = layer_norm(x, layer.g1, layer.t1, model)
        let y = layer_norm(&x, &layer.g1, &layer.t1, model, backend)?;

        // y = multi_head_attention!(y, layer, k, v, pos, model)
        let y = multi_head_attention(&y, layer, k_colmaj, v_colmaj, pos, model, backend)?;

        // x += y
        x = x.add(&y, backend)?;

        // y = layer_norm(x, layer.g2, layer.t2, model)
        let y = layer_norm(&x, &layer.g2, &layer.t2, model, backend)?;

        // y = feed_forward(y, layer, model)
        let y = feed_forward(&y, layer, model, backend)?;

        // x += y
        x = x.add(&y, backend)?;
    }

    // x = layer_norm(x, model.gf, model.tf, model)
    x = layer_norm(&x, &model.gf, &model.tf, model, backend)?;

    // x = transpose(model.wte) * x
    x = model.wte_transposed.matmul(&x, backend)?;

    Ok(x)
}

// Pretty-print a 2D `TypedTensor` for debug.
#[allow(dead_code)]
fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    if tensor.shape().len() != 2 {
        return Err("`tensor` not 2D".into());
    }
    let num_rows = tensor.shape()[0];
    let num_cols = tensor.shape()[1];
    if num_rows == 0 {
        return Err("`num_rows` not 0".into());
    }
    if num_cols == 0 {
        return Err("`num_cols` not 0".into());
    }

    println!("┏{:━^14}┯━━━┯{:━^14}┓ ╮", "", "");
    println!(
        "┃ {:<+12.6e} │ ⋯ │ {:<+12.6e} ┃ │",
        tensor.get(&[0, 0]).unwrap(),
        tensor.get(&[0, num_cols - 1]).unwrap(),
    );
    println!("┠{:─^14}┼───┼{:─^14}┨ │", "", "");
    println!("┃ {:^12} │   │ {:^12} ┃ {num_rows}", "⋮", "⋮");
    println!("┠{:─^14}┼───┼{:─^14}┨ │", "", "");
    println!(
        "┃ {:<+12.6e} │ ⋯ │ {:<+12.6e} ┃ │",
        tensor.get(&[num_rows - 1, 0]).unwrap(),
        tensor.get(&[num_rows - 1, num_cols - 1]).unwrap(),
    );
    println!("┗{:━^14}┷━━━┷{:━^14}┛ ╯", "", "");
    println!("╰{num_cols:─^33}╯");

    Ok(())
}
