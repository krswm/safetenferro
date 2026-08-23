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
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub mod loader;
pub mod model;
pub mod tokenizer;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        println!("GPT-2 Inference with tenferro");
        println!(
            "Usage: \x1b[1m{} <path to model repository> <sampling temperature> <your prompt>\x1b[22m",
            &args[0]
        );
        println!("You may have to enclose 'your prompt' with quotes.");
        return Ok(());
    }

    let (token_to_id, id_to_token) = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)?;
        let id_to_token: HashMap<usize, String> = token_to_id
            .iter()
            .map(|(id, token)| (*token, id.clone()))
            .collect();
        (token_to_id, id_to_token)
    };

    let ranks = {
        let path = &format!("{}/merges.txt", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for line in reader.lines().map_while(Result::ok) {
            // Skip a comment line.
            if line.starts_with("#") {
                continue;
            }
            let mut split = line.split(" ");
            let token0 = split.next().unwrap().to_string();
            let token1 = split.next().unwrap().to_string();
            ranks.insert((token0, token1), rank);
            rank += 1;
        }
        ranks
    };

    let model = {
        let tensors = {
            let path = &format!("{}/model.safetensors", &args[1]);
            loader::load_safetensors(path)?
        };
        let config: HashMap<String, Value> = {
            let path = &format!("{}/config.json", &args[1]);
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };
        model::get_model(tensors, config)?
    };

    let (is_deterministic, beta) = {
        let temperature: f32 = args[2].parse()?;
        if temperature < 0.0f32 {
            println!("Temperature must be ≥ 0.0.");
            return Ok(());
        }
        let value = 1.0f32 / temperature;
        (
            temperature == 0.0f32,
            TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?,
        )
    };

    // ==== Tokenization ====

    // Token IDs
    let ids = tokenizer::tokenize(&token_to_id, &ranks, &args[3])?;
    if ids.is_empty() {
        println!("Your prompt should not be empty.");
        return Ok(());
    } else if ids.len() >= model.n_ctx {
        println!("Your prompt exceeds the context length. Try shorter prompt.");
        return Ok(());
    }

    // ==== Inference ====

    let mut num_prompted_tokens = 0usize;
    let mut num_processed_tokens = 0usize;
    let mut num_generated_tokens = 0usize;
    let mut id = 0usize;
    let mut utf8_buffer: Vec<u8> = Vec::new();
    let mut k_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];
    let mut v_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];
    let mut backend = CpuBackend::new();

    let performance_timer = Instant::now();
    for pos in 0..model.n_ctx {
        if pos < ids.len() {
            id = ids[pos];
            let decoded = tokenizer::decode_unique_encoding(&id_to_token[&id], &mut utf8_buffer);
            print!("\x1b[1;90m{decoded}\x1b[22;39m");
            std::io::stdout().flush()?;
            num_prompted_tokens += 1;
        }

        let logits = transformer::transformer(
            id,
            pos,
            &model,
            &mut k_colmaj_caches,
            &mut v_colmaj_caches,
            &mut backend,
        )?;
        num_processed_tokens += 1;

        if pos >= ids.len() - 1 {
            if is_deterministic {
                id = logits
                    .host_data()?
                    .iter()
                    .enumerate()
                    .max_by(|(_, prob0), (_, prob1)| prob0.total_cmp(prob1))
                    .map(|(id, _)| id)
                    .unwrap();
            } else {
                // logits ./ temperature
                let x = logits.mul(&beta, &mut backend)?;

                // maximum(logits ./ temperature)
                let maximum = {
                    let value = *x
                        .host_data()?
                        .iter()
                        .max_by(|value0, value1| value0.total_cmp(value1))
                        .unwrap();
                    TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
                };

                // numerator = exp.((logits ./ temperature) .- maximum(logits ./ temperature))
                let numerator = x.sub(&maximum, &mut backend)?.exp(&mut backend)?;

                // denominator = sum(exp.((logits ./ temperature) .- maximum(logits ./ temperature)))
                let denominator = numerator.reduce_sum(&[0], &mut backend)?;

                // numerator ./ denominator
                let x = numerator.div(&denominator, &mut backend)?;

                let rand_prob = rand::random_range(0.0f32..1.0f32);
                let mut total_prob = 0.0f32;
                for (x_id, prob) in x.host_data()?.iter().enumerate() {
                    total_prob += prob;
                    if rand_prob < total_prob {
                        id = x_id;
                        break;
                    }
                }
            }
            let decoded = tokenizer::decode_unique_encoding(&id_to_token[&id], &mut utf8_buffer);
            print!("\x1b[1m{decoded}\x1b[22m");
            std::io::stdout().flush()?;
            num_generated_tokens += 1;
        }
    }
    println!();
    let performance_time = performance_timer.elapsed().as_secs_f64();

    println!("\x1b[90mTook {performance_time:.3} s\x1b[39m");
    println!(
        "\x1b[90m{num_prompted_tokens} {} prompted\x1b[39m",
        if num_prompted_tokens == 1 {
            "token"
        } else {
            "tokens"
        }
    );
    println!(
        "\x1b[90m{num_processed_tokens} {} processed by the transformer \
        ({:.3} tok/s)\x1b[39m",
        if num_processed_tokens == 1 {
            "token"
        } else {
            "tokens"
        },
        (num_processed_tokens as f64) / performance_time
    );
    println!(
        "\x1b[90m{num_generated_tokens} {} generated\x1b[39m",
        if num_generated_tokens == 1 {
            "token"
        } else {
            "tokens"
        }
    );

    Ok(())
}
