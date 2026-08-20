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

pub mod loader;
pub mod tokenizer;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("GPT-2 Inference with tenferro");
        println!(
            "Usage: \x1b[1m{} <path to model repository> <your prompt>\x1b[22m",
            &args[0]
        );
        println!("You may have to enclose 'your prompt' with quotes.");
        return Ok(());
    }

    // ==== Loading Files ====

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
        transformer::get_model(tensors, config)?
    };

    // ==== Tokenization ====

    // Token IDs
    let ids = tokenizer::tokenize(&token_to_id, &ranks, &args[2])?;
    if ids.is_empty() {
        println!("Your prompt should not be empty.");
        return Ok(());
    } else if ids.len() >= model.n_ctx {
        println!("Your prompt exceeds the context length. Try shorter prompt.");
        return Ok(());
    }

    // ==== Inference ====

    let mut k_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];
    let mut v_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];

    let mut backend = CpuBackend::new();
    let mut id = 0usize;
    let mut utf8_buffer: Vec<u8> = Vec::new();
    let mut num_prompted_tokens = 0usize;
    let mut num_processed_tokens = 0usize;
    let mut num_generated_tokens = 0usize;

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
            // Greedy sampling: Choose the token with the highest probability.
            id = logits
                .host_data()?
                .iter()
                .enumerate()
                .max_by(|(_, prob0), (_, prob1)| prob0.total_cmp(prob1))
                .map(|(id, _)| id)
                .unwrap();
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
