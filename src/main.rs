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

    // k and v are 3D tensors with indices (i, h, p).
    //
    // - i = 0..(n_embd / n_head)  Index for elements in a head vector
    // - h = 0..n_head             Index for head vector
    // - p = 0..(pos + 1)          Token position
    //
    // `pos` increases by one for each transformer call, so overall lengths of k and v grow.
    //
    // I deliberately chose this order of indices from the fact that tenferro uses column-major.
    //
    // For each transformer layer, this program has to concatinate a new head-vector to the cached matrix.
    // Since p is the last index, this program only has to extend the new head-vector
    // to the column-major representation of the cached matrix.
    //
    // For each transformer layer, this program obtains a new head-vector.
    // A head-vector is ordered like:
    //   (i=0, h=0), (i=1, h=0), ..., (i=last, h=0), (i=0, h=1), (i=1, h=1), ...
    // Since i is the first and h is the second indices, this program can extend the new head-vector
    // to the column-major representation of the cached matrix without transposing.
    let mut k_cache_colmaj = vec![Vec::<f32>::new(); model.n_layer];
    let mut v_cache_colmaj = vec![Vec::<f32>::new(); model.n_layer];

    let mut utf8_buffer: Vec<u8> = Vec::new();

    let performance_timer = Instant::now();
    let mut backend = CpuBackend::new();
    for (pos, id) in ids[..(ids.len() - 1)].iter().enumerate() {
        let decoded = tokenizer::decode_unique_encoding(&id_to_token[id], &mut utf8_buffer);
        print!("\x1b[1;90m{decoded}\x1b[22;39m");
        std::io::stdout().flush()?;
        transformer::transform(
            &mut k_cache_colmaj,
            &mut v_cache_colmaj,
            &model,
            *id,
            pos,
            &mut backend,
        )?;
    }
    let mut id = ids[ids.len() - 1];
    let decoded = tokenizer::decode_unique_encoding(&id_to_token[&id], &mut utf8_buffer);
    print!("\x1b[1;90m{decoded}\x1b[22;39m");
    std::io::stdout().flush()?;
    for pos in (ids.len() - 1)..model.n_ctx {
        let logits = transformer::transform(
            &mut k_cache_colmaj,
            &mut v_cache_colmaj,
            &model,
            id,
            pos,
            &mut backend,
        )?;

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
    }
    let performance_time = performance_timer.elapsed().as_secs_f64();

    println!();
    println!(
        "\x1b[90mTransformer processed {} tokens\x1b[39m",
        model.n_ctx
    );
    println!(
        "\x1b[90mTook {performance_time:.3} s | {:.3} tokens/s",
        (model.n_ctx as f64) / performance_time
    );
    println!(
        "\x1b[90m{} tokens prompted | {} tokens generated",
        ids.len(),
        model.n_ctx - ids.len() + 1 // There is an extra token.
    );

    Ok(())
}
