# Safetensors Loader for tenferro

This crate loads tensors from a Safetensors file as `tenferro_runtime::Tensor`s.

[Safetensors](https://github.com/safetensors/safetensors) is a serialization format to store tensors in a file.

[tenferro](https://github.com/tensor4all/tenferro-rs) is a Rust-native tensor library.

*safetenferro* is a wordplay of *Safetensors* and *tenferro*.

> [!NOTE]
> This crate is **not** a part of the official Safetensors (Hugging Face) project.
>
> This crate is **not** a part of the official tenferro (tensor4all) project.

> [!WARNING]
> **Anything of this crate (including public API and git URL) is subject to change.**
>
> Use this crate at your own risk.

## Example

Add this crate to `Cargo.toml`.
This crate is not available at crates.io.

```toml
safetenferro = { git = "https://github.com/krswm/safetenferro.git" }
```

Example: Load a Safetensors file at `tensors.safetensors`.

```rust
use std::collections::HashMap;
use std::error::Error;

use tenferro_runtime::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let tensors: HashMap<String, Tensor> = safetenferro::load_safetensors_permuted("tensors.safetensors")?;

    let tensor: &Tensor = &tensors["tensor_name"];

    println!("{:?} {:?} {:?}", tensor.dtype(), tensor.shape(), tensor.as_slice::<f64>()?);

    Ok(())
}
```

## Limitations (a.k.a. TODO)

- This crate does not write a Safetensors file.
- Tensors with dtypes other than `F32`, `F64`, `I32`, and `I64` are not supported and **silently ignored**.
- Metadata is not supported and ignored.
- The function in this crate does not validate the file for the points what [the specification](https://github.com/safetensors/safetensors#format) says a Safetensor file must obey.
  Namely:
  - The header must start with `{`.
  - The byte buffer need to be entirely indexed.

## [Dependencies](Cargo.toml)

- `serde_json` and `serde` to parse JSON.
- `tenferro-runtime`.

This crate requires `python3` command on the shell to generate a Safetensors file for unit test.
Just for `release` build, you do not need `python3`.

## Credits

- [Safetensors](https://github.com/safetensors/safetensors) for the specification, reference implementation for unit test, and inspiration.
- [serde-json](https://github.com/serde-rs/json) for a JSON library for Rust.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust.

## Development

This is a hobby project of mine I started from scratch.

This crate originated from a part of [my GPT-2 inference engine built on tenferro](https://github.com/krswm/slope-rs).

- 2026-07-08: I started implementing a Safetensors loader for tenferro as a part of my GPT-2 inference engine.
- 2026-10-01: I started to turn the loader code into a separate crate.

I enjoyed working on this project!
I learned some concepts of Rust: traits and generics.

I did **not** use generative AI for this project at all.
