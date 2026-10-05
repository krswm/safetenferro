# Safetensors Loader for tenferro

This crate loads tensors from a Safetensors file as tenferro’s `Tensor`s.

[Safetensors](https://github.com/safetensors/safetensors) is a serialization
format to store tensors in a file.

*safetenferro* is a wordplay of *Safetensors* and *tenferro*.

> [!NOTE]
> This repository is **not** a part of the official Safetensors (Hugging Face) project.
> This repository is **not** a part of the official tenferro (tensor4all) project.
> This repository is a hobby project of mine.
> Use at your own risk.

## Example

Load tensors from a file `$OUT_DIR/tensors.safetensors`.

```rust
use std::path::PathBuf;

let out_dir = env!("OUT_DIR");
let path = PathBuf::from(&out_dir).join("tensors.safetensors");

safetenferro::load_safetensors_permuted(path)?;
```

## Limitations (a.k.a. TODO)

- This crate does not write a Safetensors file.
- Tensors with dtypes other than `F32`, `F64`, `I32`, and `I64` are not supported and **silently ignored**.
- Metadata is not supported and ignored.
- This function does not validate the file for the points what [the specification](https://github.com/safetensors/safetensors#format) says a Safetensor file must obey.
  The points are:
  - The header must start with `{`.
  - The byte buffer need to be entirely indexed.

## Development

This repository originated from a part of [my GPT-2 inference engine built on tenferro](https://github.com/krswm/slope-rs).

I did **not** use generative AI for this project at all.
