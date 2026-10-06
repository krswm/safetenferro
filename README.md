# Safetensors Loader for tenferro

This crate loads tensors from a Safetensors file as `tenferro_runtime::Tensor`s.

[Safetensors](https://github.com/safetensors/safetensors) is a serialization format to store tensors in a file.

[tenferro](https://github.com/tensor4all/tenferro-rs) is a Rust-native tensor library.

*safetenferro* is a wordplay of *Safetensors* and *tenferro*.

> [!NOTE]
> This repository is **not** a part of the official Safetensors (Hugging Face) project.
>
> This repository is **not** a part of the official tenferro (tensor4all) project.
>
> **Anything of this crate (including public API and git URL) is subject of change.**
>
> Use at your own risk.

## Limitations (a.k.a. TODO)

- This crate does not write a Safetensors file.
- Tensors with dtypes other than `F32`, `F64`, `I32`, and `I64` are not supported and **silently ignored**.
- Metadata is not supported and ignored.
- The function in this crate does not validate the file for the points what [the specification](https://github.com/safetensors/safetensors#format) says a Safetensor file must obey.
  Namely:
  - The header must start with `{`.
  - The byte buffer need to be entirely indexed.

## Credits

- [Safetensors](https://github.com/safetensors/safetensors) for the specification, reference implementation for unit test, and inspiration.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust.

## Development

This is a hobby project of mine I started from scratch.

This crate originated from a part of [my GPT-2 inference engine built on tenferro](https://github.com/krswm/slope-rs).

- 2026-07-08: I started implementing a Safetensors loader for tenferro as a part of my GPT-2 inference engine.
- 2026-10-01: I started to turn the loader code into a separate crate.

I enjoyed working on this project!
I learned some concepts of Rust: traits and generics.

I did **not** use generative AI for this project at all.
