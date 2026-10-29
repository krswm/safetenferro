import torch
from safetensors.torch import save_file

tensors: dict[str, torch.Tensor] = {
    "f16_tensor": torch.tensor(range(24), dtype=torch.float16).reshape(2, 3, 4),
    "f32_tensor": torch.tensor(range(24), dtype=torch.float32).reshape(2, 3, 4),
    "f64_tensor": torch.tensor(range(24), dtype=torch.float64).reshape(2, 3, 4),
    "bf16_tensor": torch.tensor(range(24), dtype=torch.bfloat16).reshape(2, 3, 4),
    "i8_tensor": torch.tensor(range(24), dtype=torch.int8).reshape(2, 3, 4),
    "u8_tensor": torch.tensor(range(24), dtype=torch.uint8).reshape(2, 3, 4),
    "i16_tensor": torch.tensor(range(24), dtype=torch.int16).reshape(2, 3, 4),
    "u16_tensor": torch.tensor(range(24), dtype=torch.uint16).reshape(2, 3, 4),
    "i32_tensor": torch.tensor(range(24), dtype=torch.int32).reshape(2, 3, 4),
    "u32_tensor": torch.tensor(range(24), dtype=torch.uint32).reshape(2, 3, 4),
    "i64_tensor": torch.tensor(range(24), dtype=torch.int64).reshape(2, 3, 4),
    "u64_tensor": torch.tensor(range(24), dtype=torch.uint64).reshape(2, 3, 4),
    "bool_tensor": torch.tensor([], dtype=torch.bool),
}
metadata = {"metadata_key": "metadata_value"}
save_file(tensors, "tensors.safetensors", metadata)
