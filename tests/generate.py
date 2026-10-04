import numpy
from safetensors.numpy import save_file

tensors: dict[str, numpy.ndarray] = {
    "F32_tensor": numpy.array(range(24), dtype=numpy.float32).reshape(2, 3, 4),
    "F64_tensor": numpy.array(range(24), dtype=numpy.float64).reshape(2, 3, 4),
    "I32_tensor": numpy.array(range(24), dtype=numpy.int32).reshape(2, 3, 4),
    "I64_tensor": numpy.array(range(24), dtype=numpy.int64).reshape(2, 3, 4),
    "BOOL_tensor": numpy.array([], dtype=numpy.bool),
    "I8_tensor": numpy.array([], dtype=numpy.int8),
    "U8_tensor": numpy.array([], dtype=numpy.uint8),
    "I16_tensor": numpy.array([], dtype=numpy.int16),
    "U16_tensor": numpy.array([], dtype=numpy.uint16),
    "U32_tensor": numpy.array([], dtype=numpy.uint32),
    "U64_tensor": numpy.array([], dtype=numpy.uint64),
    "F16_tensor": numpy.array([], dtype=numpy.float16),
    "C64_tensor": numpy.array([], dtype=numpy.complex64),
}
metadata = {"metadata_key": "metadata_value"}
save_file(tensors, "tensors.safetensors", metadata)
