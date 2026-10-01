# Generate a Safetensors file for testing purpose.

import numpy
from safetensors.numpy import save_file

tensors: dict[str, numpy.ndarray] = {
    "F32": numpy.array(range(24), dtype=numpy.float32).reshape(2, 3, 4),
    "F64": numpy.array(range(24), dtype=numpy.float64).reshape(2, 3, 4),
    "I32": numpy.array(range(24), dtype=numpy.int32).reshape(2, 3, 4),
    "I64": numpy.array(range(24), dtype=numpy.int64).reshape(2, 3, 4),
}
save_file(tensors, "test/tensors.safetensors")
