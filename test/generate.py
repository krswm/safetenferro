# Generate a Safetensors file for testing purpose.

import numpy
from safetensors.numpy import save_file


def get(dtype: type) -> numpy.ndarray:
    x = [
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        [
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ],
    ]
    return numpy.array(x, dtype=dtype)


tensors = {
    "F32": get(numpy.float32),
    "F64": get(numpy.float64),
    "I8": get(numpy.int8),
    "U8": get(numpy.uint8),
    "I16": get(numpy.int16),
    "U16": get(numpy.uint16),
    "I32": get(numpy.int32),
    "U32": get(numpy.uint32),
    "I64": get(numpy.int64),
    "U64": get(numpy.uint64),
}
save_file(tensors, "test/tensors.safetensors")
