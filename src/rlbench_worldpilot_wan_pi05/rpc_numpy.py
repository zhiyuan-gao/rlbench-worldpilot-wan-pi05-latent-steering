from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_NDARRAY_SENTINEL = "__rlbench_worldpilot_ndarray__"


def encode_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            _NDARRAY_SENTINEL: True,
            "dtype": array.dtype.str,
            "shape": array.shape,
            "data": array.tobytes(),
        }
    if isinstance(value, Mapping):
        return {key: encode_arrays(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(encode_arrays(item) for item in value)
    if isinstance(value, list):
        return [encode_arrays(item) for item in value]
    return value


def decode_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get(_NDARRAY_SENTINEL):
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"])).copy()
        return {key: decode_arrays(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(decode_arrays(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [decode_arrays(item) for item in value]
    return value
