# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for detecting model dtype from checkpoint files."""

from __future__ import annotations

import json
import os
import struct
from typing import Optional

import torch

# Mapping from safetensors dtype strings to torch dtypes.
_SAFETENSORS_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def detect_safetensors_dtype(model_path: str) -> Optional[torch.dtype]:
    """Detect the dominant floating-point dtype from a safetensors checkpoint.

    Reads only the header (no weight data loaded), making this very cheap.
    Returns the most common floating-point dtype among the tensors, or
    ``None`` if no safetensors file is found.

    Args:
        model_path: Directory containing ``model.safetensors``, or a direct
            path to a ``.safetensors`` file.
    """
    if os.path.isdir(model_path):
        sf_path = os.path.join(model_path, "model.safetensors")
    else:
        sf_path = model_path

    if not os.path.isfile(sf_path):
        return None

    try:
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
    except (OSError, struct.error, json.JSONDecodeError):
        return None

    # Count floating-point dtypes only.
    float_dtypes = {"F16", "BF16", "F32", "F64"}
    counts: dict[str, int] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dt = meta.get("dtype", "")
        if dt in float_dtypes:
            counts[dt] = counts.get(dt, 0) + 1

    if not counts:
        return None

    dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
    return _SAFETENSORS_DTYPE_MAP.get(dominant)


def resolve_model_dtype(
    torch_dtype: Optional[torch.dtype],
    model_path: Optional[str],
    default: torch.dtype = torch.bfloat16,
) -> torch.dtype:
    """Resolve the dtype to use for model components.

    Priority order:
    1. Explicit ``torch_dtype`` from YAML config (``precision`` field).
    2. Auto-detected from the safetensors checkpoint header.
    3. ``default`` fallback.

    Args:
        torch_dtype: Dtype from config, or ``None`` to auto-detect.
        model_path: Path to the checkpoint directory.
        default: Fallback dtype when detection is not possible.
    """
    if torch_dtype is not None:
        return torch_dtype

    if model_path is not None:
        detected = detect_safetensors_dtype(model_path)
        if detected is not None:
            return detected

    return default
