"""
Base Problem class for Tensara problem definitions.

This module provides the Problem base class that all problem definitions
inherit from. It is used by the local runner (run_local.py) to load and
execute problem definitions without requiring the full Tensara platform.
"""

import hashlib
import torch
from typing import Dict


class Problem:
    """Base class for all Tensara problem definitions."""

    parameters = []
    is_exact = False

    TYPE_TO_TORCH_DTYPE = {
        "float": torch.float32,
        "double": torch.float64,
        "float16": torch.float16,
        "int": torch.int32,
        "size_t": torch.int64,
        "uint8_t": torch.uint8,
        "uint32_t": torch.int32,
        "uint64_t": torch.int64,
    }

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def get_seed(seed_str: str) -> int:
        """Generate a deterministic seed from a string."""
        return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

    def param_dtype(self, index: int) -> torch.dtype:
        """Get the torch dtype for a parameter by index."""
        param_type = self.parameters[index]["type"]
        return self.TYPE_TO_TORCH_DTYPE.get(param_type, torch.float32)

    def get_function_signature(self) -> Dict:
        """Return the function signature derived from parameters."""
        return {"parameters": self.parameters}
