"""
LLM Inference Acceleration Engine

A research-oriented implementation of LLM inference optimization techniques:
- Quantization (GPTQ, AWQ, Mixed-Precision)
- Speculative Decoding
- KV Cache Optimization (PagedAttention)
- Comprehensive Benchmarking

This is a rigorous implementation focusing on research quality,
not just wrapping existing tools.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .core.config import EngineConfig
from .core.model_wrapper import OptimizedModel

__all__ = [
    "EngineConfig",
    "OptimizedModel",
]
