"""
Core Configuration for Inference Engine

Defines all optimization strategies and their hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class QuantizationType(Enum):
    """Supported quantization methods"""
    NONE = "none"
    INT8 = "int8"  # Basic INT8 quantization
    GPTQ = "gptq"  # Gradient-based Post-training Quantization
    AWQ = "awq"    # Activation-aware Weight Quantization
    MIXED = "mixed"  # Mixed-precision quantization


class AttentionBackend(Enum):
    """KV cache implementation backends"""
    STANDARD = "standard"      # Standard PyTorch attention
    PAGED = "paged"            # PagedAttention (vLLM-style)
    FLASH = "flash"            # FlashAttention


@dataclass
class QuantizationConfig:
    """
    Quantization configuration
    
    Key papers:
    - GPTQ: https://arxiv.org/abs/2210.17323
    - AWQ: https://arxiv.org/abs/2306.00978
    """
    method: QuantizationType = QuantizationType.GPTQ
    bits: int = 4              # 2, 3, 4, 8
    group_size: int = 128      # Group size for quantization
    
    # GPTQ-specific
    actorder: bool = True      # Activation order heuristic
    true_sequential: bool = True
    
    # AWQ-specific
    zero_point: bool = True
    
    # Mixed-precision
    sensitive_layers: List[str] = field(default_factory=list)
    sensitive_bits: int = 8    # Higher precision for sensitive layers
    
    # Calibration
    calibration_samples: int = 128
    calibration_dataset: str = "c4"  # c4, wikitext2, ptb


@dataclass
class SpeculativeConfig:
    """
    Speculative decoding configuration
    
    Key papers:
    - Speculative Decoding: https://arxiv.org/abs/2211.17192
    - Lookahead Decoding: https://arxiv.org/abs/2402.02057
    """
    enabled: bool = True
    draft_model_name: Optional[str] = None  # If None, use smaller version of target
    
    # Draft model config
    draft_model_size: str = "auto"  # auto, or specific like "1b", "7b"
    
    # Speculation parameters
    num_speculative_tokens: int = 5  # K in the paper (gamma)
    acceptance_threshold: float = 0.9
    
    # Adaptive speculation
    adaptive_speculation: bool = True
    min_acceptance_rate: float = 0.6  # Reduce K if acceptance < threshold


@dataclass
class KVCacheConfig:
    """
    KV cache optimization configuration
    
    Key papers:
    - PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
    - FlashAttention: https://arxiv.org/abs/2205.14135
    """
    backend: AttentionBackend = AttentionBackend.PAGED
    
    # PagedAttention specific
    block_size: int = 16       # Tokens per block
    max_num_blocks: int = 1024
    gpu_memory_utilization: float = 0.90
    
    # Memory management
    enable_prefix_caching: bool = True  # Reuse common prefixes
    
    # Continuous batching
    max_num_seqs: int = 256    # Max concurrent sequences
    max_model_len: int = 8192
    
    # FlashAttention specific
    flash_attn_dtype: str = "fp16"


@dataclass
class EngineConfig:
    """
    Main configuration for the inference engine
    
    Combines all optimization strategies into a unified configuration.
    """
    model_name: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    dtype: str = "float16"  # float16, bfloat16, float32
    
    # Optimization modules
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    
    # Generation config
    max_new_tokens: int = 512
    batch_size: int = 1
    
    # Benchmarking
    benchmark_warmup: int = 3
    benchmark_iterations: int = 10
    
    def validate(self):
        """Validate configuration"""
        if self.quantization.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Unsupported bits: {self.quantization.bits}")
        
        if self.speculative.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        
        return True
