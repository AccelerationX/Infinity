"""
Comprehensive Benchmark Metrics for LLM Inference

Defines rigorous metrics for evaluating:
1. Performance (speed, throughput)
2. Accuracy (perplexity, downstream tasks)
3. Memory efficiency
4. Energy efficiency (optional)
"""
import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class PerformanceMetrics:
    """
    Performance-related metrics
    """
    # Latency metrics
    time_to_first_token_ms: float = 0.0  # TTFT: Time to generate first token
    time_per_output_token_ms: float = 0.0  # TPOT: Time per token after first
    total_latency_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    sequences_per_second: float = 0.0
    
    # Batch metrics
    batch_size: int = 1
    total_tokens_generated: int = 0
    total_sequences: int = 0
    
    # Hardware utilization
    gpu_utilization_percent: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    
    def throughput(self) -> float:
        """Overall throughput in tokens/second"""
        if self.total_latency_ms > 0:
            return self.total_tokens_generated / (self.total_latency_ms / 1000)
        return 0.0


@dataclass
class AccuracyMetrics:
    """
    Accuracy-related metrics
    
    Key insight: Speedup is meaningless without accuracy validation.
    """
    # Perplexity (lower is better)
    perplexity: float = 0.0
    perplexity_delta: float = 0.0  # Change from baseline
    
    # Generation quality
    bleu_score: Optional[float] = None  # For translation/summarization
    rouge_l: Optional[float] = None     # For summarization
    
    # Task-specific accuracy
    downstream_accuracy: Optional[float] = None  # Accuracy on downstream tasks
    
    # Statistical significance
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    p_value: Optional[float] = None
    
    def is_acceptable(self, max_perplexity_increase: float = 0.1) -> bool:
        """
        Check if accuracy degradation is acceptable.
        
        Args:
            max_perplexity_increase: Maximum allowed perplexity increase
        
        Returns:
            True if accuracy is acceptable
        """
        return self.perplexity_delta < max_perplexity_increase


@dataclass
class MemoryMetrics:
    """
    Memory efficiency metrics
    """
    # Model size
    model_size_mb: float = 0.0
    model_size_params: int = 0
    
    # KV cache
    kv_cache_size_mb: float = 0.0
    kv_cache_utilization: float = 0.0  # Actual used / allocated
    
    # Peak memory
    peak_memory_mb: float = 0.0
    peak_memory_allocated_mb: float = 0.0
    
    # Memory efficiency
    memory_efficiency: float = 0.0  # model_size / peak_memory
    
    # Quantization-specific
    compression_ratio: float = 1.0  # Original size / Quantized size
    bits_per_parameter: float = 16.0  # Default FP16


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result
    """
    name: str  # Configuration name
    config: Dict = field(default_factory=dict)
    
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    
    # Metadata
    timestamp: str = ""
    hardware_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "config": self.config,
            "performance": {
                "ttft_ms": self.performance.time_to_first_token_ms,
                "tpot_ms": self.performance.time_per_output_token_ms,
                "total_latency_ms": self.performance.total_latency_ms,
                "tokens_per_second": self.performance.tokens_per_second,
                "throughput": self.performance.throughput(),
            },
            "accuracy": {
                "perplexity": self.accuracy.perplexity,
                "perplexity_delta": self.accuracy.perplexity_delta,
                "is_acceptable": self.accuracy.is_acceptable(),
            },
            "memory": {
                "model_size_mb": self.memory.model_size_mb,
                "compression_ratio": self.memory.compression_ratio,
                "peak_memory_mb": self.memory.peak_memory_mb,
            },
            "hardware": self.hardware_info,
        }
    
    def save(self, path: str):
        """Save to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class PerplexityEvaluator:
    """
    Evaluates model perplexity on a dataset.
    
    Perplexity = exp(-mean(log P(x_i | x_<i)))
    
    Lower perplexity = better model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate(
        self,
        dataset,  # List of texts
        max_length: int = 512,
        stride: int = 512,
    ) -> float:
        """
        Evaluate perplexity on a dataset.
        
        Uses sliding window approach for long texts.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in dataset:
                # Tokenize
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                
                input_ids = encodings.input_ids.to(self.device)
                
                # Compute loss
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate
                num_tokens = input_ids.shape[1]
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        mean_loss = total_loss / total_tokens
        perplexity = np.exp(mean_loss)
        
        return perplexity


class LatencyBenchmark:
    """
    Benchmarks inference latency with rigorous methodology.
    
    Best practices:
    1. Warmup runs to stabilize GPU caches
    2. Multiple iterations for statistical significance
    3. Synchronize CUDA before timing
    4. Report both mean and confidence intervals
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
    
    def benchmark_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark single prompt.
        
        Returns:
            Dictionary with TTFT, TPOT, total latency
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
        
        # Benchmark
        ttft_list = []
        tpot_list = []
        
        for _ in range(self.benchmark_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            # Time to first token
            with torch.no_grad():
                first_output = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
            
            torch.cuda.synchronize()
            ttft = time.time() - start
            ttft_list.append(ttft * 1000)  # Convert to ms
            
            # Time per output token
            if max_new_tokens > 1:
                start = time.time()
                
                with torch.no_grad():
                    full_output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                
                torch.cuda.synchronize()
                total_time = time.time() - start
                
                # TPOT = (total_time - ttft) / (num_tokens - 1)
                num_new_tokens = full_output.shape[1] - inputs.input_ids.shape[1]
                if num_new_tokens > 1:
                    tpot = (total_time - ttft) / (num_new_tokens - 1) * 1000
                    tpot_list.append(tpot)
        
        # Compute statistics
        results = {
            "ttft_mean_ms": np.mean(ttft_list),
            "ttft_std_ms": np.std(ttft_list),
            "ttft_p95_ms": np.percentile(ttft_list, 95),
        }
        
        if tpot_list:
            results.update({
                "tpot_mean_ms": np.mean(tpot_list),
                "tpot_std_ms": np.std(tpot_list),
                "tpot_p95_ms": np.percentile(tpot_list, 95),
            })
        
        return results
    
    def benchmark_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict[str, float]:
        """Benchmark batch inference"""
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
        
        # Benchmark
        latencies = []
        
        for _ in range(self.benchmark_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            torch.cuda.synchronize()
            latency = time.time() - start
            latencies.append(latency)
        
        total_tokens = sum(
            output.shape[1] - inp.shape[1]
            for output, inp in zip(outputs, inputs.input_ids)
        )
        
        return {
            "batch_size": len(prompts),
            "latency_mean_s": np.mean(latencies),
            "latency_std_s": np.std(latencies),
            "throughput_tokens_per_s": total_tokens / np.mean(latencies),
            "throughput_seq_per_s": len(prompts) / np.mean(latencies),
        }


def compare_configurations(
    results: List[BenchmarkResult],
    baseline_name: str = "fp16",
) -> None:
    """
    Compare different configurations and print summary.
    """
    print("\n" + "="*80)
    print("Benchmark Comparison Summary")
    print("="*80)
    
    # Find baseline
    baseline = None
    for r in results:
        if r.name == baseline_name:
            baseline = r
            break
    
    if baseline is None:
        print(f"Baseline '{baseline_name}' not found!")
        return
    
    # Print header
    print(f"\n{'Configuration':<20} {'Speedup':<12} {'PPL Delta':<12} {'Memory':<12} {'Acceptable'}")
    print("-" * 80)
    
    for result in results:
        speedup = baseline.performance.throughput() / result.performance.throughput()
        ppl_delta = result.accuracy.perplexity_delta
        memory = result.memory.model_size_mb
        acceptable = "✓" if result.accuracy.is_acceptable() else "✗"
        
        print(f"{result.name:<20} {speedup:<12.2f}x {ppl_delta:<12.3f} {memory:<12.1f} {acceptable}")
    
    print("\n" + "="*80)
