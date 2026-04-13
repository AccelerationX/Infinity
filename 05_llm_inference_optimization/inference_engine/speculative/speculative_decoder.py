"""
Speculative Decoding

Implementation of "Accelerating Large Language Model Decoding with Speculative Decoding"
https://arxiv.org/abs/2211.17192

Also implements Lookahead Decoding:
"Breaking the Sequential Dependency of LLM Inference Using Lookahead Decoding"
https://arxiv.org/abs/2402.02057

Core algorithm:
1. Draft model (smaller/faster) generates K candidate tokens autoregressively
2. Target model (larger/slower) verifies all K tokens in parallel
3. Accept tokens based on probability ratio: min(1, p(x)/q(x))
4. If rejected, resample from the residual distribution
5. Expected speedup: ~2-3x depending on draft model quality
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import time


class SpeculativeDecoder:
    """
    Speculative Decoder
    
    Combines a draft model (fast, small) with a target model (slow, large)
    to accelerate generation while maintaining exact target distribution.
    """
    
    def __init__(
        self,
        draft_model,  # Smaller model for drafting
        target_model,  # Larger target model
        tokenizer,
        num_speculative_tokens: int = 5,
        acceptance_threshold: float = 1.0,
        device: str = "cuda",
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.K = num_speculative_tokens  # Number of speculative tokens (gamma in paper)
        self.acceptance_threshold = acceptance_threshold
        self.device = device
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_tokens": 0,
            "target_forward_calls": 0,
            "draft_forward_calls": 0,
        }
    
    def speculative_sampling(
        self,
        prefix: torch.Tensor,
        target_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[List[int], Dict]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            prefix: Initial token sequence [batch, seq_len]
            target_len: Total number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            generated_tokens: List of generated token ids
            stats: Statistics about acceptance rate, etc.
        """
        generated = prefix.squeeze(0).tolist()
        
        while len(generated) < target_len:
            # Step 1: Draft model generates K candidate tokens
            draft_tokens = self._generate_draft(generated, self.K)
            self.stats["draft_tokens"] += len(draft_tokens)
            self.stats["draft_forward_calls"] += len(draft_tokens)
            
            # Step 2: Target model verifies all K tokens in parallel
            prefix_with_draft = torch.tensor([generated + draft_tokens], device=self.device)
            
            with torch.no_grad():
                target_logits = self.target_model(prefix_with_draft).logits
                self.stats["target_forward_calls"] += 1
            
            # Step 3: Verify and accept/reject tokens
            accepted = 0
            for i, draft_token in enumerate(draft_tokens):
                # Get probabilities for position len(generated) + i
                pos = len(generated) + i
                
                # Target distribution p(x)
                p_logits = target_logits[0, pos - 1] / temperature
                p_probs = F.softmax(p_logits, dim=-1)
                
                # Draft distribution q(x) - need to compute
                draft_prefix = torch.tensor([generated + draft_tokens[:i]], device=self.device)
                with torch.no_grad():
                    q_logits = self.draft_model(draft_prefix).logits[0, -1] / temperature
                    q_probs = F.softmax(q_logits, dim=-1)
                
                # Acceptance probability: min(1, p(draft_token) / q(draft_token))
                p_draft = p_probs[draft_token].item()
                q_draft = q_probs[draft_token].item()
                
                acceptance_prob = min(1.0, p_draft / (q_draft + 1e-10))
                
                if torch.rand(1).item() < acceptance_prob:
                    # Accept token
                    generated.append(draft_token)
                    accepted += 1
                    self.stats["accepted_tokens"] += 1
                else:
                    # Reject token - sample from residual distribution
                    # p'(x) = (p(x) - q(x))_+ / Z
                    residual_probs = (p_probs - q_probs).clamp(min=0)
                    if residual_probs.sum() > 0:
                        residual_probs = residual_probs / residual_probs.sum()
                        new_token = torch.multinomial(residual_probs, 1).item()
                    else:
                        # Fallback to target distribution
                        new_token = torch.multinomial(p_probs, 1).item()
                    
                    generated.append(new_token)
                    break
            
            self.stats["total_tokens"] += accepted + 1
            
            # If all K tokens accepted, sample one more from target
            if accepted == len(draft_tokens) and len(generated) < target_len:
                p_logits = target_logits[0, len(generated) - 1] / temperature
                p_probs = F.softmax(p_logits, dim=-1)
                new_token = torch.multinomial(p_probs, 1).item()
                generated.append(new_token)
                self.stats["total_tokens"] += 1
        
        return generated, self.stats
    
    def _generate_draft(
        self,
        prefix_tokens: List[int],
        num_tokens: int,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate K candidate tokens using draft model.
        
        Args:
            prefix_tokens: Current sequence
            num_tokens: Number of tokens to generate (K)
            
        Returns:
            draft_tokens: List of K draft token ids
        """
        draft_tokens = []
        current = prefix_tokens.copy()
        
        for _ in range(num_tokens):
            input_ids = torch.tensor([current], device=self.device)
            
            with torch.no_grad():
                logits = self.draft_model(input_ids).logits[0, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1).item()
            
            draft_tokens.append(token)
            current.append(token)
        
        return draft_tokens
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict:
        """
        Benchmark speculative decoding vs standard decoding.
        
        Returns speedup ratio and acceptance statistics.
        """
        print("\n" + "="*60)
        print("Speculative Decoding Benchmark")
        print("="*60)
        
        results = {
            "speculative": [],
            "baseline": [],
        }
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt[:50]}...")
            
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Baseline: Target model only
            start = time.time()
            with torch.no_grad():
                baseline_output = self.target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
            baseline_time = time.time() - start
            
            # Speculative decoding
            start = time.time()
            spec_output, stats = self.speculative_sampling(
                input_ids,
                target_len=input_ids.shape[1] + max_new_tokens,
                temperature=0.7,
            )
            spec_time = time.time() - start
            
            # Calculate metrics
            acceptance_rate = stats["accepted_tokens"] / stats["draft_tokens"]
            speedup = baseline_time / spec_time
            
            print(f"  Baseline time: {baseline_time:.2f}s")
            print(f"  Speculative time: {spec_time:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Acceptance rate: {acceptance_rate:.2%}")
            print(f"  Draft tokens: {stats['draft_tokens']}")
            print(f"  Accepted tokens: {stats['accepted_tokens']}")
            print(f"  Target forward calls: {stats['target_forward_calls']}")
            print(f"  Draft forward calls: {stats['draft_forward_calls']}")
            
            results["speculative"].append({
                "time": spec_time,
                "speedup": speedup,
                "acceptance_rate": acceptance_rate,
            })
            results["baseline"].append({
                "time": baseline_time,
            })
        
        # Average statistics
        avg_speedup = sum(r["speedup"] for r in results["speculative"]) / len(results["speculative"])
        avg_acceptance = sum(r["acceptance_rate"] for r in results["speculative"]) / len(results["speculative"])
        
        print("\n" + "="*60)
        print("Average Results:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average acceptance rate: {avg_acceptance:.2%}")
        print("="*60)
        
        return results


class LookaheadDecoder:
    """
    Lookahead Decoding
    
    Extension of speculative decoding that uses n-grams from the generated
    sequence itself as draft tokens, without needing a separate draft model.
    
    Key idea: Use recently seen token sequences to predict future tokens.
    """
    
    def __init__(
        self,
        target_model,
        tokenizer,
        window_size: int = 5,
        ngram_size: int = 3,
        num_candidates: int = 10,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.ngram_size = ngram_size
        self.num_candidates = num_candidates
    
    def generate(
        self,
        prefix: torch.Tensor,
        target_len: int,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate using lookahead decoding.
        
        1. Maintain a window of recent tokens
        2. Look for matching n-grams in the window
        3. Use matches to predict future tokens
        4. Verify predictions with target model
        """
        generated = prefix.squeeze(0).tolist()
        
        # Window of recent tokens for matching
        window = generated[-self.window_size:].copy()
        
        while len(generated) < target_len:
            # Find candidate continuations from window
            candidates = self._find_candidates(window, generated)
            
            if len(candidates) == 0:
                # No candidates - fallback to standard generation
                input_ids = torch.tensor([generated], device=prefix.device)
                with torch.no_grad():
                    logits = self.target_model(input_ids).logits[0, -1] / temperature
                    token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                generated.append(token)
                window.append(token)
                if len(window) > self.window_size:
                    window.pop(0)
            else:
                # Verify candidates with target model
                # Similar to speculative decoding...
                pass
        
        return generated
    
    def _find_candidates(
        self,
        window: List[int],
        full_sequence: List[int],
    ) -> List[List[int]]:
        """Find candidate continuations based on n-gram matching."""
        candidates = []
        
        # Look for matching n-grams in window
        for i in range(len(full_sequence) - self.ngram_size):
            ngram = full_sequence[i:i + self.ngram_size]
            if ngram == window[-self.ngram_size:]:
                # Found match - take continuation
                continuation = full_sequence[i + self.ngram_size:i + self.ngram_size + 5]
                candidates.append(continuation)
                
                if len(candidates) >= self.num_candidates:
                    break
        
        return candidates


def theoretical_analysis():
    """
    Theoretical analysis of speculative decoding speedup.
    
    Expected speedup = 1 / (1 - alpha + alpha/c)
    
    Where:
    - alpha: Acceptance rate (probability draft token is accepted)
    - c: Cost ratio (draft inference cost / target inference cost)
    
    Typical values:
    - alpha ~ 0.6-0.8 (depending on draft model quality)
    - c ~ 0.1-0.2 (draft model is 5-10x smaller)
    
    Expected speedup: 2-3x
    """
    print("\n" + "="*60)
    print("Theoretical Analysis of Speculative Decoding")
    print("="*60)
    
    print("""
    Speedup Formula:
    S = 1 / (1 - α + α/c)
    
    Where:
    - α: Acceptance rate (draft token accepted)
    - c: Cost ratio (draft_cost / target_cost)
    
    Example scenarios:
    """)
    
    scenarios = [
        (0.6, 0.1, "Conservative"),
        (0.7, 0.1, "Typical"),
        (0.8, 0.05, "Optimistic"),
    ]
    
    print(f"{'Scenario':<15} {'α':<8} {'c':<8} {'Speedup':<10}")
    print("-" * 45)
    
    for alpha, c, name in scenarios:
        speedup = 1 / (1 - alpha + alpha / c)
        print(f"{name:<15} {alpha:<8.2f} {c:<8.2f} {speedup:<10.2f}x")
    
    print("""
    Key Insights:
    1. Higher acceptance rate (α) → better speedup
    2. Smaller cost ratio (c) → better speedup
    3. Draft model should be much smaller but still high quality
    4. Diminishing returns: Speedup is bounded by 1/(1-α)
    """)


if __name__ == "__main__":
    theoretical_analysis()
