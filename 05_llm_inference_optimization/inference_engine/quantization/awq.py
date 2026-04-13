"""
AWQ (Activation-aware Weight Quantization)

Implementation of "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
https://arxiv.org/abs/2306.00978

Core insight:
Not all weights are equally important. Weights corresponding to large activation
magnitudes are more important and should be quantized with higher precision.

Key algorithm:
1. Observe activation distribution from calibration data
2. Identify "salient" weight channels (high activation magnitude)
3. Apply per-channel scaling to protect salient channels
4. Quantize with group-wise symmetric quantization
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class AWQQuantizer:
    """
    AWQ Quantizer
    
    Protects salient weight channels by scaling them before quantization.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        search_scale: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.search_scale = search_scale
        
        # Quantization range
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -(2 ** (bits - 1))
    
    def find_salient_channels(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify salient weight channels based on activation magnitude.
        
        Args:
            weight: Weight matrix [out_features, in_features]
            activation: Activation from calibration data [batch, seq_len, in_features]
            
        Returns:
            channel_importance: Importance score for each input channel
        """
        # Compute activation magnitude per channel
        # Average over batch and sequence dimensions
        act_magnitude = activation.abs().mean(dim=(0, 1))  # [in_features]
        
        return act_magnitude
    
    def search_best_scale(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
        num_grids: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for optimal scaling factors to protect salient channels.
        
        The scaling factor s is applied as: w' = w * s
        After quantization: w_q = round(w' / scale) * scale / s
        
        This effectively gives higher precision to channels with larger s.
        
        Args:
            weight: Weight matrix
            activation: Calibration activations
            num_grids: Number of grid points for search
            
        Returns:
            best_scale: Optimal scaling factors per channel
            best_clip_scale: Optimal clipping scale
        """
        # Find salient channels
        channel_importance = self.find_salient_channels(weight, activation)
        
        # Grid search for scaling factors
        # Typical range: [0.5, 1.5]
        scale_candidates = torch.linspace(0.5, 1.5, num_grids)
        
        best_loss = float('inf')
        best_scale = torch.ones(weight.shape[1], device=weight.device)
        
        # Simplified: Use activation magnitude as scaling
        # In full AWQ, we do grid search with real loss evaluation
        if self.search_scale:
            # Normalize importance to [0.8, 1.2] range
            importance_norm = channel_importance / channel_importance.mean()
            best_scale = 1.0 + 0.2 * (importance_norm - importance_norm.mean())
            best_scale = best_scale.clamp(0.5, 1.5)
        
        return best_scale
    
    def quantize_layer(
        self,
        weight: torch.Tensor,
        activation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize a single layer with AWQ.
        
        Args:
            weight: Weight matrix [out_features, in_features]
            activation: Calibration activations (for finding salient channels)
            
        Returns:
            Dictionary with quantized weights and metadata
        """
        weight = weight.float()
        orig_shape = weight.shape
        
        # Search for optimal scaling
        if activation is not None and self.search_scale:
            scales = self.search_best_scale(weight, activation)
            # Apply scaling to weights
            weight_scaled = weight * scales.unsqueeze(0)
        else:
            scales = torch.ones(weight.shape[1], device=weight.device)
            weight_scaled = weight
        
        # Reshape for group-wise quantization
        if self.group_size > 0:
            assert weight.shape[1] % self.group_size == 0
            weight_scaled = weight_scaled.reshape(-1, self.group_size)
        
        # Compute quantization parameters
        w_min = weight_scaled.min(dim=1, keepdim=True)[0]
        w_max = weight_scaled.max(dim=1, keepdim=True)[0]
        
        if self.zero_point:
            scale = (w_max - w_min) / (self.qmax - self.qmin)
            zero = self.qmin - w_min / scale
        else:
            # Symmetric quantization
            abs_max = torch.max(w_max.abs(), w_min.abs())
            scale = abs_max / self.qmax
            zero = torch.zeros_like(scale)
        
        # Avoid division by zero
        scale = scale.clamp(min=1e-5)
        
        # Quantize
        w_int = torch.round(weight_scaled / scale + zero).clamp(self.qmin, self.qmax)
        
        # Dequantize (for validation)
        weight_dq = (w_int - zero) * scale
        
        # Reshape back
        weight_dq = weight_dq.reshape(orig_shape)
        
        # Compute quantization error
        error = (weight - weight_dq / scales.unsqueeze(0)).abs().mean()
        
        return {
            "qweight": w_int.reshape(orig_shape).to(torch.int8),
            "scales": scale.reshape(-1),
            "zeros": zero.reshape(-1),
            "activation_scales": scales,
            "error": error.item(),
        }
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        device: str = "cuda",
    ) -> nn.Module:
        """
        Quantize entire model with AWQ.
        
        Args:
            model: Model to quantize
            calibration_data: List of calibration inputs
            device: Device to use
            
        Returns:
            Quantized model
        """
        print(f"Starting AWQ quantization...")
        print(f"  Bits: {self.bits}")
        print(f"  Group size: {self.group_size}")
        print(f"  Zero point: {self.zero_point}")
        
        # Collect activations
        print("  Collecting activations...")
        activations = {}
        
        def register_hook(name):
            def hook(module, input, output):
                if isinstance(input[0], torch.Tensor):
                    activations[name] = input[0].detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(register_hook(name))
                hooks.append(hook)
        
        # Run calibration
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                _ = model(batch.to(device))
                break  # Just one batch for now
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Quantize each layer
        quantized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                print(f"  Quantizing: {name}")
                
                weight = module.weight.data
                activation = activations.get(name)
                
                if activation is not None:
                    activation = activation.to(device)
                
                # Quantize
                quant_params = self.quantize_layer(weight, activation)
                
                # Replace weight with dequantized version
                # In real implementation, we would use custom CUDA kernels
                module.weight.data = quant_params["qweight"].float()
                
                # Store quantization params
                module.register_buffer("awq_scales", quant_params["scales"])
                module.register_buffer("awq_zeros", quant_params["zeros"])
                
                quantized_count += 1
        
        print(f"AWQ quantization complete! Quantized {quantized_count} layers.")
        return model


class AWQLinear(nn.Module):
    """
    AWQ Quantized Linear Layer
    
    Efficient inference with fused scaling and dequantization.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Quantized weights (int8)
        num_groups = in_features // group_size
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, num_groups, group_size, dtype=torch.int8)
        )
        self.register_buffer("scales", torch.zeros(out_features, num_groups))
        self.register_buffer("zeros", torch.zeros(out_features, num_groups))
        
        # AWQ activation scales
        self.register_buffer(
            "act_scales",
            torch.ones(in_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused scaling.
        
        Steps:
        1. Apply activation scaling to input
        2. Dequantize weights
        3. Compute linear output
        4. Apply inverse scaling
        """
        # Apply activation scaling
        x_scaled = x * self.act_scales.unsqueeze(0).unsqueeze(0)
        
        # Dequantize weights
        weight = (self.qweight.float() - self.zeros.unsqueeze(-1)) * self.scales.unsqueeze(-1)
        weight = weight.reshape(self.out_features, self.in_features)
        
        # Linear computation
        out = torch.nn.functional.linear(x_scaled, weight)
        
        # Apply inverse activation scaling
        # This is fused into the computation in optimized CUDA kernels
        out = out / self.act_scales.mean()
        
        return out


def compare_awq_vs_gptq():
    """
    Compare AWQ vs GPTQ quantization
    
    Key differences:
    - AWQ considers activation magnitudes (salient channels)
    - GPTQ uses second-order information (Hessian)
    - AWQ typically better for low-bit (< 4-bit) quantization
    """
    print("\n" + "="*60)
    print("AWQ vs GPTQ Comparison")
    print("="*60)
    
    comparison = """
    Aspect          | GPTQ                          | AWQ
    ----------------|-------------------------------|-------------------------------
    Core Idea       | Hessian-based error comp.     | Activation-aware scaling
    Salient Weights | Treated equally               | Higher precision for important
    Calibration     | Need Hessian (more data)      | Need activations (less data)
    Speed           | Slower (Cholesky)             | Faster (simple scaling)
    Accuracy        | Good for 4-bit                | Better for < 4-bit
    Hardware        | General                       | Optimized for specific kernels
    
    Key Insight:
    - AWQ works because activations have outliers (large magnitude values)
    - Weights corresponding to these outlier channels are more important
    - By scaling these channels, we reduce their quantization error
    """
    
    print(comparison)


if __name__ == "__main__":
    compare_awq_vs_gptq()
