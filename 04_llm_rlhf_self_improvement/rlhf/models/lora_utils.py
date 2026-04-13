"""
LoRA工具函数
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


def load_model_with_lora(
    model_name: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
):
    """
    加载模型并应用LoRA
    
    Args:
        model_name: 模型名称或路径
        use_lora: 是否使用LoRA
        lora_r: LoRA秩
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目标模块列表
        torch_dtype: 数据类型
        device_map: 设备映射
        
    Returns:
        model: 加载的模型
        tokenizer: 分词器
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    # 应用LoRA
    if use_lora:
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer
