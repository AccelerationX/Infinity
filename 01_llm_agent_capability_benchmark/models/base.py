"""
模型接口基类
支持不同 LLM 的统一位装
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModel(ABC):
    """LLM 模型基类"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成模型响应
        
        Returns:
            {
                "content": str,  # 模型生成的文本
                "tool_calls": List[Dict],  # 工具调用（如果有）
                "usage": Dict,  # token 使用情况
                "latency": float,  # 延迟（秒）
            }
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用（API key 是否配置等）"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "config": self.config,
        }
