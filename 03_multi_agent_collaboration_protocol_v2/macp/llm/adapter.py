"""
LLM适配器 - 统一调用不同模型
支持：Kimi K2.5, DeepSeek V3.2
"""
import json
import time
from typing import Dict, List, Optional, Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

from ..config import get_model_config, MACPConfig


class LLMAdapter:
    """
    LLM适配器
    
    统一接口调用不同模型API
    """
    
    def __init__(self, model_name: str = "kimi"):
        """
        初始化适配器
        
        Args:
            model_name: 模型名称 (kimi/deepseek)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required")
        
        self.model_name = model_name
        self.config = get_model_config(model_name)
        
        if not self.config:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )
        
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
    
    def chat(self,
             system_prompt: Optional[str] = None,
             user_prompt: str = "",
             tools: Optional[List[Dict]] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        对话请求
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            tools: 工具定义列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            {
                "content": str,
                "tool_calls": List[Dict],
                "usage": Dict,
                "model": str,
            }
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        # 构建请求参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Kimi要求temperature必须固定为1.0
        if self.model_name == "kimi":
            params["temperature"] = 1.0
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        # 重试机制
        max_retries = MACPConfig.AGENT_CONFIG["max_retries"]
        retry_delay = MACPConfig.AGENT_CONFIG["retry_delay"]
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                
                result = {
                    "content": response.choices[0].message.content or "",
                    "model": self.model,
                }
                
                # 提取工具调用
                message = response.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments),
                            },
                            "type": tc.type,
                        }
                        for tc in message.tool_calls
                    ]
                
                # 提取token使用
                if hasattr(response, "usage"):
                    result["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                continue
        
        # 所有重试失败
        raise last_error
    
    def stream_chat(self,
                    system_prompt: Optional[str] = None,
                    user_prompt: str = "",
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None):
        """
        流式对话（用于长文本生成）
        
        Yields:
            文本片段
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        
        if self.model_name == "kimi":
            params["temperature"] = 1.0
        
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
