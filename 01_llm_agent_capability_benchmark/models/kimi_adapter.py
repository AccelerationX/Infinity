"""
Kimi (Moonshot) API 模型适配器
"""
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import BaseModel


class KimiModel(BaseModel):
    """
    Kimi K2.5 模型适配器
    
    API 信息:
    - 中国用户: https://api.moonshot.cn/v1
    - 国际用户: https://api.moonshot.ai/v1
    - Model: kimi-k2.5
    - 兼容 OpenAI API
    """
    
    def __init__(
        self, 
        model_name: str = "kimi-k2.5", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        # 优先使用中国版 API，如果用户需要国际版可以手动设置
        self.base_url = base_url or os.getenv("KIMI_BASE_URL") or "https://api.moonshot.cn/v1"
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate(
        self, 
        prompt: str, 
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Kimi client not initialized. Check API key.")
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        # Kimi K2.5 只支持 temperature=1
        kim_temperature = 1.0
        
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=kim_temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kim_temperature,
                    max_tokens=max_tokens,
                )
            
            latency = time.time() - start_time
            
            result = {
                "content": response.choices[0].message.content or "",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency": latency,
            }
            
            # 提取工具调用
            if response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
            
            return result
            
        except Exception as e:
            return {
                "content": "",
                "tool_calls": [],
                "usage": {},
                "latency": time.time() - start_time,
                "error": str(e),
            }
