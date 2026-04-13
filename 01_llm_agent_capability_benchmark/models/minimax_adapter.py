"""
MiniMax API 模型适配器 (Token Plan 版本)
"""
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import BaseModel


class MiniMaxModel(BaseModel):
    """
    MiniMax M2.7 Token Plan 适配器
    
    API 信息:
    - Token Plan 端点: https://www.minimaxi.com/v1
    - Model: MiniMax-M2.7 / MiniMax-M2.7-highspeed
    - Token Plan 与普通 API Key 不互通
    """
    
    def __init__(
        self, 
        model_name: str = "MiniMax-M2.7", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        # Token Plan 使用 www.minimaxi.com 域名
        self.base_url = base_url or os.getenv("MINIMAX_BASE_URL") or "https://www.minimaxi.com/v1"
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
        temperature: float = 1.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("MiniMax client not initialized. Check API key.")
        
        start_time = time.time()
        
        # Token Plan 可能不支持 tools，将 tools 信息加入 prompt
        if tools:
            tool_desc = "\n\n你可以使用以下工具：\n"
            for tool in tools:
                tool_desc += f"- {tool['function']['name']}: {tool['function']['description']}\n"
            tool_desc += "\n如果需要使用工具，请按以下格式回复：\nTOOL_CALL: {\"name\": \"工具名\", \"arguments\": {...}}"
            prompt = prompt + tool_desc
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
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
            
            return result
            
        except Exception as e:
            return {
                "content": "",
                "tool_calls": [],
                "usage": {},
                "latency": time.time() - start_time,
                "error": str(e),
            }
