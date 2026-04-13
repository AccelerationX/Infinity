"""
OpenAI API 模型适配器
"""
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI GPT 系列模型适配器"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None
    
    def generate(
        self, 
        prompt: str, 
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("OpenAI client not initialized. Check API key.")
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
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
