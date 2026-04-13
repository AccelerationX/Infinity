"""
MACP 配置文件
集中管理API密钥和模型配置
"""
import os
from typing import Dict, Optional

# ============================================
# API 配置
# ============================================

# Kimi K2.5 配置
KIMI_API_KEY = "sk-0peMQC6Qvk3I7zpOeJ9VcXxAOzV8tpdMBNyyVTvGpNRSiWhG"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = "moonshot-v1-8k"  # 使用兼容OpenAI格式的模型名称

# DeepSeek V3.2 配置
DEEPSEEK_API_KEY = "sk-e523a87483174aa79a900f2c46e1a3cf"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# MiniMax (Token Plan API)
# 注意：这是Token Plan API，使用简单HTTP调用而非OpenAI格式
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://www.minimaxi.com/v1"


# ============================================
# 全局配置
# ============================================

class MACPConfig:
    """MACP全局配置"""
    
    # 默认模型
    DEFAULT_MODEL = "kimi"
    
    # 模型配置
    MODELS: Dict[str, Dict] = {
        "kimi": {
            "api_key": KIMI_API_KEY,
            "base_url": KIMI_BASE_URL,
            "model": KIMI_MODEL,
            "temperature": 1.0,  # Kimi要求固定1.0
            "max_tokens": 4096,
        },
        "deepseek": {
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    }
    
    # Agent配置
    AGENT_CONFIG = {
        "max_retries": 3,
        "retry_delay": 2,  # seconds
        "timeout": 60,
    }
    
    # 调度器配置
    SCHEDULER_CONFIG = {
        "max_workers": 3,
        "task_timeout": 300,  # 5 minutes
        "checkpoint_interval": 30,  # seconds
    }
    
    # Web服务器配置
    WEB_CONFIG = {
        "host": "0.0.0.0",
        "port": 8080,
        "debug": False,
    }


def get_model_config(model_name: str) -> Optional[Dict]:
    """获取模型配置"""
    return MACPConfig.MODELS.get(model_name)


def configure_models():
    """
    配置模型环境变量
    在程序启动时调用
    """
    # 设置OpenAI SDK需要的环境变量
    os.environ["OPENAI_API_KEY"] = KIMI_API_KEY
    os.environ["OPENAI_BASE_URL"] = KIMI_BASE_URL
    
    print("模型配置完成:")
    print(f"  - Kimi K2.5: {KIMI_BASE_URL}")
    print(f"  - DeepSeek V3.2: {DEEPSEEK_BASE_URL}")
