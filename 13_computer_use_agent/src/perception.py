"""
视觉感知引擎：截图 + UI 解析的统一入口
"""
from PIL import Image
from .schemas.models import ScreenObservation


def capture_screen() -> Image.Image:
    """
    截取当前屏幕。需要在有图形界面的环境中运行。
    如果 pyautogui 未安装，会自动抛出 ImportError。
    """
    try:
        import pyautogui
    except ImportError as e:
        raise RuntimeError(
            "pyautogui is required for screen capture. "
            "Install: pip install pyautogui"
        ) from e
    return pyautogui.screenshot()


class PerceptionEngine:
    """
    视觉感知引擎，支持多种后端：
    - mock: 用于测试，返回预设元素
    - qwen_vl: 本地 Qwen2-VL 模型解析
    - openai_vision: OpenAI GPT-4V/GPT-4o API 解析
    """

    def __init__(self, backend_name: str = "mock"):
        self.backend_name = backend_name
        if backend_name == "qwen_vl":
            from .backends.qwen_vl import QwenVLBackend
            self.backend = QwenVLBackend()
        elif backend_name == "openai_vision":
            from .backends.openai_vision import OpenAIVisionBackend
            self.backend = OpenAIVisionBackend()
        elif backend_name == "uia":
            from .backends.uia_backend import UIABackend
            self.backend = UIABackend()
        elif backend_name == "mock":
            from .backends.mock import MockBackend
            self.backend = MockBackend()
        else:
            raise ValueError(f"Unknown perception backend: {backend_name}")

    def observe(self) -> ScreenObservation:
        """截取当前屏幕并用选定的后端解析 UI 元素"""
        img = capture_screen()
        elements, description = self.backend.parse(img)
        return ScreenObservation(
            screenshot=img,
            elements=elements,
            raw_description=description,
        )
