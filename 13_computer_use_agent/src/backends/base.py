"""
视觉解析后端基类
"""
from abc import ABC, abstractmethod
from ..schemas.models import UIElement


class PerceptionBackend(ABC):
    """所有视觉解析后端的抽象基类"""

    @abstractmethod
    def parse(self, image) -> tuple[list[UIElement], str]:
        """
        解析截图，返回 (UIElement 列表, 原始描述文本)
        """
        raise NotImplementedError
