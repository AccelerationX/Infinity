"""
Mock 视觉解析后端：用于框架测试和快速原型验证
"""
from .base import PerceptionBackend
from ..schemas.models import UIElement


class MockBackend(PerceptionBackend):
    """
    返回预设的 UI 元素，不依赖任何外部模型或 API。
    用于验证 Agent 循环、Executor 和 Verifier 的正确性。
    """

    def parse(self, image) -> tuple[list[UIElement], str]:
        elements = [
            UIElement(
                element_id="btn_ok",
                element_type="button",
                text="OK",
                bbox=[100, 200, 200, 240],
                confidence=0.99,
            ),
            UIElement(
                element_id="input_search",
                element_type="input",
                text="",
                bbox=[100, 260, 400, 290],
                confidence=0.95,
            ),
            UIElement(
                element_id="link_help",
                element_type="link",
                text="Help",
                bbox=[500, 100, 560, 120],
                confidence=0.90,
            ),
        ]
        return elements, "A mock screen with OK button, search input, and help link."
