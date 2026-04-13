"""
Windows UI Automation (UIA) 感知后端

通过 Windows 系统级的 Accessibility Tree 获取精确的控件信息，
包括控件类型、文本标签和屏幕坐标。

这是专业 GUI Agent 的核心技术路径之一，
被 Claude Computer Use、Playwright、Microsoft Power Automate 等系统广泛采用。
"""
import uiautomation as uia
from .base import PerceptionBackend
from ..schemas.models import UIElement


class UIABackend(PerceptionBackend):
    """
    基于 uiautomation 库的 Windows 控件树解析器。
    只遍历前台窗口（Foreground Window），递归深度可控。
    """

    # 我们关心的控件类型映射
    _TYPE_MAP = {
        "ButtonControl": "button",
        "EditControl": "input",
        "DocumentControl": "input",
        "TextControl": "text",
        "MenuItemControl": "menu_item",
        "ListItemControl": "list_item",
        "HyperlinkControl": "link",
        "ComboBoxControl": "select",
        "CheckBoxControl": "checkbox",
        "RadioButtonControl": "radio",
        "TabItemControl": "tab",
        "TreeItemControl": "tree_item",
        "ToolBarControl": "toolbar",
    }

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def parse(self, image) -> tuple[list[UIElement], str]:
        """
        解析当前前台窗口的控件树。
        image 参数在此后端中不被使用（UIA 不依赖截图做检测），
        但保留接口一致性。
        """
        root = uia.GetForegroundControl()
        elements: list[UIElement] = []
        self._traverse(root, depth=0, elements=elements)

        # 生成结构化描述文本，供 Planner 使用
        desc_lines = [
            f"Window: {root.Name} ({root.ControlTypeName})"
        ]
        for e in elements[:50]:  # 限制长度避免 prompt 爆炸
            desc_lines.append(
                f"  [{e.element_id}] {e.element_type}: '{e.text}' bbox={e.bbox}"
            )
        description = "\n".join(desc_lines)

        return elements, description

    def _traverse(self, control, depth: int, elements: list[UIElement]):
        if depth > self.max_depth:
            return

        rect = control.BoundingRectangle
        # 过滤掉不可见或没有尺寸的控件
        if rect and rect.width() > 2 and rect.height() > 2:
            ctype = control.ControlTypeName
            mapped_type = self._TYPE_MAP.get(ctype, ctype)
            elem_id = f"uia_{len(elements)}"
            elements.append(
                UIElement(
                    element_id=elem_id,
                    element_type=mapped_type,
                    text=control.Name or "",
                    bbox=[
                        int(rect.left),
                        int(rect.top),
                        int(rect.right),
                        int(rect.bottom),
                    ],
                    confidence=1.0,
                )
            )

        for child in control.GetChildren():
            self._traverse(child, depth + 1, elements)
