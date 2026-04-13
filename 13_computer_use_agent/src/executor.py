"""
原子化执行器：将 ActionPlan 转换为精确的 GUI 操作
"""
import time
from .schemas.models import ActionPlan, UIElement


class AtomicExecutor:
    """
    所有 GUI 操作都经过“原子化”封装：
    1. 定位（hover）并做短暂验证
    2. 执行核心动作
    3. 等待 UI 响应
    这样可以显著降低误触率。
    """

    def __init__(self):
        self.element_map: dict[str, UIElement] = {}
        try:
            import pyautogui
            self.pyautogui = pyautogui
            self.pyautogui.FAILSAFE = True
            self.pyautogui.PAUSE = 0.1
        except ImportError as e:
            raise RuntimeError(
                "pyautogui is required for execution. Install: pip install pyautogui"
            ) from e

    def update_elements(self, elements: list[UIElement]):
        """更新当前屏幕的元素映射表"""
        self.element_map = {e.element_id: e for e in elements}

    def execute(self, plan: ActionPlan) -> dict:
        """执行单个 ActionPlan，返回执行元数据"""
        action = plan.action
        if action == "click":
            return self._atomic_click(plan.target_element_id)
        elif action == "type":
            return self._atomic_type(plan.value)
        elif action == "hotkey":
            return self._atomic_hotkey(plan.value)
        elif action == "scroll":
            amount = int(plan.value) if plan.value else -300
            self.pyautogui.scroll(amount)
            return {"status": "scrolled", "amount": amount}
        elif action == "wait":
            seconds = float(plan.value) if plan.value else 1.0
            time.sleep(seconds)
            return {"status": "waited", "seconds": seconds}
        elif action in ("terminate", "noop"):
            return {"status": action}
        return {"status": "unknown_action", "error": f"Unsupported action: {action}"}

    def _atomic_click(self, element_id: str | None) -> dict:
        """原子化点击：移动 -> 停顿 -> 点击 -> 停顿"""
        if not element_id or element_id not in self.element_map:
            return {"status": "error", "error": f"Element '{element_id}' not found in current screen"}

        elem = self.element_map[element_id]
        x = (elem.bbox[0] + elem.bbox[2]) // 2
        y = (elem.bbox[1] + elem.bbox[3]) // 2

        self.pyautogui.moveTo(x, y, duration=0.2)
        time.sleep(0.1)
        self.pyautogui.click()
        time.sleep(0.2)
        return {"status": "clicked", "target": element_id, "coordinates": [x, y]}

    def _atomic_type(self, text: str) -> dict:
        """在当前焦点处输入文本"""
        self.pyautogui.typewrite(text, interval=0.01)
        return {"status": "typed", "text": text}

    def _atomic_hotkey(self, key: str) -> dict:
        """按下指定按键（如 enter, esc, win）"""
        self.pyautogui.press(key)
        return {"status": "hotkey", "key": key}
