"""
任务评估器：在执行完成后，用独立于 Agent 的手段验证任务目标是否达成
"""
import time


class TaskEvaluator:
    """
    使用剪贴板、UIA、OCR 等外部手段检查任务结果，
    避免 Agent 自说自话"已完成"的幻觉问题。
    """

    def __init__(self):
        try:
            import pyautogui
            self.pyautogui = pyautogui
        except ImportError as e:
            raise RuntimeError("pyautogui required for TaskEvaluator") from e

    def check_notepad_contains(self, expected_text: str) -> tuple[bool, str]:
        """
        通过 UIA 直接读取前台记事本窗口中的文本控件内容。
        比剪贴板方法更稳定，不受窗口焦点漂移影响。
        """
        try:
            import uiautomation as uia
        except ImportError:
            return False, "uiautomation not installed"

        root = uia.GetForegroundControl()

        def _find_text(control) -> str | None:
            ctype = control.ControlTypeName
            if ctype in ("EditControl", "DocumentControl"):
                # 方法 1: ValuePattern
                try:
                    val = control.GetValuePattern()
                    if val and val.Value:
                        return val.Value
                except Exception:
                    pass
                # 方法 2: TextPattern
                try:
                    tp = control.GetTextPattern()
                    if tp:
                        return tp.DocumentRange.GetText(-1)
                except Exception:
                    pass
                # 方法 3: LegacyIAccessiblePattern
                try:
                    la = control.GetLegacyIAccessiblePattern()
                    if la and la.Value:
                        return la.Value
                except Exception:
                    pass

            for child in control.GetChildren():
                result = _find_text(child)
                if result is not None:
                    return result
            return None

        actual = _find_text(root) or ""
        # 某些 UIA 后端会吞掉空格，做模糊匹配
        if expected_text in actual or expected_text.replace(" ", "") in actual.replace(" ", ""):
            return True, "Found expected text in notepad"
        else:
            return False, f"Expected '{expected_text}', got '{actual[:200]}'"

    def check_active_window_title(self, expected_substring: str) -> tuple[bool, str]:
        """检查当前前台窗口标题是否包含指定字符串"""
        try:
            import pygetwindow as gw
            title = gw.getActiveWindow().title
            if expected_substring in title:
                return True, f"Active window title matches: {title}"
            return False, f"Active window title: {title}"
        except Exception as e:
            return False, f"Error checking window title: {e}"
