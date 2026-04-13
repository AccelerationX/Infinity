"""
测试 AtomicExecutor 的原子化操作逻辑
"""
import sys
from unittest.mock import MagicMock

# 在导入 executor 前 mock pyautogui，避免未安装时报错
mock_pyautogui = MagicMock()
mock_pyautogui.FAILSAFE = True
mock_pyautogui.PAUSE = 0.1
mock_pyautogui.moveTo = MagicMock()
mock_pyautogui.click = MagicMock()
mock_pyautogui.typewrite = MagicMock()
mock_pyautogui.press = MagicMock()
mock_pyautogui.scroll = MagicMock()
sys.modules["pyautogui"] = mock_pyautogui

from src.executor import AtomicExecutor
from src.schemas.models import ActionPlan, UIElement


def test_click_with_element():
    exe = AtomicExecutor()
    exe.update_elements([
        UIElement(element_id="btn_ok", element_type="button", text="OK", bbox=[100, 200, 200, 240]),
    ])
    plan = ActionPlan(action="click", target_element_id="btn_ok")
    result = exe.execute(plan)
    assert result["status"] == "clicked"
    assert result["target"] == "btn_ok"
    mock_pyautogui.moveTo.assert_called()
    mock_pyautogui.click.assert_called()


def test_click_without_element():
    exe = AtomicExecutor()
    exe.update_elements([])
    plan = ActionPlan(action="click", target_element_id="missing")
    result = exe.execute(plan)
    assert result["status"] == "error"


def test_type_and_hotkey():
    exe = AtomicExecutor()
    plan_type = ActionPlan(action="type", value="hello")
    assert exe.execute(plan_type)["status"] == "typed"
    mock_pyautogui.typewrite.assert_called_with("hello", interval=0.01)

    plan_hotkey = ActionPlan(action="hotkey", value="enter")
    assert exe.execute(plan_hotkey)["status"] == "hotkey"
    mock_pyautogui.press.assert_called_with("enter")


def test_terminal_actions():
    exe = AtomicExecutor()
    assert exe.execute(ActionPlan(action="noop"))["status"] == "noop"
    assert exe.execute(ActionPlan(action="terminate"))["status"] == "terminate"
