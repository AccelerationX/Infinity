"""
测试 VisualVerifier 的断言策略
"""
from PIL import Image
from src.verifier import VisualVerifier
from src.schemas.models import ScreenObservation, ActionPlan


def make_screenshot(color: tuple) -> Image.Image:
    return Image.new("RGB", (640, 480), color)


def test_pixel_diff_detects_change():
    v = VisualVerifier(backend="pixel_diff")
    before = ScreenObservation(screenshot=make_screenshot((0, 0, 0)))
    after = ScreenObservation(screenshot=make_screenshot((255, 255, 255)))
    passed, msg = v.verify(ActionPlan(action="click"), before, after)
    assert passed is True
    assert "Screen changed" in msg


def test_pixel_diff_detects_no_change():
    v = VisualVerifier(backend="pixel_diff")
    img = make_screenshot((128, 128, 128))
    before = ScreenObservation(screenshot=img)
    after = ScreenObservation(screenshot=img)
    passed, msg = v.verify(ActionPlan(action="click"), before, after)
    assert passed is False
    assert "barely changed" in msg


def test_noop_skips_verification():
    v = VisualVerifier(backend="pixel_diff")
    passed, msg = v.verify(ActionPlan(action="noop"), ScreenObservation(), ScreenObservation())
    assert passed is True
    assert "No verification needed" in msg
