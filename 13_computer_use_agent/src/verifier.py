"""
视觉断言引擎：操作后的状态验证
"""
from .schemas.models import ScreenObservation, ActionPlan


class VisualVerifier:
    """
    提供多策略的屏幕状态验证：
    - pixel_diff: 像素级差异检测（快速，无需模型）
    - llm: 视觉模型判断（高准确，高延迟）
    """

    def __init__(self, backend: str = "pixel_diff"):
        self.backend = backend

    def verify(
        self,
        plan: ActionPlan,
        before: ScreenObservation,
        after: ScreenObservation,
        exec_meta: dict | None = None,
    ) -> tuple[bool, str]:
        """验证操作是否达到了预期效果"""
        if plan.action in ("noop", "terminate"):
            return True, "No verification needed for terminal/noop action"

        if self.backend == "pixel_diff":
            return self._pixel_diff(plan, before, after, exec_meta)

        if self.backend == "llm":
            return self._llm_judge(plan, before, after)

        return True, f"Verification skipped (backend={self.backend})"

    def _pixel_diff(
        self,
        plan: ActionPlan,
        before: ScreenObservation,
        after: ScreenObservation,
        exec_meta: dict | None,
    ) -> tuple[bool, str]:
        """
        基于降采样灰度图的像素差异检测。
        针对不同类型的动作使用自适应阈值：
        - click/scroll/hotkey: 需要较明显的屏幕变化 (>=2%)
        - type: 纯文本输入在白色背景上像素变化极小，使用宽松阈值 (>=0.3%)
        """
        if before.screenshot is None or after.screenshot is None:
            return False, "Missing screenshot for comparison"

        try:
            b = before.screenshot.convert("L").resize((320, 180))
            a = after.screenshot.convert("L").resize((320, 180))
            b_pixels = list(b.getdata() if hasattr(b, "getdata") else b.get_flattened_data())
            a_pixels = list(a.getdata() if hasattr(a, "getdata") else a.get_flattened_data())
            total = len(b_pixels)
            diff_pixels = sum(1 for p1, p2 in zip(b_pixels, a_pixels) if abs(p1 - p2) > 20)
            ratio = diff_pixels / total if total > 0 else 0.0

            # 自适应阈值
            if plan.action == "type":
                threshold = 0.003
            else:
                threshold = 0.02

            if ratio >= threshold:
                return True, f"Screen changed ({ratio:.2%} >= threshold {threshold:.2%})"
            else:
                return (
                    False,
                    f"Screen barely changed ({ratio:.2%} < threshold {threshold:.2%})",
                )
        except Exception as e:
            return False, f"Pixel diff error: {e}"

    def _llm_judge(
        self, plan: ActionPlan, before: ScreenObservation, after: ScreenObservation
    ) -> tuple[bool, str]:
        """TODO: 接入视觉模型做状态变化判断"""
        return True, "LLM judge not fully implemented yet"
