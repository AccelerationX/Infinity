"""
ReAct Planner：基于 LLM 的观察-思考-行动规划器
"""
import json
from .schemas.models import ActionPlan, ScreenObservation


class ReActPlanner:
    """
    接收当前屏幕观察和历史记录，生成下一步最优行动。
    支持接入任意 OpenAI-compatible LLM。
    """

    def __init__(self, llm_client=None):
        """
        llm_client: callable(prompt: str) -> response: str
        如果为 None，则使用基于规则的 fallback planner。
        """
        self.llm = llm_client

    def plan(self, task: str, observation: ScreenObservation, history: list[dict]) -> ActionPlan:
        if self.llm is None:
            return self._fallback_plan(task, observation, history)
        return self._llm_plan(task, observation, history)

    def _llm_plan(self, task: str, observation: ScreenObservation, history: list[dict]) -> ActionPlan:
        elements_desc = "\n".join(
            [
                f"- [{e.element_id}] {e.element_type}: '{e.text}' at bbox={e.bbox}"
                for e in observation.elements
            ]
        ) or "No interactive elements detected."

        history_text = json.dumps(history, ensure_ascii=False, indent=2) if history else "None"

        prompt = f"""You are a GUI automation assistant. Given the task and current screen state, output the SINGLE next action in strict JSON format.

Task: {task}

Current UI Elements:
{elements_desc}

History of previous actions:
{history_text}

Respond ONLY with a JSON object containing:
- thought: string (your reasoning)
- action: one of [click, type, hotkey, scroll, wait, terminate, noop]
- target_element_id: string or null (the element_id to interact with)
- value: string (text to type, key to press, or scroll amount)
- expected_outcome: string (what you expect to see after this action)

JSON:"""

        raw = self.llm(prompt).strip()
        # 提取 JSON 块
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # LLM 输出非 JSON，fallback 到 noop
            return ActionPlan(
                thought=f"Failed to parse LLM output: {raw[:200]}",
                action="noop",
            )

        return ActionPlan(
            thought=data.get("thought", ""),
            action=data.get("action", "noop"),
            target_element_id=data.get("target_element_id"),
            value=data.get("value", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )

    def _fallback_plan(self, task: str, observation: ScreenObservation, history: list[dict]) -> ActionPlan:
        """
        无 LLM 时的智能规则后备逻辑。
        支持从历史记录中识别任务是否已完成，避免重复执行。
        """
        import re

        # 如果上一步已经成功完成且验证通过，则终止
        if history:
            last = history[-1]
            if last.get("verification_passed") and last.get("plan", {}).get("action") not in ("noop", "terminate"):
                return ActionPlan(
                    thought="Fallback: previous action succeeded, task appears complete",
                    action="terminate",
                    expected_outcome="Task finished",
                )

        if not observation.elements:
            return ActionPlan(action="noop")

        # 提取引号内的文本（中文/英文引号）
        quoted = re.search(r"['\"'\"'](.+?)['\"'\"']", task)
        extracted_text = quoted.group(1) if quoted else "hello world"

        # 1. 打开应用
        if "打开" in task or "open" in task.lower():
            return ActionPlan(
                thought="Fallback: open application via Win key",
                action="hotkey",
                value="win",
                expected_outcome="Start menu opens",
            )

        # 2. 点击按钮
        if "点击" in task or "click" in task.lower() or "按" in task:
            btn = next(
                (e for e in observation.elements if e.element_type == "button"),
                None,
            )
            if btn:
                return ActionPlan(
                    thought="Fallback: clicking the first available button",
                    action="click",
                    target_element_id=btn.element_id,
                    expected_outcome="Button is clicked",
                )

        # 3. 输入文本
        if "输入" in task or "搜索" in task or "type" in task.lower() or "写" in task:
            inp = next(
                (e for e in observation.elements if e.element_type == "input"),
                None,
            )
            if inp:
                return ActionPlan(
                    thought="Fallback: typing into the first available input",
                    action="type",
                    target_element_id=inp.element_id,
                    value=extracted_text,
                    expected_outcome="Text appears in input field",
                )

        return ActionPlan(action="noop")
