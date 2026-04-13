"""
Computer Use Agent 主循环

核心流程：
  Observe (截图+视觉解析)
    -> Plan (LLM ReAct 规划)
    -> Act (原子化执行)
    -> Verify (视觉断言)
    -> Loop (最多 max_steps 步)
"""
import argparse
from .perception import PerceptionEngine
from .planner import ReActPlanner
from .executor import AtomicExecutor
from .verifier import VisualVerifier
from .memory import SkillMemory
from .schemas.models import ExecutionRecord, SkillMacro


class ComputerUseAgent:
    """
    具备视觉感知、LLM 规划、原子化执行、视觉断言和技能记忆的 GUI Agent。
    """

    def __init__(
        self,
        perception_backend: str = "mock",
        planner_llm=None,
        verifier_backend: str = "pixel_diff",
        max_steps: int = 15,
    ):
        self.perception = PerceptionEngine(backend_name=perception_backend)
        self.planner = ReActPlanner(llm_client=planner_llm)
        self.executor = AtomicExecutor()
        self.verifier = VisualVerifier(backend=verifier_backend)
        self.memory = SkillMemory()
        self.max_steps = max_steps
        self.history: list[ExecutionRecord] = []

    def run(self, task: str) -> list[ExecutionRecord]:
        """执行一个 GUI 任务，返回完整执行记录"""
        print(f"[Task] {task}")
        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")

            # 1. Observe
            obs_before = self.perception.observe()
            self.executor.update_elements(obs_before.elements)

            # 2. Plan
            history_dicts = [r.model_dump() for r in self.history]
            plan = self.planner.plan(task, obs_before, history_dicts)
            print(f"[Plan] {plan.thought}")
            print(f"       action={plan.action}, target={plan.target_element_id}, value={plan.value!r}")

            if plan.action == "terminate":
                print("[Agent] Planner requested termination.")
                break

            # 3. Act
            exec_meta = self.executor.execute(plan)
            print(f"[Exec] {exec_meta}")

            # 4. Verify
            obs_after = self.perception.observe()
            passed, msg = self.verifier.verify(plan, obs_before, obs_after, exec_meta)
            print(f"[Verify] {'PASS' if passed else 'FAIL'} — {msg}")

            # 5. Record
            record = ExecutionRecord(
                step=step,
                observation_before=obs_before,
                plan=plan,
                observation_after=obs_after,
                verification_passed=passed,
                verification_message=msg,
            )
            self.history.append(record)

            if not passed and plan.action not in ("noop", "terminate"):
                print("[Agent] Verification failed. Replanning will occur in next step.")

        return self.history

    def save_macro(self, name: str, trigger: str | None = None):
        """将刚刚的执行历史保存为可复用宏"""
        steps = [
            r.plan for r in self.history
            if r.plan and r.plan.action not in ("noop", "terminate")
        ]
        macro = SkillMacro(
            name=name,
            trigger_description=trigger or "",
            steps=steps,
        )
        self.memory.save(macro)
        print(f"[Memory] Saved macro '{name}' with {len(steps)} step(s).")

    def load_macro(self, name: str) -> SkillMacro | None:
        """加载已保存的宏"""
        return self.memory.load(name)


# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser(description="Computer Use Agent")
    parser.add_argument("--task", default="点击屏幕上的按钮", help="任务描述")
    parser.add_argument("--backend", default="mock", help="视觉后端: mock | qwen_vl | openai_vision")
    parser.add_argument("--verifier", default="pixel_diff", help="验证后端: pixel_diff | llm")
    parser.add_argument("--max-steps", type=int, default=10, help="最大步数")
    args = parser.parse_args()

    agent = ComputerUseAgent(
        perception_backend=args.backend,
        verifier_backend=args.verifier,
        max_steps=args.max_steps,
    )
    agent.run(args.task)


if __name__ == "__main__":
    main()
