"""
学习型 Agent：具备技能优先路由、失败回放和冲突消解能力的主 Agent
"""
import uuid
from .skill_library import SkillLibrary
from .failure_replay import FailureReplay
from .conflict_resolver import ConflictResolver
from .param_filler import ParamFiller
from .schemas.models import ExperienceRecord


class LearningAgent:
    """
    主 Agent 入口。收到用户请求后：
    1. 先在 Skill Library 中做语义检索
    2. 如果有匹配，做冲突消解选择最佳技能
    3. 注入失败回放中的注意事项
    4. 用 ParamFiller 将用户请求中的值填入技能模板
    5. 返回结构化执行指令
    6. 如果没有匹配，回退到通用推理逻辑
    """

    def __init__(self, skill_library: SkillLibrary, fallback_agent=None):
        self.skills = skill_library
        self.failure_replay = FailureReplay(skill_library.store)
        self.resolver = ConflictResolver()
        self.filler = ParamFiller()
        self.fallback = fallback_agent

    def handle(self, user_request: str, env_state: dict | None = None) -> dict:
        """处理用户请求，返回决策结果"""
        # 1. 技能语义检索
        candidates = self.skills.search_by_embedding(user_request, top_k=5)
        candidates = [(s, score) for s, score in candidates if score > 0.45]

        if not candidates:
            return self._fallback(user_request, env_state)

        # 2. 冲突消解
        ranked = self.resolver.resolve(candidates)
        best_skill, match_score = ranked[0]

        # 3. 失败回放
        cautions = self.failure_replay.get_cautions(best_skill.name)

        # 4. 参数填充
        filled_action = self.filler.fill(user_request, best_skill)

        # 5. 构造经验记录
        exp = ExperienceRecord(
            session_id=str(uuid.uuid4()),
            user_request=user_request,
            env_state=env_state or {},
            agent_actions=[filled_action],
            user_feedback="neutral",
        )

        return {
            "mode": "skill",
            "skill_name": best_skill.name,
            "match_score": round(match_score, 4),
            "cautions": cautions,
            "action": filled_action,
            "experience_record": exp,
        }

    def _fallback(self, user_request: str, env_state: dict | None = None) -> dict:
        """无匹配技能时的回退逻辑"""
        if self.fallback:
            result = self.fallback(user_request, env_state)
            result["mode"] = "fallback"
            return result
        return {
            "mode": "fallback",
            "skill_name": None,
            "match_score": 0.0,
            "cautions": [],
            "action": {"type": "llm_reasoning", "prompt": user_request},
            "experience_record": None,
        }

    def feedback(
        self,
        skill_name: str | None,
        success: bool,
        failure_pattern: str = "",
        root_cause: str = "",
    ):
        """
        接收外部执行反馈，更新技能统计或记录失败案例。
        """
        if skill_name:
            self.skills.record_usage(skill_name, success)
            if not success:
                self.failure_replay.record(skill_name, failure_pattern, root_cause)
