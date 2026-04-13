"""
失败回放模块：记录和检索技能失败案例，注入注意事项
"""
from .storage.sqlite_store import SQLiteStore
from .schemas.models import FailureNote


class FailureReplay:
    """
    当某个技能执行失败时，记录失败模式和根因。
    下次调用该技能前，自动将相关注意事项注入上下文，
    帮助 Agent 避免重复犯错。
    """

    def __init__(self, store: SQLiteStore):
        self.store = store

    def record(self, skill_name: str, failure_pattern: str, root_cause: str = ""):
        """记录一次技能失败"""
        self.store.add_failure(
            FailureNote(
                skill_name=skill_name,
                failure_pattern=failure_pattern,
                root_cause=root_cause,
            )
        )

    def get_cautions(self, skill_name: str, limit: int = 3) -> list[str]:
        """获取指定技能的最近注意事项"""
        notes = self.store.get_failures(skill_name)
        cautions = []
        for note in notes[:limit]:
            text = note.root_cause or note.failure_pattern
            cautions.append(f"注意：{text}")
        return cautions
