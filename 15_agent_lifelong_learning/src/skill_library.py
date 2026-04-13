"""
技能库模块：技能的存储、检索、版本管理和成功率追踪
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from .storage.sqlite_store import SQLiteStore
from .schemas.models import SkillTemplate


class SkillLibrary:
    """
    提供技能的持久化存储、基于 embedding 的语义检索、
    使用记录更新和版本管理。
    """

    def __init__(self, store: SQLiteStore, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.store = store
        self.model = SentenceTransformer(model_name)

    def add(self, skill: SkillTemplate):
        """添加或更新技能；如果已存在则自动升级版本号"""
        existing = self.store.get_skill(skill.name)
        if existing:
            skill.version = existing.version + 1
            skill.usage_count = existing.usage_count
            skill.success_rate = existing.success_rate
            skill.created_at = existing.created_at
        self.store.add_skill(skill)

    def get(self, name: str) -> SkillTemplate | None:
        return self.store.get_skill(name)

    def list_all(self) -> list[SkillTemplate]:
        return self.store.list_skills()

    def search_by_embedding(self, query: str, top_k: int = 3, min_score: float = 0.3) -> list[tuple[SkillTemplate, float]]:
        """
        使用 trigger_patterns 的平均 embedding 与用户查询做语义匹配。
        """
        query_vec = self.model.encode([query], normalize_embeddings=True)
        skills = self.list_all()
        results = []
        for skill in skills:
            if not skill.trigger_patterns:
                continue
            skill_vecs = self.model.encode(skill.trigger_patterns, normalize_embeddings=True)
            avg_vec = skill_vecs.mean(axis=0)
            score = float(query_vec.flatten().dot(avg_vec.flatten()))
            if score >= min_score:
                results.append((skill, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def record_usage(self, name: str, success: bool):
        """更新技能的使用次数和成功率（增量更新）"""
        skill = self.store.get_skill(name)
        if not skill:
            return
        skill.usage_count += 1
        prev_total = skill.success_rate * (skill.usage_count - 1)
        skill.success_rate = (prev_total + (1.0 if success else 0.0)) / skill.usage_count
        self.store.add_skill(skill)
