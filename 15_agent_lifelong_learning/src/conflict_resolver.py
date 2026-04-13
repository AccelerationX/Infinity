"""
冲突消解模块：当多个技能匹配同一请求时，选择最优技能
"""
from .schemas.models import SkillTemplate


class ConflictResolver:
    """
    多技能冲突消解策略：
    综合语义相似度、历史成功率和使用频次，计算综合得分并排序。
    """

    def resolve(
        self, candidates: list[tuple[SkillTemplate, float]]
    ) -> list[tuple[SkillTemplate, float]]:
        """
        candidates: [(SkillTemplate, semantic_similarity), ...]
        返回按综合得分降序排列的列表。
        """

        def composite_score(item):
            skill, sim = item
            # 语义相似度权重 60%
            # 历史成功率权重 30%
            # 使用频次（归一化到 0-1）权重 10%
            usage_bonus = min(skill.usage_count / 20.0, 1.0) * 0.1
            return sim * 0.6 + skill.success_rate * 0.3 + usage_bonus

        return sorted(candidates, key=composite_score, reverse=True)
