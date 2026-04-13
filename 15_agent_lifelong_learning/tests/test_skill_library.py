"""
测试 SkillLibrary 的检索、版本管理和成功率追踪
"""
import os
import tempfile
from src.storage.sqlite_store import SQLiteStore
from src.skill_library import SkillLibrary
from src.schemas.models import SkillTemplate


def test_skill_embedding_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)

        lib.add(
            SkillTemplate(
                name="query_stock",
                description="查询股票",
                trigger_patterns=["查一下茅台股价", "看看比亚迪走势"],
            )
        )
        lib.add(
            SkillTemplate(
                name="query_weather",
                description="查询天气",
                trigger_patterns=["今天北京天气怎么样", "杭州明天会下雨吗"],
            )
        )

        results = lib.search_by_embedding("帮我查一下贵州茅台的股价", top_k=2)
        assert len(results) > 0
        assert results[0][0].name == "query_stock"


def test_record_usage_and_success_rate():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)
        lib.add(SkillTemplate(name="calc", description="计算"))

        lib.record_usage("calc", success=True)
        lib.record_usage("calc", success=True)
        lib.record_usage("calc", success=False)

        skill = lib.get("calc")
        assert skill.usage_count == 3
        assert abs(skill.success_rate - 2 / 3) < 1e-6
