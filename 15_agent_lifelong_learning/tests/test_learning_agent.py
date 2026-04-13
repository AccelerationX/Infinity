"""
测试 LearningAgent 的端到端路由、失败回放和反馈闭环
"""
import os
import tempfile
from src.storage.sqlite_store import SQLiteStore
from src.skill_library import SkillLibrary
from src.learning_agent import LearningAgent
from src.schemas.models import SkillTemplate


def test_skill_routing():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)
        lib.add(
            SkillTemplate(
                name="query_stock",
                description="查询股票",
                trigger_patterns=["查一下茅台股价", "看看比亚迪走势"],
                action_template={"tool": "stock_query", "params": {"code": "{{stock}}"}},
            )
        )
        agent = LearningAgent(lib)
        result = agent.handle("帮我查一下贵州茅台的股价")
        assert result["mode"] == "skill"
        assert result["skill_name"] == "query_stock"
        assert result["action"]["tool"] == "stock_query"


def test_fallback_when_no_skill_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)
        agent = LearningAgent(lib)
        result = agent.handle("今天有什么新闻")
        assert result["mode"] == "fallback"
        assert result["skill_name"] is None


def test_failure_replay_injection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)
        lib.add(
            SkillTemplate(
                name="query_stock",
                description="查询股票",
                trigger_patterns=["查一下茅台股价"],
                action_template={"tool": "stock_query"},
            )
        )
        agent = LearningAgent(lib)
        # 先反馈一次失败
        agent.feedback("query_stock", success=False, root_cause="股票代码不能为空")
        # 再次调用时应注入注意事项
        result = agent.handle("帮我查一下茅台")
        assert "注意：股票代码不能为空" in result["cautions"]


def test_feedback_updates_success_rate():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        lib = SkillLibrary(store)
        lib.add(
            SkillTemplate(
                name="query_stock",
                description="查询股票",
                trigger_patterns=["查一下茅台股价"],
            )
        )
        agent = LearningAgent(lib)
        agent.feedback("query_stock", success=True)
        agent.feedback("query_stock", success=False)
        skill = lib.get("query_stock")
        assert skill.usage_count == 2
        assert skill.success_rate == 0.5
