"""
测试 SQLiteStore 的 CRUD 能力
"""
import os
import tempfile
from src.storage.sqlite_store import SQLiteStore
from src.schemas.models import ExperienceRecord, SkillTemplate, FailureNote


def test_experience_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        record = ExperienceRecord(
            session_id="sess-001",
            user_request="查询茅台股价",
            agent_actions=[{"tool": "stock_query"}],
            user_feedback="positive",
        )
        rid = store.add_experience(record)
        assert rid > 0

        records = store.get_experiences()
        assert len(records) == 1
        assert records[0].user_request == "查询茅台股价"


def test_skill_versioning():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        skill = SkillTemplate(name="query_stock", description="查股票")
        store.add_skill(skill)

        # 更新同一技能，版本应自动递增（由 SkillLibrary 控制，此处直接测试存储层）
        skill2 = SkillTemplate(name="query_stock", description="查股票行情", version=2)
        store.add_skill(skill2)

        loaded = store.get_skill("query_stock")
        assert loaded.description == "查股票行情"
        assert loaded.version == 2


def test_failure_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        note = FailureNote(skill_name="query_stock", failure_pattern="股票代码不存在", root_cause="参数错误")
        store.add_failure(note)

        notes = store.get_failures("query_stock")
        assert len(notes) == 1
        assert notes[0].failure_pattern == "股票代码不存在"
