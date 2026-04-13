"""
测试 PatternMiner 的聚类挖掘能力
"""
import os
import tempfile
from src.storage.sqlite_store import SQLiteStore
from src.pattern_miner import PatternMiner
from src.schemas.models import ExperienceRecord


def test_mine_patterns_from_experiences():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path)
        # 插入 3 类经验
        for req in ["查茅台股价", "查比亚迪股价", "查宁德时代股价"]:
            store.add_experience(ExperienceRecord(session_id="s1", user_request=req, agent_actions=[{"tool": "stock"}]))
        for req in ["北京天气", "上海天气"]:
            store.add_experience(ExperienceRecord(session_id="s1", user_request=req, agent_actions=[{"tool": "weather"}]))
        for req in ["计算 1+1", "计算 2*3"]:
            store.add_experience(ExperienceRecord(session_id="s1", user_request=req, agent_actions=[{"tool": "calc"}]))

        miner = PatternMiner(store)
        patterns = miner.mine(n_clusters=3, min_cluster_size=1)
        assert len(patterns) >= 2  # 至少应挖出 2 个簇
        # 检查 representative_request 不为空
        for p in patterns:
            assert p["representative_request"]
            assert p["size"] > 0
