"""
演示：将 14_mcp_tool_hub 的审计日志格式转换为 15 的经验池记录

本脚本模拟从 MCP Hub 导入数据，然后跑一轮完整的终身学习闭环。
"""
import json
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.sqlite_store import SQLiteStore
from src.pattern_miner import PatternMiner
from src.skill_distiller import SkillDistiller
from src.skill_library import SkillLibrary
from src.learning_agent import LearningAgent
from src.schemas.models import ExperienceRecord


# 模拟 14 的审计日志条目
MOCK_AUDIT_LOGS = [
    {
        "session_id": "mcp-001",
        "intent": "帮我查一下北京天气",
        "executed_steps": [
            {"tool_name": "weather_get", "arguments": {"city": "北京"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-002",
        "intent": "杭州明天怎么样",
        "executed_steps": [
            {"tool_name": "weather_get", "arguments": {"city": "杭州", "date": "tomorrow"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-003",
        "intent": "上海气温",
        "executed_steps": [
            {"tool_name": "weather_get", "arguments": {"city": "上海"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-004",
        "intent": "查一下茅台股价",
        "executed_steps": [
            {"tool_name": "stock_query", "arguments": {"code": "600519", "metric": "price"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-005",
        "intent": "比亚迪走势",
        "executed_steps": [
            {"tool_name": "stock_query", "arguments": {"code": "002594", "metric": "kline"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-006",
        "intent": "计算 1+1",
        "executed_steps": [
            {"tool_name": "calculate", "arguments": {"expression": "1+1"}, "success": True}
        ],
    },
    {
        "session_id": "mcp-007",
        "intent": "算一下 23*45",
        "executed_steps": [
            {"tool_name": "calculate", "arguments": {"expression": "23*45"}, "success": True}
        ],
    },
]


def import_audit_logs(store: SQLiteStore, logs: list[dict]):
    """将 MCP Hub 风格的审计日志转换为 ExperienceRecord 并存入 SQLite"""
    print("[Bridge] 导入 MCP 审计日志到经验池...")
    for entry in logs:
        actions = [
            {"tool": step["tool_name"], "params": step["arguments"]}
            for step in entry["executed_steps"]
        ]
        all_success = all(step["success"] for step in entry["executed_steps"])
        feedback = "positive" if all_success else "negative"
        record = ExperienceRecord(
            session_id=entry["session_id"],
            user_request=entry["intent"],
            env_state={"source": "mcp_hub"},
            agent_actions=actions,
            user_feedback=feedback,
            timestamp=datetime.now(),
        )
        store.add_experience(record)
    print(f"       成功导入 {len(logs)} 条记录。")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bridge.db")
        store = SQLiteStore(db_path)

        # 1. 导入模拟的 MCP 审计日志
        import_audit_logs(store, MOCK_AUDIT_LOGS)

        # 2. 模式挖掘
        print("\n[Bridge] 模式挖掘...")
        miner = PatternMiner(store)
        patterns = miner.mine(n_clusters=3, min_cluster_size=2)
        print(f"       发现 {len(patterns)} 个模式")

        # 3. 技能蒸馏并入库
        print("\n[Bridge] 技能蒸馏...")
        distiller = SkillDistiller(llm_client=None)
        lib = SkillLibrary(store)
        for p in patterns:
            skill = distiller.distill(p)
            print(f"       -> {skill.name} (params={skill.params})")
            lib.add(skill)

        # 4. LearningAgent 用学习到的技能响应新请求
        print("\n[Bridge] LearningAgent 使用从 MCP 日志中学到的技能...")
        agent = LearningAgent(lib)
        test_queries = [
            "帮我看看深圳天气",
            "宁德时代股价",
            "计算 100+200",
            "播放一首歌",  # 无匹配
        ]
        for q in test_queries:
            result = agent.handle(q)
            mode = result["mode"]
            skill = result.get("skill_name") or "N/A"
            action = result.get("action", {})
            print(f"       [{mode}] '{q}' -> skill={skill}, action={action}")

        print("\n[Done] Bridge Demo 完成。")


if __name__ == "__main__":
    main()
