"""
端到端演示：Agent 终身学习闭环

完整流程：
  1. 向经验池注入一批模拟对话数据
  2. PatternMiner 聚类挖掘重复模式
  3. SkillDistiller 将模式抽象为 SkillTemplate
  4. SkillLibrary 存储技能
  5. LearningAgent 用新技能响应用户请求
  6. 模拟一次失败，观察 FailureReplay 的注意事项注入
  7. 模拟多次成功，观察成功率提升
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.sqlite_store import SQLiteStore
from src.pattern_miner import PatternMiner
from src.skill_distiller import SkillDistiller
from src.skill_library import SkillLibrary
from src.learning_agent import LearningAgent
from src.schemas.models import ExperienceRecord


def seed_experiences(store: SQLiteStore):
    """向经验池注入 3 类模拟对话数据"""
    print("[Step 1/7] 注入模拟经验数据...")

    stock_requests = [
        "帮我查一下贵州茅台的K线",
        "看看比亚迪最近5天的走势",
        "查一下宁德时代股价",
        "茅台股票怎么样",
        "五粮液今天的价格",
    ]
    weather_requests = [
        "今天北京天气怎么样",
        "杭州明天会下雨吗",
        "上海未来一周气温",
        "深圳今天空气质量",
    ]
    calc_requests = [
        "计算 23 + 45 * 2",
        "帮我算一下 (100 - 30) / 7",
        "199 乘以 3 加 17",
    ]

    for i, req in enumerate(stock_requests):
        store.add_experience(
            ExperienceRecord(
                session_id=f"sess-stock-{i}",
                user_request=req,
                env_state={},
                agent_actions=[{"tool": "stock_query", "params": {"code": "auto", "metric": "auto"}}],
                user_feedback="positive",
            )
        )
    for i, req in enumerate(weather_requests):
        store.add_experience(
            ExperienceRecord(
                session_id=f"sess-weather-{i}",
                user_request=req,
                env_state={},
                agent_actions=[{"tool": "weather_query", "params": {"city": "auto", "date": "auto"}}],
                user_feedback="positive",
            )
        )
    for i, req in enumerate(calc_requests):
        store.add_experience(
            ExperienceRecord(
                session_id=f"sess-calc-{i}",
                user_request=req,
                env_state={},
                agent_actions=[{"tool": "calculator", "params": {"expression": "auto"}}],
                user_feedback="positive",
            )
        )
    total = len(store.get_experiences())
    print(f"       已注入 {total} 条经验记录。")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        store = SQLiteStore(db_path)

        # 1. 注入数据
        seed_experiences(store)

        # 2. 模式挖掘
        print("\n[Step 2/7] 模式挖掘（K-Means 聚类）...")
        miner = PatternMiner(store)
        patterns = miner.mine(n_clusters=3, min_cluster_size=2)
        print(f"       发现 {len(patterns)} 个重复模式")
        for p in patterns:
            print(f"       - Cluster {p['cluster_id']}: "
                  f"size={p['size']}, representative='{p['representative_request']}'")

        # 3. 技能蒸馏
        print("\n[Step 3/7] 技能蒸馏（Fallback Distiller）...")
        distiller = SkillDistiller(llm_client=None)
        lib = SkillLibrary(store)
        for p in patterns:
            skill = distiller.distill(p)
            print(f"       蒸馏出技能: {skill.name} | params={skill.params}")
            lib.add(skill)

        # 4. 展示技能库
        print("\n[Step 4/7] 当前技能库快照")
        for s in lib.list_all():
            print(f"       - {s.name}: {s.description} (v{s.version})")

        # 5. LearningAgent 使用技能
        print("\n[Step 5/7] LearningAgent 响应新请求...")
        agent = LearningAgent(lib)

        test_queries = [
            "帮我看看比亚迪的走势",
            "北京明天天气如何",
            "计算 100 + 200",
            "今天有什么新闻",  # 无匹配技能
        ]
        for q in test_queries:
            result = agent.handle(q)
            mode = result["mode"]
            skill = result.get("skill_name") or "N/A"
            action = result.get("action", {})
            print(f"       [{mode}] '{q}' -> skill={skill}, action={action}")

        # 6. 模拟失败与失败回放
        print("\n[Step 6/7] 模拟失败并注入注意事项...")
        # 假装 query_stock 执行失败了
        agent.feedback("query_stock", success=False, root_cause="股票代码不能为空")
        # 再次请求同一类任务
        result = agent.handle("查一下茅台股价")
        print(f"       再次请求 '查一下茅台股价' -> cautions={result['cautions']}")

        # 7. 成功率提升
        print("\n[Step 7/7] 多次反馈提升成功率...")
        for _ in range(5):
            agent.feedback("query_stock", success=True)
            agent.feedback("query_weather", success=True)
        for s in lib.list_all():
            print(f"       - {s.name}: usage={s.usage_count}, success_rate={s.success_rate:.2%}")

        print("\n[Done] Demo 完成。")


if __name__ == "__main__":
    main()
