"""
Agent Lifelong Learning Benchmark

评估维度：
1. 技能发现准确率：PatternMiner 挖出的簇是否与真实类别一致
2. 技能路由准确率：LearningAgent 对新请求是否匹配到正确技能
3. 参数填充准确率：ParamFiller 是否正确提取并填充参数
4. 失败回放有效性：失败后是否注入 expected caution
5. 成功率追踪：多次 feedback 后 success_rate 是否收敛正确
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


BENCHMARK_SEED = {
    "query_stock": [
        "帮我查一下贵州茅台的K线",
        "看看比亚迪最近5天的走势",
        "查一下宁德时代股价",
        "茅台股票怎么样",
    ],
    "query_weather": [
        "今天北京天气怎么样",
        "杭州明天会下雨吗",
        "上海未来一周气温",
    ],
    "calculate_expression": [
        "计算 23 + 45 * 2",
        "帮我算一下 (100 - 30) / 7",
        "199 乘以 3 加 17",
    ],
}

# 测试路由的用例
ROUTING_CASES = [
    ("帮我看看比亚迪的走势", "query_stock"),
    ("北京明天天气如何", "query_weather"),
    ("计算 100 + 200", "calculate_expression"),
    ("今天有什么新闻", None),  # fallback
]

# 测试参数填充的用例
PARAM_CASES = [
    ("query_stock", "帮我查一下 000001 的股价", {"stock_code": "000001", "metric": "price"}),
    ("query_weather", "杭州明天天气", {"city": "杭州", "date": "tomorrow"}),
    ("calculate_expression", "计算 1+2", {"expression": "1+2"}),
]


def build_env():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "benchmark.db")
    store = SQLiteStore(db_path)

    for category, requests in BENCHMARK_SEED.items():
        for i, req in enumerate(requests):
            store.add_experience(
                ExperienceRecord(
                    session_id=f"sess-{category}-{i}",
                    user_request=req,
                    env_state={},
                    agent_actions=[{"tool": category}],
                    user_feedback="positive",
                )
            )

    miner = PatternMiner(store)
    patterns = miner.mine(n_clusters=3, min_cluster_size=2)

    distiller = SkillDistiller(llm_client=None)
    lib = SkillLibrary(store)
    for p in patterns:
        skill = distiller.distill(p)
        lib.add(skill)

    agent = LearningAgent(lib)
    return store, lib, agent, tmpdir


def eval_skill_discovery(lib: SkillLibrary) -> tuple[bool, str]:
    """评估是否发现了全部 3 个预期技能类别"""
    names = {s.name for s in lib.list_all()}
    expected = {"query_stock", "query_weather", "calculate_expression"}
    ok = expected.issubset(names)
    return ok, f"discovered={names}, expected={expected}"


def eval_routing(agent: LearningAgent) -> tuple[float, list]:
    """评估 LearningAgent 的路由准确率"""
    correct = 0
    details = []
    for query, expected_skill in ROUTING_CASES:
        result = agent.handle(query)
        actual = result.get("skill_name")
        match = (actual == expected_skill)
        if match:
            correct += 1
        details.append({
            "query": query,
            "expected": expected_skill,
            "actual": actual,
            "match": match,
        })
    return correct / len(ROUTING_CASES), details


def eval_param_filling(lib: SkillLibrary) -> tuple[float, list]:
    """评估参数填充准确率"""
    from src.param_filler import ParamFiller
    filler = ParamFiller()
    correct = 0
    details = []
    for skill_name, query, expected_params in PARAM_CASES:
        skill = lib.get(skill_name)
        if not skill:
            details.append({"skill": skill_name, "ok": False, "reason": "skill not found"})
            continue
        filled = filler.fill(query, skill)
        # 检查 expected_params 中的键值是否都在 filled 中正确出现
        ok = True
        for key, val in expected_params.items():
            found = str(filled).find(str(val)) >= 0
            if not found:
                ok = False
        if ok:
            correct += 1
        details.append({"skill": skill_name, "query": query, "filled": filled, "ok": ok})
    return correct / len(PARAM_CASES), details


def eval_failure_replay(agent: LearningAgent, lib: SkillLibrary) -> tuple[bool, str]:
    """评估失败回放是否有效注入注意事项"""
    agent.feedback("query_stock", success=False, root_cause="股票代码不能为空")
    result = agent.handle("查一下茅台股价")
    cautions = result.get("cautions", [])
    ok = any("股票代码不能为空" in c for c in cautions)
    return ok, f"cautions={cautions}"


def eval_success_rate_tracking(lib: SkillLibrary) -> tuple[bool, str]:
    """评估多次 feedback 后成功率统计是否正确"""
    # query_stock 当前状态：1 次失败
    # 再执行 3 成功 + 1 失败
    agent = LearningAgent(lib)
    for _ in range(3):
        agent.feedback("query_stock", success=True)
    agent.feedback("query_stock", success=False)

    skill = lib.get("query_stock")
    expected_rate = 3 / 5  # 3 成功 / 5 总计
    ok = abs(skill.success_rate - expected_rate) < 1e-6 and skill.usage_count == 5
    return ok, f"success_rate={skill.success_rate}, usage={skill.usage_count}"


def main():
    print("[Setup] 构建学习环境并蒸馏技能...")
    store, lib, agent, tmpdir = build_env()

    results = {}

    # 1. 技能发现
    print("\n[Eval 1] 技能发现准确率")
    ok, msg = eval_skill_discovery(lib)
    results["skill_discovery"] = ok
    print(f"       {'PASS' if ok else 'FAIL'} — {msg}")

    # 2. 路由准确率
    print("\n[Eval 2] 技能路由准确率")
    acc, details = eval_routing(agent)
    results["routing_accuracy"] = acc
    print(f"       Accuracy: {acc:.0%}")
    for d in details:
        status = "OK" if d["match"] else "MISMATCH"
        print(f"       [{status}] '{d['query']}' -> expected={d['expected']}, actual={d['actual']}")

    # 3. 参数填充
    print("\n[Eval 3] 参数填充准确率")
    acc2, details2 = eval_param_filling(lib)
    results["param_filling_accuracy"] = acc2
    print(f"       Accuracy: {acc2:.0%}")
    for d in details2:
        status = "OK" if d.get("ok") else "FAIL"
        print(f"       [{status}] {d['skill']}: {d.get('query', '')} -> {d.get('filled', {})}")

    # 4. 失败回放
    print("\n[Eval 4] 失败回放有效性")
    ok4, msg4 = eval_failure_replay(agent, lib)
    results["failure_replay"] = ok4
    print(f"       {'PASS' if ok4 else 'FAIL'} — {msg4}")

    # 5. 成功率追踪
    print("\n[Eval 5] 成功率追踪正确性")
    ok5, msg5 = eval_success_rate_tracking(lib)
    results["success_rate_tracking"] = ok5
    print(f"       {'PASS' if ok5 else 'FAIL'} — {msg5}")

    # 汇总
    print("\n" + "=" * 50)
    print("Benchmark Summary")
    print("=" * 50)
    total_score = sum([
        1.0 if results["skill_discovery"] else 0.0,
        results["routing_accuracy"],
        results["param_filling_accuracy"],
        1.0 if results["failure_replay"] else 0.0,
        1.0 if results["success_rate_tracking"] else 0.0,
    ])
    print(f"Total Score: {total_score:.1f} / 5.0")
    for k, v in results.items():
        display = f"{v:.0%}" if isinstance(v, float) else ("PASS" if v else "FAIL")
        print(f"  {k:30s}: {display}")

    print(f"\n[Cleanup] 临时数据目录: {tmpdir}")


if __name__ == "__main__":
    main()
