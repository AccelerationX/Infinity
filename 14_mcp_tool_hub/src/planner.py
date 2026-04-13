"""
DAG 规划模块：基于工具签名的依赖推断与拓扑排序
"""
import networkx as nx
from .schemas.models import ToolMetadata, ToolChainStep, ToolChainPlan
from .argument_resolver import ArgumentResolver


class DAGPlanner:
    """
    根据语义匹配结果和用户意图，生成可执行的工具链计划。
    当前版本支持基于分数排序的串行计划，并自动填充工具参数；
    后续版本将增强为基于 input/output 签名的自动依赖推断。
    """

    def __init__(self, argument_resolver: ArgumentResolver | None = None):
        self.arg_resolver = argument_resolver or ArgumentResolver()

    def plan(
        self,
        intent: str,
        matched_tools: list[tuple[ToolMetadata, float]],
        fixed_args: dict | None = None,
        min_score: float = 0.42,
    ) -> ToolChainPlan:
        """
        为匹配到的工具生成执行计划。
        只有语义匹配分数 >= min_score 的工具才会进入计划。
        """
        steps = []
        for idx, (tool, score) in enumerate(matched_tools):
            if score < min_score:
                continue
            step_id = f"step_{idx}"
            args = {}
            if fixed_args and tool.name in fixed_args:
                args = fixed_args[tool.name]
            else:
                args = self.arg_resolver.resolve(intent, tool)

            steps.append(
                ToolChainStep(
                    step_id=step_id,
                    tool_name=tool.name,
                    arguments=args,
                    reason=f"matched score={score:.3f}, risk={tool.risk_level}",
                )
            )

        # 构建 DAG 并做拓扑排序
        # 目前按匹配分数降序作为默认执行顺序
        G = nx.DiGraph()
        for s in steps:
            G.add_node(s.step_id, step=s)

        # 简单启发式：如果同时匹配到 list_directory 和 read_file，
        # 优先执行 list_directory（探索），再执行 read_file（读取）
        priority_map = {
            "list_directory": 0,
            "search_duckduckgo": 1,
            "fetch_url": 2,
            "read_file": 3,
            "calculate": 4,
            "write_file": 5,
            "run_shell": 6,
        }

        ordered = sorted(
            steps,
            key=lambda s: priority_map.get(s.tool_name.split("/")[-1], 99),
        )

        # 重新分配 depends_on：目前为串行，每一步依赖前一步
        for i in range(1, len(ordered)):
            ordered[i].depends_on = [ordered[i - 1].step_id]

        # 风险评估
        levels = {t.risk_level for t, _ in matched_tools}
        estimated_risk = "safe"
        if RiskLevel.DANGEROUS in levels:
            estimated_risk = "dangerous"
        elif RiskLevel.CAUTION in levels:
            estimated_risk = "caution"

        return ToolChainPlan(steps=ordered, estimated_risk=estimated_risk)


# 避免循环引用：在函数内部引用 RiskLevel
from .schemas.models import RiskLevel
