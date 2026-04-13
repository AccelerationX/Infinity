"""
Tool Hub：多 MCP Server 聚合、语义路由、DAG 编排与安全审计
"""
import asyncio
import uuid
from datetime import datetime
from .client import MCPClient
from .schemas.models import (
    ToolMetadata,
    ToolChainPlan,
    ExecutionResult,
    AuditLogEntry,
    RiskLevel,
)
from .security import SecurityManager
from .audit_logger import AuditLogger
from .semantic_matcher import SemanticMatcher
from .planner import DAGPlanner
from .argument_resolver import ArgumentResolver


class ToolHub:
    """
    核心中枢：管理多个 MCPClient，统一暴露工具发现、
    语义匹配、链式规划与执行能力。
    """

    def __init__(self, argument_resolver: ArgumentResolver | None = None):
        self.clients: dict[str, MCPClient] = {}
        self.tool_index: dict[str, ToolMetadata] = {}
        self.security = SecurityManager()
        self.audit = AuditLogger()
        self.matcher = SemanticMatcher()
        self.planner = DAGPlanner(argument_resolver=argument_resolver)

    async def register_client(
        self,
        client: MCPClient,
        risk_overrides: dict[str, str] | None = None,
    ):
        """
        注册一个新的 MCP Server Client，并重建索引。
        risk_overrides: {tool_name: risk_level}，用于覆盖 FastMCP Server 没有 risk_level 字段的情况。
        """
        await client.connect()
        self.clients[client.server_name] = client
        overrides = risk_overrides or {}

        for t in client.tools:
            raw_name = t["name"]
            meta = ToolMetadata(
                name=raw_name,
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                risk_level=overrides.get(raw_name, t.get("risk_level", RiskLevel.SAFE)),
                server_name=client.server_name,
            )
            # 处理跨 Server 工具重名
            key = meta.name
            if key in self.tool_index:
                key = f"{meta.server_name}/{meta.name}"
                meta.name = key
            self.tool_index[key] = meta

        self.matcher.index(list(self.tool_index.values()))

    async def shutdown(self):
        """优雅关闭所有 Client 连接"""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()
        self.tool_index.clear()
        self.matcher.index([])

    def list_tools(self) -> list[ToolMetadata]:
        """返回当前注册的所有工具元数据"""
        return list(self.tool_index.values())

    async def match_and_plan(
        self, intent: str, top_k: int = 5, fixed_args: dict | None = None
    ) -> ToolChainPlan:
        """语义匹配 -> DAG 规划 的完整链路"""
        matches = self.matcher.match(intent, top_k=top_k)
        return self.planner.plan(intent, matches, fixed_args=fixed_args)

    async def execute_plan(
        self,
        intent: str,
        plan: ToolChainPlan,
        user_confirmed: bool = False,
        session_id: str | None = None,
    ) -> list[ExecutionResult]:
        """
        执行一个工具链计划，并在执行前做安全检查。
        所有执行结果会被写入审计日志。
        """
        session_id = session_id or str(uuid.uuid4())

        tools_in_plan = [
            self.tool_index[s.tool_name]
            for s in plan.steps
            if s.tool_name in self.tool_index
        ]
        plan_risk = self.security.assess_plan_risk(tools_in_plan)

        log_entry = AuditLogEntry(
            timestamp=datetime.now(),
            session_id=session_id,
            intent=intent,
            plan=plan,
            executed_steps=[],
            user_confirmed=user_confirmed,
            passed_security=True,
        )

        results = []
        for step in plan.steps:
            tool = self.tool_index.get(step.tool_name)
            if not tool:
                err = ExecutionResult(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    output=None,
                    success=False,
                    error_message="Tool not found in registry",
                )
                results.append(err)
                continue

            permitted, msg = self.security.check_tool_permission(tool, user_confirmed)
            if not permitted:
                err = ExecutionResult(
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    output=None,
                    success=False,
                    error_message=msg,
                )
                results.append(err)
                log_entry.passed_security = False
                break

            client = self.clients[tool.server_name]
            try:
                resp = await client.call_tool(
                    tool.name.split("/")[-1], step.arguments
                )
                output_texts = [
                    c.text for c in resp.content if hasattr(c, "text")
                ]
                output = "\n".join(output_texts) if output_texts else str(resp.content)
                results.append(
                    ExecutionResult(
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        output=output,
                        success=not resp.isError,
                    )
                )
            except Exception as e:
                results.append(
                    ExecutionResult(
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        output=None,
                        success=False,
                        error_message=str(e),
                    )
                )

        log_entry.executed_steps = results
        self.audit.log(log_entry)
        return results
