"""
审计日志模块：以 JSONL 格式持久化所有工具调用链路
"""
import json
import os
from datetime import datetime
from .schemas.models import AuditLogEntry


class AuditLogger:
    """
    记录每一次从用户意图到工具执行完成的完整链路，
    用于后续的安全审计、成功率分析和策略优化。
    """

    def __init__(self, log_path: str = "experiments/audit.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, entry: AuditLogEntry):
        """追加写入一条审计记录"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def load_all(self) -> list[AuditLogEntry]:
        """读取所有审计记录"""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(AuditLogEntry.model_validate_json(line))
                except Exception:
                    continue
        return entries

    def get_tool_success_rate(self, tool_name: str) -> float:
        """计算某个工具的历史成功率"""
        entries = self.load_all()
        total = 0
        success = 0
        for entry in entries:
            for step in entry.executed_steps:
                if step.tool_name == tool_name:
                    total += 1
                    if step.success:
                        success += 1
        return success / total if total > 0 else 0.0

    def get_recent_errors(self, limit: int = 10) -> list[dict]:
        """获取最近的失败记录摘要"""
        entries = self.load_all()
        errors = []
        for entry in reversed(entries):
            for step in entry.executed_steps:
                if not step.success and step.error_message:
                    errors.append({
                        "timestamp": entry.timestamp.isoformat(),
                        "tool": step.tool_name,
                        "error": step.error_message,
                        "intent": entry.intent,
                    })
                    if len(errors) >= limit:
                        return errors
        return errors
