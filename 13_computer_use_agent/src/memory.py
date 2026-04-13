"""
技能记忆模块：保存和复用成功的 GUI 操作序列（宏）
"""
import json
import os
from datetime import datetime
from .schemas.models import SkillMacro, ActionPlan


class SkillMemory:
    """
    将成功的任务执行序列持久化为 JSON 格式的 SkillMacro，
    供后续相同或相似任务直接调用，减少推理开销。
    """

    def __init__(self, storage_dir: str = "experiments/macros"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        return os.path.join(self.storage_dir, f"{safe}.json")

    def save(self, macro: SkillMacro):
        """保存宏到磁盘"""
        path = self._path(macro.name)
        with open(path, "w", encoding="utf-8") as f:
            # Pydantic v2 的 model_dump 可以处理嵌套模型
            json.dump(macro.model_dump(), f, ensure_ascii=False, indent=2, default=str)

    def load(self, name: str) -> SkillMacro | None:
        """从磁盘加载宏"""
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return SkillMacro.model_validate(data)

    def list_macros(self) -> list[str]:
        """列出所有已保存的宏名称"""
        files = os.listdir(self.storage_dir)
        return [f.replace(".json", "") for f in files if f.endswith(".json")]

    def delete(self, name: str):
        """删除指定宏"""
        path = self._path(name)
        if os.path.exists(path):
            os.remove(path)
