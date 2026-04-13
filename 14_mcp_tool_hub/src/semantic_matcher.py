"""
语义匹配模块：基于 Sentence-BERT 的意图-工具检索
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from .schemas.models import ToolMetadata


class SemanticMatcher:
    """
    将用户自然语言意图与工具描述做语义匹配，
    替代传统的关键词匹配，提升泛化能力。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.tools: list[ToolMetadata] = []
        self.embeddings: np.ndarray | None = None

    def index(self, tools: list[ToolMetadata]):
        """为工具列表构建语义索引"""
        self.tools = tools
        texts = [f"{t.name}: {t.description}" for t in tools]
        if texts:
            self.embeddings = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            self.embeddings = None

    # 关键词硬过滤门：若查询不包含任一关键词，则直接排除该工具
    _KEYWORD_GATES: dict[str, list[str]] = {
        "search_duckduckgo": ["搜索", "查找", "网页", "url", "site", "search", "query"],
        "fetch_url": ["网页", "url", "site", "抓取", "fetch", "打开链接"],
        "run_shell": ["命令", "shell", "执行", "run command", "cmd", "终端"],
    }

    def _passes_keyword_gate(self, query: str, tool_name: str) -> bool:
        gates = self._KEYWORD_GATES.get(tool_name.split("/")[-1], [])
        if not gates:
            return True
        query_lower = query.lower()
        return any(g.lower() in query_lower for g in gates)

    def match(
        self, query: str, top_k: int = 5, min_score: float = 0.3
    ) -> list[tuple[ToolMetadata, float]]:
        """
        检索与查询最相关的工具。
        先经过 keyword gate 过滤，再做语义匹配。
        返回 [(ToolMetadata, cosine_similarity), ...]，按相似度降序。
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        #  keyword gate 预过滤
        filtered_tools = [t for t in self.tools if self._passes_keyword_gate(query, t.name)]
        if not filtered_tools:
            return []

        texts = [f"{t.name}: {t.description}" for t in filtered_tools]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = np.dot(embeddings, query_vec.T).squeeze()
        # 处理只有 1 个工具时 scores 是 0 维数组的情况
        if scores.ndim == 0:
            sorted_scores = np.array([float(scores)])
            top_indices = np.array([0])
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]
            sorted_scores = scores

        results = []
        for idx in top_indices:
            score = float(sorted_scores[idx]) if sorted_scores.ndim > 0 else float(sorted_scores)
            if score >= min_score:
                results.append((filtered_tools[idx], score))
        return results
