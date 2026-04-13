"""
模式挖掘模块：从经验池中提取重复行为模式
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from .storage.sqlite_store import SQLiteStore


class PatternMiner:
    """
    使用 Sentence-BERT + K-Means 对用户请求进行聚类，
    识别高置信度的重复交互模式，作为后续技能蒸馏的输入。
    """

    def __init__(self, store: SQLiteStore, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.store = store
        self.model = SentenceTransformer(model_name)

    def mine(self, n_clusters: int = 5, min_cluster_size: int = 2) -> list[dict]:
        """
        从经验池中挖掘重复模式。
        返回的每个模式包含代表性请求、同类请求样本和对应的动作序列。
        """
        experiences = self.store.get_experiences(limit=10000)
        if len(experiences) < n_clusters * 2:
            # 数据不足时动态调整聚类数
            n_clusters = max(1, len(experiences) // 2)

        if not experiences:
            return []

        requests = [e.user_request for e in experiences]
        embeddings = self.model.encode(
            requests, convert_to_numpy=True, normalize_embeddings=True
        )
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        patterns = []
        for cid in range(n_clusters):
            indices = np.where(labels == cid)[0]
            if len(indices) < min_cluster_size:
                continue

            cluster_requests = [requests[i] for i in indices]
            cluster_actions = [experiences[i].agent_actions for i in indices]
            centroid = kmeans.cluster_centers_[cid]
            distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)
            rep_idx = indices[np.argmin(distances)]

            patterns.append(
                {
                    "cluster_id": int(cid),
                    "size": int(len(indices)),
                    "representative_request": requests[rep_idx],
                    "sample_requests": cluster_requests[:5],
                    "sample_actions": cluster_actions[:3],
                }
            )

        return patterns
