"""Baseline models for Session-based Recommendation: POP, S-POP, Item-KNN, SKNN, V-SKNN."""

from collections import Counter, defaultdict
import math
from typing import Dict, List, Sequence, Set


class BaseSessionRecommender:
    """Common interface for session-based recommenders."""

    name: str = "Base"

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "BaseSessionRecommender":
        raise NotImplementedError

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        raise NotImplementedError


class PopularityRecommender(BaseSessionRecommender):
    """POP: Recommends globally most frequent items in training sessions."""

    name = "POP"

    def __init__(self) -> None:
        self.ranked_items: List[int] = []

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "PopularityRecommender":
        counts = Counter(item for sess in train_sessions for item in sess)
        self.ranked_items = [item for item, _ in counts.most_common()]
        return self

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        return self.ranked_items[:top_k]


class SessionPopularityRecommender(BaseSessionRecommender):
    """S-POP: Prioritizes items already clicked in the current session, backed off by POP."""

    name = "S-POP"

    def __init__(self) -> None:
        self.pop = PopularityRecommender()

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "SessionPopularityRecommender":
        self.pop.fit(train_sessions)
        return self

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        sess_counts = Counter(session_prefix)
        recs = [item for item, _ in sess_counts.most_common()]
        seen = set(recs)
        for item in self.pop.ranked_items:
            if len(recs) >= top_k:
                break
            if item not in seen:
                recs.append(item)
                seen.add(item)
        return recs[:top_k]


class ItemKNNRecommender(BaseSessionRecommender):
    """Item-KNN: Cosine co-occurrence similarity based on the last clicked item."""

    name = "Item-KNN"

    def __init__(self, k_neighbors: int = 100) -> None:
        self.k_neighbors = k_neighbors
        self.neighbors: Dict[int, List[int]] = {}
        self.pop = PopularityRecommender()

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "ItemKNNRecommender":
        self.pop.fit(train_sessions)
        item_sessions: Dict[int, Set[int]] = defaultdict(set)
        for idx, sess in enumerate(train_sessions):
            for item in set(sess):
                item_sessions[item].add(idx)

        co_counts: Dict[int, Counter] = defaultdict(Counter)
        for sess in train_sessions:
            unique_items = list(set(sess))
            for i in unique_items:
                for j in unique_items:
                    if i != j:
                        co_counts[i][j] += 1

        self.neighbors = {}
        for item, related in co_counts.items():
            norm_i = math.sqrt(len(item_sessions[item]))
            scored = []
            for other, cnt in related.items():
                norm_j = math.sqrt(len(item_sessions[other]))
                sim = cnt / (norm_i * norm_j + 1e-12)
                scored.append((sim, other))
            scored.sort(reverse=True)
            self.neighbors[item] = [other for _, other in scored[: self.k_neighbors]]
        return self

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        if not session_prefix:
            return self.pop.predict_next(session_prefix, top_k=top_k)
        last_item = session_prefix[-1]
        recs = list(self.neighbors.get(last_item, [])[:top_k])
        if len(recs) < top_k:
            seen = set(recs)
            for item in self.pop.ranked_items:
                if len(recs) >= top_k:
                    break
                if item not in seen:
                    recs.append(item)
                    seen.add(item)
        return recs[:top_k]


class SessionKNNRecommender(BaseSessionRecommender):
    """SKNN and V-SKNN (Vector-weighted / recency-aware Session-KNN)."""

    def __init__(
        self,
        k_neighbors: int = 100,
        sample_size: int = 1000,
        recency_weighting: bool = False,
        decay: float = 0.5,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.sample_size = sample_size
        self.recency_weighting = recency_weighting
        self.decay = decay
        self.name = "V-SKNN" if recency_weighting else "SKNN"

        self.sessions: List[Set[int]] = []
        self.item_to_sessions: Dict[int, List[int]] = defaultdict(list)
        self.pop = PopularityRecommender()

    def fit(self, train_sessions: Sequence[Sequence[int]]) -> "SessionKNNRecommender":
        self.pop.fit(train_sessions)
        self.sessions = []
        self.item_to_sessions = defaultdict(list)

        for idx, sess in enumerate(train_sessions):
            item_set = set(sess)
            self.sessions.append(item_set)
            for item in item_set:
                self.item_to_sessions[item].append(idx)
        return self

    def _query_weights(self, session_prefix: Sequence[int]) -> Dict[int, float]:
        weights: Dict[int, float] = {}
        n = len(session_prefix)
        for pos, item in enumerate(session_prefix):
            if self.recency_weighting:
                # More recent clicks get higher weight: linear/exponential decay from the end
                dist_from_end = n - 1 - pos
                w = self.decay ** dist_from_end
                weights[item] = max(weights.get(item, 0.0), w)
            else:
                weights[item] = 1.0
        return weights

    def predict_next(self, session_prefix: Sequence[int], top_k: int = 20) -> List[int]:
        if not session_prefix:
            return self.pop.predict_next(session_prefix, top_k=top_k)

        query_weights = self._query_weights(session_prefix)
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

        # Gather candidate historical sessions (most recent sample_size per item)
        candidate_ids: Set[int] = set()
        for item in query_weights:
            sess_list = self.item_to_sessions.get(item, [])
            if len(sess_list) > self.sample_size:
                sess_list = sess_list[-self.sample_size :]
            candidate_ids.update(sess_list)

        # Score candidate sessions by weighted cosine similarity
        scored_sessions = []
        for s_id in candidate_ids:
            hist_items = self.sessions[s_id]
            dot = sum(w for item, w in query_weights.items() if item in hist_items)
            if dot <= 0:
                continue
            sim = dot / (query_norm * math.sqrt(len(hist_items)) + 1e-12)
            scored_sessions.append((sim, s_id))

        scored_sessions.sort(reverse=True)
        top_sessions = scored_sessions[: self.k_neighbors]

        item_scores: Dict[int, float] = defaultdict(float)
        for sim, s_id in top_sessions:
            for item in self.sessions[s_id]:
                item_scores[item] += sim

        ranked = [item for item, _ in sorted(item_scores.items(), key=lambda x: x[1], reverse=True)]
        if len(ranked) < top_k:
            seen = set(ranked)
            for item in self.pop.ranked_items:
                if len(ranked) >= top_k:
                    break
                if item not in seen:
                    ranked.append(item)
                    seen.add(item)
        return ranked[:top_k]
