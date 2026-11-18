"""Evaluation metrics for Session-based Recommendation: HR@K (Recall@K) and MRR@K."""

from typing import Iterable, Sequence, Tuple


def hit_rate_at_k(recommended: Sequence[int], target: int, k: int = 20) -> float:
    """Return 1.0 if target item is in top-k recommended items, else 0.0."""
    return 1.0 if target in recommended[:k] else 0.0


def mrr_at_k(recommended: Sequence[int], target: int, k: int = 20) -> float:
    """Return reciprocal rank (1 / rank) if target is in top-k, else 0.0."""
    top_k = recommended[:k]
    if target in top_k:
        rank = top_k.index(target) + 1
        return 1.0 / rank
    return 0.0


def evaluate_predictions(
    predictions: Iterable[Tuple[Sequence[int], int]],
    k: int = 20,
) -> dict[str, float]:
    """Compute average HR@K and MRR@K over a collection of (recommended_items, target_item)."""
    hr_total = 0.0
    mrr_total = 0.0
    count = 0
    for rec, target in predictions:
        hr_total += hit_rate_at_k(rec, target, k=k)
        mrr_total += mrr_at_k(rec, target, k=k)
        count += 1
    if count == 0:
        return {f"HR@{k}": 0.0, f"MRR@{k}": 0.0, "count": 0}
    return {
        f"HR@{k}": hr_total / count,
        f"MRR@{k}": mrr_total / count,
        "count": count,
    }
