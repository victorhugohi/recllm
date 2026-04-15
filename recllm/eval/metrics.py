"""Recommendation evaluation metrics.

Implements standard metrics following RecBole conventions for comparability.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from recllm.data.base import InteractionData
    from recllm.models.base import BaseModel


def ndcg_at_k(ranked_list: np.ndarray, ground_truth: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        ranked_list: Array of recommended item IDs, ordered by score descending.
        ground_truth: Set of relevant item IDs.
        k: Cutoff position.

    Returns:
        NDCG@K score in [0, 1].
    """
    ranked_list = ranked_list[:k]
    dcg = 0.0
    for i, item_id in enumerate(ranked_list):
        if int(item_id) in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(ranked_list: np.ndarray, ground_truth: set[int], k: int) -> float:
    """Hit Rate at K: 1 if any relevant item in top-K, else 0.

    Args:
        ranked_list: Array of recommended item IDs.
        ground_truth: Set of relevant item IDs.
        k: Cutoff position.

    Returns:
        1.0 or 0.0.
    """
    for item_id in ranked_list[:k]:
        if int(item_id) in ground_truth:
            return 1.0
    return 0.0


def mrr(ranked_list: np.ndarray, ground_truth: set[int]) -> float:
    """Mean Reciprocal Rank: 1/position of first relevant item.

    Args:
        ranked_list: Array of recommended item IDs.
        ground_truth: Set of relevant item IDs.

    Returns:
        Reciprocal rank in (0, 1], or 0 if no relevant item found.
    """
    for i, item_id in enumerate(ranked_list):
        if int(item_id) in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_list: np.ndarray, ground_truth: set[int], k: int) -> float:
    """Precision at K: fraction of top-K items that are relevant.

    Args:
        ranked_list: Array of recommended item IDs.
        ground_truth: Set of relevant item IDs.
        k: Cutoff position.

    Returns:
        Precision score in [0, 1].
    """
    hits = sum(1 for item_id in ranked_list[:k] if int(item_id) in ground_truth)
    return hits / k


def recall_at_k(ranked_list: np.ndarray, ground_truth: set[int], k: int) -> float:
    """Recall at K: fraction of relevant items that appear in top-K.

    Args:
        ranked_list: Array of recommended item IDs.
        ground_truth: Set of relevant item IDs.
        k: Cutoff position.

    Returns:
        Recall score in [0, 1].
    """
    if len(ground_truth) == 0:
        return 0.0
    hits = sum(1 for item_id in ranked_list[:k] if int(item_id) in ground_truth)
    return hits / len(ground_truth)


def _parse_metric(metric_str: str) -> tuple[str, int | None]:
    """Parse metric string like 'ndcg@10' into (name, k)."""
    if "@" in metric_str:
        name, k_str = metric_str.split("@", 1)
        return name.lower(), int(k_str)
    return metric_str.lower(), None


def compute_metrics(
    model: BaseModel,
    test_data: InteractionData,
    metrics: list[str],
    n_recommendations: int = 100,
) -> dict[str, float]:
    """Compute evaluation metrics for a fitted model on test data.

    Uses full-ranking evaluation: for each test user, rank all items
    and compute metrics against the user's test interactions.

    Args:
        model: Fitted recommendation model.
        test_data: Test interactions.
        metrics: List of metric strings (e.g., ["ndcg@10", "hr@10", "mrr"]).
        n_recommendations: Number of items to rank per user.

    Returns:
        Dict mapping metric names to averaged values.
    """
    # Build per-user ground truth from test data
    test_arrays = test_data.to_numpy()
    user_ground_truth: dict[int, set[int]] = {}
    for u, i in zip(test_arrays["user_id"], test_arrays["item_id"], strict=False):
        user_ground_truth.setdefault(int(u), set()).add(int(i))

    # Build per-user seen items from train data (for exclusion)
    # Access via the parent's train split if available
    user_seen: dict[int, set[int]] = {}
    if hasattr(model, "_user_interactions"):
        # BPR and similar models store this during training
        user_seen = {
            k: v for k, v in model._user_interactions.items()
        }

    # Compute metrics per user
    parsed_metrics = [_parse_metric(m) for m in metrics]
    metric_sums: dict[str, float] = {m: 0.0 for m in metrics}
    n_evaluated = 0

    for user_id, relevant_items in user_ground_truth.items():
        # Get recommendations
        try:
            seen = user_seen.get(user_id, set())
            recs = model.recommend(
                user_id, n=n_recommendations, exclude_seen=True, seen_items=seen
            )
            ranked_items = np.array([item_id for item_id, _ in recs])
        except (RuntimeError, KeyError):
            continue

        if len(ranked_items) == 0:
            continue

        n_evaluated += 1

        for metric_str, (name, k) in zip(metrics, parsed_metrics, strict=False):
            if name == "ndcg":
                metric_sums[metric_str] += ndcg_at_k(ranked_items, relevant_items, k or 10)
            elif name in ("hr", "hit_rate"):
                metric_sums[metric_str] += hit_rate_at_k(ranked_items, relevant_items, k or 10)
            elif name == "mrr":
                metric_sums[metric_str] += mrr(ranked_items, relevant_items)
            elif name in ("precision", "prec"):
                metric_sums[metric_str] += precision_at_k(ranked_items, relevant_items, k or 10)
            elif name == "recall":
                metric_sums[metric_str] += recall_at_k(ranked_items, relevant_items, k or 10)

    # Average over users
    if n_evaluated == 0:
        return {m: 0.0 for m in metrics}

    return {m: metric_sums[m] / n_evaluated for m in metrics}
