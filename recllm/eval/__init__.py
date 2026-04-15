"""Evaluation metrics, statistical testing, and visualization."""

from recllm.eval.metrics import compute_metrics, hit_rate_at_k, mrr, ndcg_at_k
from recllm.eval.significance import (
    SignificanceResult,
    bootstrap_ci,
    compute_per_user_metrics,
    paired_t_test,
    wilcoxon_test,
)

__all__ = [
    "compute_metrics",
    "ndcg_at_k",
    "hit_rate_at_k",
    "mrr",
    "paired_t_test",
    "wilcoxon_test",
    "bootstrap_ci",
    "compute_per_user_metrics",
    "SignificanceResult",
]
