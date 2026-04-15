"""Statistical significance testing for recommendation experiments.

Provides paired tests to determine whether differences between models
are statistically significant, essential for rigorous experimental evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of a statistical significance test.

    Attributes:
        test_name: Name of the statistical test used.
        statistic: Test statistic value.
        p_value: p-value of the test.
        significant: Whether the result is significant at the given alpha.
        alpha: Significance level used.
        effect_size: Cohen's d or rank-biserial correlation.
        model_a_mean: Mean metric value for model A.
        model_b_mean: Mean metric value for model B.
        model_a_name: Name of model A.
        model_b_name: Name of model B.
        metric_name: Name of the metric compared.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    model_a_mean: float
    model_b_mean: float
    model_a_name: str = "Model A"
    model_b_name: str = "Model B"
    metric_name: str = ""

    def summary(self) -> str:
        winner = self.model_a_name if self.model_a_mean > self.model_b_mean else self.model_b_name
        sig_str = "YES" if self.significant else "NO"
        lines = [
            f"Significance Test: {self.test_name}",
            f"  Metric: {self.metric_name}",
            f"  {self.model_a_name}: {self.model_a_mean:.4f}",
            f"  {self.model_b_name}: {self.model_b_mean:.4f}",
            f"  Statistic: {self.statistic:.4f}",
            f"  p-value: {self.p_value:.6f}",
            f"  Effect size: {self.effect_size:.4f}",
            f"  Significant (alpha={self.alpha}): {sig_str}",
            f"  Better model: {winner} (+{abs(self.model_a_mean - self.model_b_mean):.4f})",
        ]
        return "\n".join(lines)


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "",
) -> SignificanceResult:
    """Paired t-test for comparing two models' per-user metric scores.

    Assumes that the differences between paired observations are
    approximately normally distributed. For large sample sizes (n > 30),
    this is usually satisfied by the Central Limit Theorem.

    Args:
        scores_a: Per-user metric scores for model A.
        scores_b: Per-user metric scores for model B (same users, same order).
        alpha: Significance level.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        metric_name: Name of the metric being compared.

    Returns:
        SignificanceResult with test outcome.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
        )

    statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    # Cohen's d for paired samples
    diff = scores_a - scores_b
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    return SignificanceResult(
        test_name="Paired t-test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(d),
        model_a_mean=float(np.mean(scores_a)),
        model_b_mean=float(np.mean(scores_b)),
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name=metric_name,
    )


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "",
) -> SignificanceResult:
    """Wilcoxon signed-rank test for comparing two models.

    Non-parametric alternative to the paired t-test. Does not assume
    normality of differences. Preferred when sample sizes are small
    or the normality assumption is questionable.

    Args:
        scores_a: Per-user metric scores for model A.
        scores_b: Per-user metric scores for model B (same users, same order).
        alpha: Significance level.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        metric_name: Name of the metric being compared.

    Returns:
        SignificanceResult with test outcome.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
        )

    diff = scores_a - scores_b

    # Wilcoxon requires at least some non-zero differences
    if np.all(diff == 0):
        return SignificanceResult(
            test_name="Wilcoxon signed-rank test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            model_a_mean=float(np.mean(scores_a)),
            model_b_mean=float(np.mean(scores_b)),
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name=metric_name,
        )

    result = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # Rank-biserial correlation as effect size for Wilcoxon
    n = len(diff[diff != 0])
    r = 1 - (2 * statistic) / (n * (n + 1) / 2) if n > 0 else 0.0

    return SignificanceResult(
        test_name="Wilcoxon signed-rank test",
        statistic=statistic,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(r),
        model_a_mean=float(np.mean(scores_a)),
        model_b_mean=float(np.mean(scores_b)),
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        metric_name=metric_name,
    )


def bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "",
) -> dict:
    """Bootstrap confidence interval for the difference between two models.

    Computes the confidence interval of (mean_a - mean_b) by resampling.
    If the CI does not contain 0, the difference is significant.

    Args:
        scores_a: Per-user metric scores for model A.
        scores_b: Per-user metric scores for model B.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        metric_name: Name of the metric being compared.

    Returns:
        Dict with ci_lower, ci_upper, mean_diff, significant, and metadata.
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    return {
        "test_name": f"Bootstrap {confidence*100:.0f}% CI",
        "metric_name": metric_name,
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "model_a_mean": float(np.mean(scores_a)),
        "model_b_mean": float(np.mean(scores_b)),
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": not (ci_lower <= 0 <= ci_upper),
        "n_bootstrap": n_bootstrap,
    }


def compute_per_user_metrics(
    model,
    test_data,
    metric: str = "ndcg@10",
    n_recommendations: int = 100,
) -> dict[int, float]:
    """Compute a metric per user (needed for paired significance tests).

    Args:
        model: Fitted recommendation model.
        test_data: Test InteractionData.
        metric: Metric string (e.g., "ndcg@10").
        n_recommendations: Number of items to rank per user.

    Returns:
        Dict mapping user_id to metric value.
    """
    from recllm.eval.metrics import (
        _parse_metric,
        hit_rate_at_k,
        mrr,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    test_arrays = test_data.to_numpy()
    user_ground_truth: dict[int, set[int]] = {}
    for u, i in zip(test_arrays["user_id"], test_arrays["item_id"], strict=False):
        user_ground_truth.setdefault(int(u), set()).add(int(i))

    user_seen: dict[int, set[int]] = {}
    if hasattr(model, "_user_interactions"):
        user_seen = dict(model._user_interactions.items())

    name, k = _parse_metric(metric)
    per_user: dict[int, float] = {}

    for user_id, relevant_items in user_ground_truth.items():
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

        if name == "ndcg":
            per_user[user_id] = ndcg_at_k(ranked_items, relevant_items, k or 10)
        elif name in ("hr", "hit_rate"):
            per_user[user_id] = hit_rate_at_k(ranked_items, relevant_items, k or 10)
        elif name == "mrr":
            per_user[user_id] = mrr(ranked_items, relevant_items)
        elif name in ("precision", "prec"):
            per_user[user_id] = precision_at_k(ranked_items, relevant_items, k or 10)
        elif name == "recall":
            per_user[user_id] = recall_at_k(ranked_items, relevant_items, k or 10)

    return per_user
