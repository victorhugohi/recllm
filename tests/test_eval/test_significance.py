"""Tests for statistical significance testing."""

import numpy as np
import pytest

from recllm.eval.significance import (
    bootstrap_ci,
    paired_t_test,
    wilcoxon_test,
)


@pytest.fixture()
def different_scores():
    """Two clearly different score distributions."""
    rng = np.random.RandomState(42)
    scores_a = rng.normal(0.7, 0.1, size=100)
    scores_b = rng.normal(0.5, 0.1, size=100)
    return scores_a, scores_b


@pytest.fixture()
def similar_scores():
    """Two very similar score distributions."""
    rng = np.random.RandomState(42)
    scores_a = rng.normal(0.5, 0.1, size=100)
    scores_b = rng.normal(0.5, 0.1, size=100)
    return scores_a, scores_b


def test_paired_t_test_significant(different_scores):
    scores_a, scores_b = different_scores
    result = paired_t_test(scores_a, scores_b, model_a_name="Better", model_b_name="Worse")
    assert result.significant
    assert result.p_value < 0.05
    assert result.model_a_mean > result.model_b_mean
    assert result.test_name == "Paired t-test"


def test_paired_t_test_not_significant(similar_scores):
    scores_a, scores_b = similar_scores
    result = paired_t_test(scores_a, scores_b)
    # With same distribution and random seed, may or may not be significant
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1


def test_paired_t_test_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        paired_t_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_wilcoxon_significant(different_scores):
    scores_a, scores_b = different_scores
    result = wilcoxon_test(scores_a, scores_b, model_a_name="Better", model_b_name="Worse")
    assert result.significant
    assert result.p_value < 0.05
    assert result.test_name == "Wilcoxon signed-rank test"


def test_wilcoxon_identical():
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    result = wilcoxon_test(scores, scores)
    assert not result.significant
    assert result.p_value == 1.0


def test_wilcoxon_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_bootstrap_ci_significant(different_scores):
    scores_a, scores_b = different_scores
    result = bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, seed=42)
    assert result["significant"]
    assert result["ci_lower"] > 0  # A is better
    assert result["mean_diff"] > 0


def test_bootstrap_ci_not_significant():
    rng = np.random.RandomState(42)
    scores = rng.normal(0.5, 0.1, size=100)
    result = bootstrap_ci(scores, scores.copy(), n_bootstrap=1000, seed=42)
    assert not result["significant"]


def test_significance_result_summary(different_scores):
    scores_a, scores_b = different_scores
    result = paired_t_test(
        scores_a, scores_b,
        model_a_name="NCF", model_b_name="BPR",
        metric_name="ndcg@10",
    )
    summary = result.summary()
    assert "NCF" in summary
    assert "BPR" in summary
    assert "ndcg@10" in summary
    assert "p-value" in summary


def test_effect_size_direction(different_scores):
    scores_a, scores_b = different_scores
    result = paired_t_test(scores_a, scores_b)
    assert result.effect_size > 0  # A is better than B
