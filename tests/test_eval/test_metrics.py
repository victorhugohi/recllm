"""Tests for evaluation metrics."""

import math

import numpy as np
import pytest

from recllm.eval.metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    _parse_metric,
)


class TestNDCG:
    def test_perfect_ranking(self):
        ranked = np.array([1, 2, 3])
        ground_truth = {1, 2, 3}
        assert ndcg_at_k(ranked, ground_truth, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        ranked = np.array([4, 5, 6])
        ground_truth = {1, 2, 3}
        assert ndcg_at_k(ranked, ground_truth, k=3) == 0.0

    def test_partial_hit_first_position(self):
        ranked = np.array([1, 5, 6])
        ground_truth = {1}
        assert ndcg_at_k(ranked, ground_truth, k=3) == pytest.approx(1.0)

    def test_partial_hit_later_position(self):
        ranked = np.array([5, 6, 1])
        ground_truth = {1}
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert ndcg_at_k(ranked, ground_truth, k=3) == pytest.approx(expected)


class TestHitRate:
    def test_hit(self):
        ranked = np.array([3, 1, 2])
        assert hit_rate_at_k(ranked, {1}, k=3) == 1.0

    def test_miss(self):
        ranked = np.array([3, 4, 5])
        assert hit_rate_at_k(ranked, {1}, k=3) == 0.0

    def test_hit_at_k_boundary(self):
        ranked = np.array([3, 4, 1, 5])
        assert hit_rate_at_k(ranked, {1}, k=2) == 0.0
        assert hit_rate_at_k(ranked, {1}, k=3) == 1.0


class TestMRR:
    def test_first_position(self):
        ranked = np.array([1, 2, 3])
        assert mrr(ranked, {1}) == pytest.approx(1.0)

    def test_second_position(self):
        ranked = np.array([2, 1, 3])
        assert mrr(ranked, {1}) == pytest.approx(0.5)

    def test_no_relevant(self):
        ranked = np.array([2, 3, 4])
        assert mrr(ranked, {1}) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        ranked = np.array([1, 2, 3])
        assert precision_at_k(ranked, {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_half_relevant(self):
        ranked = np.array([1, 4, 2, 5])
        assert precision_at_k(ranked, {1, 2, 3}, k=4) == pytest.approx(0.5)

    def test_none_relevant(self):
        ranked = np.array([4, 5, 6])
        assert precision_at_k(ranked, {1, 2, 3}, k=3) == 0.0


class TestRecall:
    def test_full_recall(self):
        ranked = np.array([1, 2, 3, 4])
        assert recall_at_k(ranked, {1, 2}, k=4) == pytest.approx(1.0)

    def test_partial_recall(self):
        ranked = np.array([1, 4, 5])
        assert recall_at_k(ranked, {1, 2}, k=3) == pytest.approx(0.5)

    def test_empty_ground_truth(self):
        ranked = np.array([1, 2])
        assert recall_at_k(ranked, set(), k=2) == 0.0


class TestParseMetric:
    def test_with_k(self):
        assert _parse_metric("ndcg@10") == ("ndcg", 10)
        assert _parse_metric("hr@5") == ("hr", 5)

    def test_without_k(self):
        assert _parse_metric("mrr") == ("mrr", None)

    def test_case_insensitive(self):
        assert _parse_metric("NDCG@10") == ("ndcg", 10)
