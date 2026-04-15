"""Tests for LLMRanker module."""

import numpy as np
import pytest

from recllm.enhance.ranker import LLMRanker
from recllm.llm.base import LLMClient


class MockLLMForRanking(LLMClient):
    """Mock LLM that returns deterministic ranking responses."""

    @property
    def model_name(self) -> str:
        return "mock-ranker"

    def generate(self, prompt: str, **kwargs) -> str:
        if "Scale of 1-10" in prompt or "scale of 1-10" in prompt:
            # Pointwise: return score based on candidate position
            if "Interstellar" in prompt:
                return "9"
            elif "Tenet" in prompt:
                return "7"
            elif "Titanic" in prompt:
                return "4"
            return "5"
        elif "rank the following" in prompt.lower():
            # Listwise: return ordering
            return "1,3,2,4"
        elif "Which item would this user prefer" in prompt:
            # Pairwise: A always wins
            return "A"
        return "5"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8), dtype=np.float32)


@pytest.fixture
def mock_llm():
    return MockLLMForRanking()


@pytest.fixture
def user_history():
    return ["The Matrix", "Inception", "Blade Runner 2049"]


@pytest.fixture
def candidates():
    return ["Tenet", "Titanic", "Interstellar", "The Notebook"]


class TestLLMRanker:
    def test_pointwise_ranking(self, mock_llm, user_history, candidates):
        ranker = LLMRanker(mock_llm, mode="pointwise")
        result = ranker.rerank(user_history, candidates)

        assert len(result) == 4
        # Interstellar should score highest (9/10)
        assert result[0][0] == 2  # index of Interstellar
        # All scores in [0, 1]
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_listwise_ranking(self, mock_llm, user_history, candidates):
        ranker = LLMRanker(mock_llm, mode="listwise")
        result = ranker.rerank(user_history, candidates)

        assert len(result) == 4
        # First in ranking gets highest score
        assert result[0][1] > result[-1][1]

    def test_pairwise_ranking(self, mock_llm, user_history, candidates):
        ranker = LLMRanker(mock_llm, mode="pairwise")
        result = ranker.rerank(user_history, candidates)

        assert len(result) == 4
        # Since A always wins, first item should dominate
        assert result[0][0] == 0  # index 0 = "Tenet" wins all pairs

    def test_custom_candidate_ids(self, mock_llm, user_history, candidates):
        ranker = LLMRanker(mock_llm, mode="pointwise")
        result = ranker.rerank(
            user_history, candidates, candidate_ids=[100, 200, 300, 400]
        )
        # IDs should be the custom ones
        ids = [r[0] for r in result]
        assert set(ids) == {100, 200, 300, 400}

    def test_max_candidates_truncation(self, mock_llm, user_history):
        many_candidates = [f"Movie {i}" for i in range(50)]
        ranker = LLMRanker(mock_llm, mode="pointwise", max_candidates=5)
        result = ranker.rerank(user_history, many_candidates)
        assert len(result) == 5

    def test_invalid_mode_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown mode"):
            LLMRanker(mock_llm, mode="invalid")

    def test_repr(self, mock_llm):
        ranker = LLMRanker(mock_llm, mode="listwise")
        assert "listwise" in repr(ranker)
        assert "mock-ranker" in repr(ranker)
