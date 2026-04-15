"""Tests for LLMExplainer module."""

import tempfile

import numpy as np
import pytest

from recllm.enhance.explainer import LLMExplainer
from recllm.llm.base import LLMClient


class MockLLMForExplaining(LLMClient):
    """Mock LLM that returns deterministic explanations."""

    def __init__(self):
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-explainer"

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return f"Mock explanation #{self.call_count} for the recommendation."

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8), dtype=np.float32)


@pytest.fixture
def mock_llm():
    return MockLLMForExplaining()


class TestLLMExplainer:
    def test_explain_conversational(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="conversational")
        result = explainer.explain(
            user_history=["The Matrix", "Inception"],
            recommended_item="Tenet",
            score=0.85,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_analytical(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="analytical")
        result = explainer.explain(
            user_history=["The Matrix"],
            recommended_item="Tenet",
            score=0.9,
        )
        assert isinstance(result, str)

    def test_explain_brief(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="brief")
        result = explainer.explain(
            user_history=["The Matrix"],
            recommended_item="Tenet",
        )
        assert isinstance(result, str)

    def test_memory_cache_avoids_redundant_calls(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="brief")

        explainer.explain(["The Matrix"], "Tenet", score=0.9)
        count_after_first = mock_llm.call_count

        explainer.explain(["The Matrix"], "Tenet", score=0.9)
        count_after_second = mock_llm.call_count

        assert count_after_second == count_after_first

    def test_disk_cache(self, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = LLMExplainer(mock_llm, style="brief", cache_dir=tmpdir)
            result1 = explainer.explain(["The Matrix"], "Tenet", score=0.9)

            # New explainer instance, same cache dir
            mock_llm2 = MockLLMForExplaining()
            explainer2 = LLMExplainer(mock_llm2, style="brief", cache_dir=tmpdir)
            result2 = explainer2.explain(["The Matrix"], "Tenet", score=0.9)

            assert result1 == result2
            assert mock_llm2.call_count == 0  # Loaded from disk

    def test_explain_batch(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="brief")
        results = explainer.explain_batch(
            user_history=["The Matrix", "Inception"],
            recommended_items=["Tenet", "Interstellar", "Arrival"],
            scores=[0.9, 0.85, 0.7],
        )
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_invalid_style_raises(self, mock_llm):
        with pytest.raises(ValueError, match="Unknown style"):
            LLMExplainer(mock_llm, style="invalid")

    def test_repr(self, mock_llm):
        explainer = LLMExplainer(mock_llm, style="analytical")
        assert "analytical" in repr(explainer)
        assert "mock-explainer" in repr(explainer)
