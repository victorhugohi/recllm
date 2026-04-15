"""Tests for the FeatureEnhancer module."""

import tempfile

import numpy as np
import polars as pl
import pytest

from recllm.data.base import InteractionData
from recllm.enhance.feature_enhancer import EnhancedFeatures, FeatureEnhancer
from recllm.llm.base import LLMClient


class MockLLMClient(LLMClient):
    """Deterministic mock LLM for testing."""

    def __init__(self, embedding_dim: int = 8):
        self._embedding_dim = embedding_dim
        self.generate_calls: list[str] = []
        self.embed_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def generate(self, prompt: str, **kwargs) -> str:
        self.generate_calls.append(prompt)
        return f"Generated description for: {prompt[:50]}"

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(texts)
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), self._embedding_dim).astype(np.float32)


@pytest.fixture
def sample_data():
    interactions = pl.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 20, 30, 10],
        "rating": [5.0, 4.0, 3.0, 5.0, 4.0],
    })
    item_features = pl.DataFrame({
        "item_id": [10, 20, 30],
        "title": ["The Matrix", "Inception", "Interstellar"],
    })
    return InteractionData(
        interactions=interactions,
        item_features=item_features,
        metadata={"name": "test"},
    )


@pytest.fixture
def mock_llm():
    return MockLLMClient(embedding_dim=8)


class TestFeatureEnhancer:
    def test_enhance_items_generates_texts(self, sample_data, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = FeatureEnhancer(mock_llm, cache_dir=tmpdir, batch_size=2)
            result = enhancer.enhance_items(sample_data, feature_col="title", embed=False)

            assert isinstance(result, EnhancedFeatures)
            assert result.entity_type == "item"
            assert result.n_entities == 3
            assert 10 in result.texts
            assert 20 in result.texts
            assert 30 in result.texts

    def test_enhance_items_with_embeddings(self, sample_data, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = FeatureEnhancer(mock_llm, cache_dir=tmpdir)
            result = enhancer.enhance_items(sample_data, embed=True)

            assert result.embeddings is not None
            assert result.embedding_dim == 8
            arr = result.to_numpy()
            assert arr.shape == (3, 8)

    def test_enhance_users(self, sample_data, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = FeatureEnhancer(mock_llm, cache_dir=tmpdir)
            result = enhancer.enhance_users(sample_data, embed=False)

            assert result.entity_type == "user"
            assert result.n_entities == 3
            assert 1 in result.texts
            assert 2 in result.texts
            assert 3 in result.texts

    def test_caching_avoids_redundant_calls(self, sample_data, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = FeatureEnhancer(mock_llm, cache_dir=tmpdir, batch_size=10)

            # First call: generates all
            enhancer.enhance_items(sample_data, embed=False)
            first_call_count = len(mock_llm.generate_calls)

            # Second call: all cached
            enhancer.enhance_items(sample_data, embed=False)
            second_call_count = len(mock_llm.generate_calls)

            assert second_call_count == first_call_count  # no new calls

    def test_enhanced_features_to_numpy_ordering(self, sample_data, mock_llm):
        with tempfile.TemporaryDirectory() as tmpdir:
            enhancer = FeatureEnhancer(mock_llm, cache_dir=tmpdir)
            result = enhancer.enhance_items(sample_data, embed=True)

            # Specific ordering
            arr = result.to_numpy(entity_ids=[30, 10, 20])
            assert arr.shape == (3, 8)

    def test_enhanced_features_repr(self):
        features = EnhancedFeatures(
            entity_type="item",
            texts={1: "test", 2: "test2"},
            model_name="mock",
        )
        assert "item" in repr(features)
        assert "n=2" in repr(features)


class TestPipeline:
    def test_pipeline_runs_with_popularity(self, sample_data):
        from recllm.models.popularity import PopularityBaseline
        from recllm.pipeline.recommendation import RecommendationPipeline

        model = PopularityBaseline()
        pipeline = RecommendationPipeline(model, seed=42)
        result = pipeline.run(sample_data, epochs=1)

        assert "ndcg@10" in result.metrics
        assert "hr@10" in result.metrics
        assert result.timing["train"] >= 0
        assert result.timing["evaluate"] >= 0

    def test_pipeline_summary(self, sample_data):
        from recllm.models.popularity import PopularityBaseline
        from recllm.pipeline.recommendation import RecommendationPipeline

        model = PopularityBaseline()
        pipeline = RecommendationPipeline(model, seed=42)
        result = pipeline.run(sample_data, epochs=1)
        summary = result.summary()

        assert "PopularityBaseline" in summary
        assert "ndcg@10" in summary

    def test_pipeline_compare(self, sample_data):
        from recllm.models.popularity import PopularityBaseline
        from recllm.pipeline.recommendation import RecommendationPipeline

        models = {
            "pop_1": PopularityBaseline(),
            "pop_2": PopularityBaseline(),
        }
        pipeline = RecommendationPipeline(models["pop_1"], seed=42)
        results = pipeline.compare(models, sample_data, epochs=1)

        assert "pop_1" in results
        assert "pop_2" in results
