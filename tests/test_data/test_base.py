"""Tests for InteractionData and data utilities."""

import numpy as np
import polars as pl

from recllm.data.base import InteractionData
from recllm.data.splitting import random_split, temporal_split


def _make_sample_data(
    n_users: int = 50, n_items: int = 100, n_interactions: int = 500,
) -> InteractionData:
    """Create a small synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    user_ids = rng.integers(0, n_users, size=n_interactions)
    item_ids = rng.integers(0, n_items, size=n_interactions)
    ratings = rng.uniform(1, 5, size=n_interactions).round(1)
    timestamps = np.sort(rng.integers(1_000_000, 2_000_000, size=n_interactions))

    df = pl.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })
    return InteractionData(interactions=df, metadata={"name": "test"})


class TestInteractionData:
    def test_basic_properties(self):
        data = _make_sample_data()
        assert data.n_interactions == 500
        assert data.n_users <= 50
        assert data.n_items <= 100
        assert 0 < data.density <= 1

    def test_to_numpy(self):
        data = _make_sample_data()
        arrays = data.to_numpy()
        assert "user_id" in arrays
        assert "item_id" in arrays
        assert "rating" in arrays
        assert len(arrays["user_id"]) == 500

    def test_filter_by_min_interactions(self):
        data = _make_sample_data(n_users=50, n_items=100, n_interactions=500)
        filtered = data.filter_by_min_interactions(min_user=3, min_item=3)
        # Should have fewer or equal interactions
        assert filtered.n_interactions <= data.n_interactions

    def test_encode_ids(self):
        data = _make_sample_data()
        encoded, user_map, item_map = data.encode_ids()
        # Encoded IDs should be contiguous from 0
        user_ids = encoded.interactions["user_id"].unique().sort().to_list()
        assert user_ids[0] == 0
        assert user_ids[-1] == len(user_ids) - 1

    def test_summary(self):
        data = _make_sample_data()
        summary = data.summary()
        assert "test" in summary
        assert "Users" in summary

    def test_repr(self):
        data = _make_sample_data()
        r = repr(data)
        assert "InteractionData" in r


class TestSplitting:
    def test_random_split(self):
        data = _make_sample_data()
        result = random_split(data, test_ratio=0.1, val_ratio=0.1, seed=42)
        assert result.train is not None
        assert result.val is not None
        assert result.test is not None
        total = result.train.n_interactions + result.val.n_interactions + result.test.n_interactions
        assert total == data.n_interactions

    def test_temporal_split(self):
        data = _make_sample_data()
        result = temporal_split(data, test_ratio=0.1, val_ratio=0.1)
        assert result.train is not None
        assert result.test is not None
        # Test set should have later timestamps than train set
        train_max_ts = result.train.interactions["timestamp"].max()
        test_min_ts = result.test.interactions["timestamp"].min()
        assert test_min_ts >= train_max_ts

    def test_random_split_reproducibility(self):
        data = _make_sample_data()
        r1 = random_split(data, seed=42)
        r2 = random_split(data, seed=42)
        assert r1.train.interactions.equals(r2.train.interactions)
