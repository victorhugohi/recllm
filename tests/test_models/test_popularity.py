"""Tests for PopularityBaseline model."""

import numpy as np
import polars as pl

from recllm.data.base import InteractionData
from recllm.models.popularity import PopularityBaseline


def _make_sample_data():
    """Create a small dataset where item 0 is most popular."""
    # Item 0 appears 10 times, item 1 appears 5 times, etc.
    user_ids = list(range(10)) + list(range(5)) + list(range(3))
    item_ids = [0] * 10 + [1] * 5 + [2] * 3
    ratings = [5.0] * len(user_ids)
    timestamps = list(range(len(user_ids)))

    df = pl.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })
    return InteractionData(interactions=df, metadata={"name": "test"})


class TestPopularityBaseline:
    def test_fit_and_predict(self):
        data = _make_sample_data()
        model = PopularityBaseline()
        model.fit(data)

        # Item 0 should have highest score
        scores = model.predict(
            np.array([0, 0, 0]),
            np.array([0, 1, 2]),
        )
        assert scores[0] > scores[1] > scores[2]

    def test_recommend(self):
        data = _make_sample_data()
        model = PopularityBaseline().fit(data)
        recs = model.recommend(user_id=0, n=3, exclude_seen=False)
        assert len(recs) == 3
        # First recommendation should be item 0 (most popular)
        assert recs[0][0] == 0

    def test_predict_unknown_item(self):
        data = _make_sample_data()
        model = PopularityBaseline().fit(data)
        scores = model.predict(np.array([0]), np.array([999]))
        assert scores[0] == 0.0

    def test_method_chaining(self):
        data = _make_sample_data()
        model = PopularityBaseline().fit(data)
        assert isinstance(model, PopularityBaseline)
