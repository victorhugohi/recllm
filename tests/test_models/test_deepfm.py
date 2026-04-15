"""Tests for DeepFM model."""

import numpy as np
import polars as pl
import pytest

from recllm.data.base import InteractionData
from recllm.models.deepfm import DeepFM


@pytest.fixture()
def small_data():
    """Minimal interaction data for testing."""
    df = pl.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        "item_id": [0, 1, 1, 2, 0, 3, 2, 3, 1, 4],
        "rating": [5.0, 3.0, 4.0, 2.0, 5.0, 1.0, 3.0, 4.0, 2.0, 5.0],
    })
    return InteractionData(interactions=df, metadata={"name": "test"})


def test_deepfm_fit_predict_bce(small_data):
    model = DeepFM(embed_dim=8, mlp_dims=[16, 8], loss="bce", device="cpu")
    model.fit(small_data, epochs=2, batch_size=4)
    scores = model.predict(np.array([0, 1]), np.array([0, 1]))
    assert scores.shape == (2,)
    assert not np.any(np.isnan(scores))


def test_deepfm_fit_predict_bpr(small_data):
    model = DeepFM(embed_dim=8, mlp_dims=[16, 8], loss="bpr", device="cpu")
    model.fit(small_data, epochs=2, batch_size=4)
    scores = model.predict(np.array([0, 1]), np.array([0, 1]))
    assert scores.shape == (2,)
    assert not np.any(np.isnan(scores))


def test_deepfm_recommend(small_data):
    model = DeepFM(embed_dim=8, mlp_dims=[16, 8], device="cpu")
    model.fit(small_data, epochs=2)
    recs = model.recommend(0, n=3)
    assert len(recs) == 3
    assert all(isinstance(item_id, int) and isinstance(score, float) for item_id, score in recs)


def test_deepfm_unknown_user(small_data):
    model = DeepFM(embed_dim=8, mlp_dims=[16, 8], device="cpu")
    model.fit(small_data, epochs=2)
    scores = model.predict(np.array([999]), np.array([0]))
    assert scores[0] == 0.0


def test_deepfm_repr():
    model = DeepFM(embed_dim=16, mlp_dims=[32, 16])
    assert isinstance(repr(model), str)
