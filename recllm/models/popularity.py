"""Popularity baseline model."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from recllm.data.base import InteractionData
from recllm.models.base import BaseModel


class PopularityBaseline(BaseModel):
    """Non-personalized popularity-based recommendation baseline.

    Recommends items ranked by their interaction frequency in the
    training data. All users receive the same ranking. This serves
    as a lower bound for personalized models.

    Example:
        >>> model = PopularityBaseline()
        >>> model.fit(train_data)
        >>> recs = model.recommend(user_id=1, n=10)
    """

    def __init__(self) -> None:
        self._item_scores: dict[int, float] = {}
        self._all_item_ids: np.ndarray = np.array([])
        self._max_count: float = 1.0

    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 1,
        val_data: InteractionData | None = None,
    ) -> Self:
        """Compute item popularity from training interactions.

        Args:
            train_data: Training interactions.
            epochs: Ignored (no iterative training).
            val_data: Ignored.

        Returns:
            Self.
        """
        counts = (
            train_data.interactions.group_by("item_id")
            .len()
            .sort("len", descending=True)
        )

        item_ids = counts["item_id"].to_numpy()
        interaction_counts = counts["len"].to_numpy().astype(float)
        self._max_count = float(interaction_counts.max()) if len(interaction_counts) > 0 else 1.0

        # Normalize scores to [0, 1]
        self._item_scores = {
            int(item_ids[i]): interaction_counts[i] / self._max_count
            for i in range(len(item_ids))
        }
        self._all_item_ids = item_ids
        return self

    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        """Return popularity scores (same for all users).

        Args:
            user_ids: Array of user IDs (ignored, non-personalized).
            item_ids: Array of item IDs.

        Returns:
            Array of popularity scores in [0, 1].
        """
        return np.array(
            [self._item_scores.get(int(iid), 0.0) for iid in item_ids],
            dtype=np.float64,
        )
