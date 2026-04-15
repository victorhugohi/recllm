"""Base model interface for all recommendation models."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from recllm.data.base import InteractionData


class BaseModel(ABC):
    """Abstract base class for all recommendation models.

    All models follow a scikit-learn-inspired API (ADR-003):
    fit() -> predict() -> recommend() -> evaluate().

    Subclasses must implement fit() and predict(). recommend() and
    evaluate() have default implementations that can be overridden.
    """

    @abstractmethod
    def fit(
        self,
        train_data: InteractionData,
        epochs: int = 20,
        val_data: InteractionData | None = None,
    ) -> Self:
        """Train the model on interaction data.

        Args:
            train_data: Training interactions.
            epochs: Number of training epochs (ignored by non-iterative models).
            val_data: Optional validation data for early stopping.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def predict(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict scores for given user-item pairs.

        Args:
            user_ids: Array of user IDs.
            item_ids: Array of item IDs (same length as user_ids).

        Returns:
            Array of predicted scores, same length as inputs.
        """
        ...

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Args:
            user_id: Target user ID.
            n: Number of recommendations.
            exclude_seen: Whether to exclude items the user already interacted with.
            seen_items: Explicit set of items to exclude. If None and exclude_seen
                is True, uses items from training data.

        Returns:
            List of (item_id, score) tuples, sorted by score descending.
        """
        if not hasattr(self, "_all_item_ids"):
            raise RuntimeError("Model must be fitted before calling recommend()")

        all_items = self._all_item_ids
        user_ids = np.full(len(all_items), user_id)
        scores = self.predict(user_ids, all_items)

        if exclude_seen and seen_items:
            for i, item_id in enumerate(all_items):
                if item_id in seen_items:
                    scores[i] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(all_items[i]), float(scores[i])) for i in top_indices]

    def evaluate(
        self,
        test_data: InteractionData,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate model on test data.

        Args:
            test_data: Test interactions.
            metrics: List of metric names. Default: ["ndcg@10", "hr@10"].

        Returns:
            Dict mapping metric names to values.
        """
        from recllm.eval.metrics import compute_metrics

        if metrics is None:
            metrics = ["ndcg@10", "hr@10"]

        return compute_metrics(self, test_data, metrics)

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: File path for saving.
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load model from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded model instance.
        """
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is {type(model)}, expected {cls}")
        return model
