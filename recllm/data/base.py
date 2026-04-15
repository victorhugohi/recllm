"""Core data structures for RecLLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class InteractionData:
    """Container for recommendation dataset with optional features.

    Stores user-item interactions and optional side information using
    Polars DataFrames internally (ADR-001). Provides conversion methods
    for pandas and NumPy interoperability.

    Attributes:
        interactions: DataFrame with columns [user_id, item_id, rating, timestamp].
            rating and timestamp are optional.
        user_features: Optional DataFrame with user_id + feature columns.
        item_features: Optional DataFrame with item_id + feature columns.
        metadata: Dataset metadata (name, split info, checksums, etc.).
    """

    interactions: pl.DataFrame
    user_features: pl.DataFrame | None = None
    item_features: pl.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    # Split references (populated after splitting)
    train: InteractionData | None = field(default=None, repr=False)
    val: InteractionData | None = field(default=None, repr=False)
    test: InteractionData | None = field(default=None, repr=False)

    @property
    def n_users(self) -> int:
        """Number of unique users."""
        return self.interactions["user_id"].n_unique()

    @property
    def n_items(self) -> int:
        """Number of unique items."""
        return self.interactions["item_id"].n_unique()

    @property
    def n_interactions(self) -> int:
        """Total number of interactions."""
        return len(self.interactions)

    @property
    def density(self) -> float:
        """Interaction matrix density as a fraction."""
        return self.n_interactions / (self.n_users * self.n_items)

    @property
    def user_ids(self) -> np.ndarray:
        """Array of unique user IDs."""
        return self.interactions["user_id"].unique().sort().to_numpy()

    @property
    def item_ids(self) -> np.ndarray:
        """Array of unique item IDs."""
        return self.interactions["item_id"].unique().sort().to_numpy()

    def to_pandas(self) -> pd.DataFrame:
        """Convert interactions to pandas DataFrame (zero-copy via Arrow)."""
        return self.interactions.to_pandas()

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert interactions to NumPy arrays.

        Returns:
            Dict with keys 'user_id', 'item_id', and optionally 'rating', 'timestamp'.
        """
        result = {
            "user_id": self.interactions["user_id"].to_numpy(),
            "item_id": self.interactions["item_id"].to_numpy(),
        }
        if "rating" in self.interactions.columns:
            result["rating"] = self.interactions["rating"].to_numpy()
        if "timestamp" in self.interactions.columns:
            result["timestamp"] = self.interactions["timestamp"].to_numpy()
        return result

    def filter_by_min_interactions(
        self, min_user: int = 5, min_item: int = 5
    ) -> InteractionData:
        """Filter users and items with fewer than min interactions.

        Iteratively removes users and items until convergence.

        Args:
            min_user: Minimum interactions per user.
            min_item: Minimum interactions per item.

        Returns:
            New InteractionData with filtered interactions.
        """
        df = self.interactions
        prev_len = -1
        while len(df) != prev_len:
            prev_len = len(df)
            # Filter users
            user_counts = df.group_by("user_id").len()
            valid_users = user_counts.filter(pl.col("len") >= min_user)["user_id"]
            df = df.filter(pl.col("user_id").is_in(valid_users.to_list()))
            # Filter items
            item_counts = df.group_by("item_id").len()
            valid_items = item_counts.filter(pl.col("len") >= min_item)["item_id"]
            df = df.filter(pl.col("item_id").is_in(valid_items.to_list()))

        metadata = {**self.metadata, "filtered": True, "min_user": min_user, "min_item": min_item}

        # Filter features to match
        user_features = self.user_features
        if user_features is not None:
            valid_user_ids = df["user_id"].unique()
            user_features = user_features.filter(pl.col("user_id").is_in(valid_user_ids.to_list()))

        item_features = self.item_features
        if item_features is not None:
            valid_item_ids = df["item_id"].unique()
            item_features = item_features.filter(pl.col("item_id").is_in(valid_item_ids.to_list()))

        return InteractionData(
            interactions=df,
            user_features=user_features,
            item_features=item_features,
            metadata=metadata,
        )

    def encode_ids(self) -> tuple[InteractionData, dict, dict]:
        """Re-encode user and item IDs to contiguous integers starting from 0.

        Returns:
            Tuple of (encoded InteractionData, user_id_map, item_id_map).
            Maps are {original_id: encoded_id}.
        """
        unique_users = self.interactions["user_id"].unique().sort().to_list()
        unique_items = self.interactions["item_id"].unique().sort().to_list()

        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}

        df = self.interactions.with_columns(
            pl.col("user_id").replace_strict(user_map).alias("user_id"),
            pl.col("item_id").replace_strict(item_map).alias("item_id"),
        )

        return (
            InteractionData(
                interactions=df,
                user_features=self.user_features,
                item_features=self.item_features,
                metadata={**self.metadata, "encoded": True},
            ),
            user_map,
            item_map,
        )

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [
            f"InteractionData: {self.metadata.get('name', 'unknown')}",
            f"  Users: {self.n_users:,}",
            f"  Items: {self.n_items:,}",
            f"  Interactions: {self.n_interactions:,}",
            f"  Density: {self.density:.4%}",
        ]
        if self.user_features is not None:
            lines.append(f"  User features: {len(self.user_features.columns) - 1} columns")
        if self.item_features is not None:
            lines.append(f"  Item features: {len(self.item_features.columns) - 1} columns")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InteractionData(users={self.n_users}, items={self.n_items}, "
            f"interactions={self.n_interactions}, density={self.density:.4%})"
        )
