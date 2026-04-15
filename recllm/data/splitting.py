"""Dataset splitting strategies for recommendation evaluation."""

from __future__ import annotations

import polars as pl

from recllm.data.base import InteractionData


def random_split(
    data: InteractionData,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> InteractionData:
    """Split interactions randomly into train/val/test.

    Args:
        data: Source InteractionData.
        test_ratio: Fraction of interactions for test set.
        val_ratio: Fraction of interactions for validation set.
        seed: Random seed for reproducibility.

    Returns:
        InteractionData with train, val, test attributes populated.
    """
    df = data.interactions.sample(fraction=1.0, seed=seed, shuffle=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_df = df.head(n_test)
    val_df = df.slice(n_test, n_val)
    train_df = df.slice(n_test + n_val)

    result = InteractionData(
        interactions=data.interactions,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "split": "random", "seed": seed},
    )
    result.train = InteractionData(
        interactions=train_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "train"},
    )
    result.val = InteractionData(
        interactions=val_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "val"},
    )
    result.test = InteractionData(
        interactions=test_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "test"},
    )
    return result


def temporal_split(
    data: InteractionData,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
) -> InteractionData:
    """Split interactions by timestamp (earlier = train, later = test).

    Requires a 'timestamp' column in interactions.

    Args:
        data: Source InteractionData with timestamp column.
        test_ratio: Fraction of interactions for test set (most recent).
        val_ratio: Fraction of interactions for validation set.

    Returns:
        InteractionData with train, val, test attributes populated.
    """
    if "timestamp" not in data.interactions.columns:
        raise ValueError("Temporal split requires a 'timestamp' column in interactions.")

    df = data.interactions.sort("timestamp")
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    train_df = df.head(n - n_test - n_val)
    val_df = df.slice(n - n_test - n_val, n_val)
    test_df = df.tail(n_test)

    result = InteractionData(
        interactions=data.interactions,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "split": "temporal"},
    )
    result.train = InteractionData(
        interactions=train_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "train"},
    )
    result.val = InteractionData(
        interactions=val_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "val"},
    )
    result.test = InteractionData(
        interactions=test_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "test"},
    )
    return result


def leave_one_out_split(
    data: InteractionData,
    time_order: bool = True,
) -> InteractionData:
    """Leave-one-out split: last interaction per user for test, second-to-last for val.

    Args:
        data: Source InteractionData.
        time_order: If True (default), use temporal ordering within each user.
            Requires 'timestamp' column. If False, random selection.

    Returns:
        InteractionData with train, val, test attributes populated.
    """
    if time_order and "timestamp" not in data.interactions.columns:
        raise ValueError("Leave-one-out with time_order requires 'timestamp' column.")

    df = data.interactions.sort(["user_id", "timestamp"]) if time_order else data.interactions

    # Assign row number within each user (descending for most recent first)
    df = df.with_columns(
        pl.col("item_id")
        .cum_count()
        .over("user_id")
        .alias("_rank")
    )
    max_ranks = df.group_by("user_id").agg(pl.col("_rank").max().alias("_max_rank"))
    df = df.join(max_ranks, on="user_id")

    # Last interaction = test, second-to-last = val, rest = train
    test_df = df.filter(pl.col("_rank") == pl.col("_max_rank")).drop(["_rank", "_max_rank"])
    val_df = df.filter(pl.col("_rank") == pl.col("_max_rank") - 1).drop(["_rank", "_max_rank"])
    train_df = df.filter(pl.col("_rank") < pl.col("_max_rank") - 1).drop(["_rank", "_max_rank"])

    result = InteractionData(
        interactions=data.interactions,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "split": "leave_one_out"},
    )
    result.train = InteractionData(
        interactions=train_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "train"},
    )
    result.val = InteractionData(
        interactions=val_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "val"},
    )
    result.test = InteractionData(
        interactions=test_df,
        user_features=data.user_features,
        item_features=data.item_features,
        metadata={**data.metadata, "partition": "test"},
    )
    return result
