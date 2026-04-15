"""Data loading, preprocessing, and splitting for recommendation datasets."""

from recllm.data.base import InteractionData
from recllm.data.movielens import MovieLens
from recllm.data.splitting import leave_one_out_split, random_split, temporal_split

__all__ = [
    "InteractionData",
    "MovieLens",
    "temporal_split",
    "random_split",
    "leave_one_out_split",
]


# Lazy imports for optional dataset loaders
def __getattr__(name: str):
    if name == "AmazonReviews":
        from recllm.data.amazon import AmazonReviews
        return AmazonReviews
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
