"""Amazon Reviews dataset loader.

Supports the Amazon Reviews 2023 dataset (McAuley Lab).
Uses the smaller "5-core" subsets where each user and item has >= 5 reviews.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from recllm.data.base import InteractionData

# Category -> URL mapping for Amazon Reviews 2023 (5-core)
_AMAZON_CATEGORIES = {
    "beauty": "All_Beauty",
    "books": "Books",
    "cds": "CDs_and_Vinyl",
    "clothing": "Clothing_Shoes_and_Jewelry",
    "digital_music": "Digital_Music",
    "electronics": "Electronics",
    "games": "Toys_and_Games",
    "grocery": "Grocery_and_Gourmet_Food",
    "home": "Home_and_Kitchen",
    "movies": "Movies_and_TV",
    "music_instruments": "Musical_Instruments",
    "office": "Office_Products",
    "pet": "Pet_Supplies",
    "sports": "Sports_and_Outdoors",
    "tools": "Tools_and_Home_Improvement",
    "video_games": "Video_Games",
}

_BASE_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories"
)


class AmazonReviews:
    """Amazon Reviews dataset loader.

    Downloads and parses Amazon product review datasets from the
    McAuley Lab (UCSD). Uses the 2023 release with per-category
    JSONL files.

    Args:
        category: Product category (e.g., "books", "electronics", "movies").
            See AmazonReviews.available_categories() for full list.
        cache_dir: Directory for downloaded files.

    Example:
        >>> data = AmazonReviews("digital_music").load()
        >>> print(data.summary())
    """

    def __init__(
        self,
        category: str = "digital_music",
        cache_dir: str | Path = "data_cache",
    ):
        category = category.lower()
        if category not in _AMAZON_CATEGORIES:
            available = ", ".join(sorted(_AMAZON_CATEGORIES.keys()))
            raise ValueError(
                f"Unknown category: {category!r}. "
                f"Available: {available}"
            )
        self.category = category
        self._amazon_name = _AMAZON_CATEGORIES[category]
        self.cache_dir = Path(cache_dir) / "amazon" / category
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: InteractionData | None = None

    @staticmethod
    def available_categories() -> list[str]:
        """Return list of available Amazon review categories."""
        return sorted(_AMAZON_CATEGORIES.keys())

    def load(self, max_reviews: int | None = None) -> AmazonReviews:
        """Download and load the dataset.

        Args:
            max_reviews: Maximum number of reviews to load (for quick testing).
                None loads all.

        Returns:
            Self for method chaining.
        """
        parquet_path = self.cache_dir / "interactions.parquet"

        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            if max_reviews:
                df = df.head(max_reviews)
        else:
            df = self._download_and_parse(max_reviews)
            df.write_parquet(parquet_path)

        self._data = InteractionData(
            interactions=df,
            metadata={
                "name": f"Amazon-{self.category}",
                "source": "amazon_reviews_2023",
                "category": self.category,
            },
        )
        return self

    def _download_and_parse(self, max_reviews: int | None) -> pl.DataFrame:
        """Download JSONL.gz file and parse into DataFrame."""
        url = f"{_BASE_URL}/{self._amazon_name}.jsonl.gz"
        gz_path = self.cache_dir / f"{self._amazon_name}.jsonl.gz"

        if not gz_path.exists():
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(gz_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"Downloading Amazon {self.category}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Parse JSONL
        user_ids = []
        item_ids = []
        ratings = []
        timestamps = []

        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_reviews and i >= max_reviews:
                    break
                review = json.loads(line)
                user_ids.append(review.get("user_id", ""))
                item_ids.append(review.get("parent_asin", review.get("asin", "")))
                ratings.append(float(review.get("rating", 0.0)))
                timestamps.append(int(review.get("timestamp", 0)))

        df = pl.DataFrame({
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": ratings,
            "timestamp": timestamps,
        })

        # Encode string IDs to integers
        unique_users = df["user_id"].unique().sort().to_list()
        unique_items = df["item_id"].unique().sort().to_list()
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}

        df = df.with_columns(
            pl.col("user_id").replace_strict(user_map).cast(pl.Int64),
            pl.col("item_id").replace_strict(item_map).cast(pl.Int64),
        )

        return df

    @property
    def data(self) -> InteractionData:
        """Access loaded data."""
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        return self._data

    def preprocess(self, min_user: int = 5, min_item: int = 5) -> AmazonReviews:
        """Filter users and items with fewer than min interactions.

        Args:
            min_user: Minimum interactions per user.
            min_item: Minimum interactions per item.

        Returns:
            Self for method chaining.
        """
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        self._data = self._data.filter_by_min_interactions(min_user, min_item)
        return self
