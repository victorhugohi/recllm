"""Yelp Academic Dataset loader.

Supports the Yelp Open Dataset (academic subset) for business recommendation.
The dataset must be downloaded manually from https://www.yelp.com/dataset
due to Yelp's terms of service requiring agreement before download.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from recllm.data.base import InteractionData


class YelpDataset:
    """Yelp Academic Dataset loader.

    Loads and processes the Yelp Open Dataset for recommendation research.
    Because Yelp requires users to agree to terms before downloading,
    the raw JSON files must be provided locally.

    Args:
        data_dir: Directory containing yelp_academic_dataset_review.json
            (and optionally yelp_academic_dataset_business.json for features).
        cache_dir: Directory for processed parquet cache.

    Example:
        >>> data = YelpDataset("~/yelp_dataset").load(max_reviews=100000)
        >>> print(data.data.summary())
    """

    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: str | Path = "data_cache/yelp",
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: InteractionData | None = None

    def load(self, max_reviews: int | None = None) -> YelpDataset:
        """Load and parse the Yelp review dataset.

        Args:
            max_reviews: Maximum number of reviews to load (None = all).

        Returns:
            Self for method chaining.
        """
        parquet_path = self.cache_dir / "interactions.parquet"

        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            if max_reviews:
                df = df.head(max_reviews)
        else:
            df = self._parse_reviews(max_reviews)
            # Only cache if we loaded all reviews
            if max_reviews is None:
                df.write_parquet(parquet_path)

        item_features = self._load_business_features()

        self._data = InteractionData(
            interactions=df,
            item_features=item_features,
            metadata={
                "name": "Yelp",
                "source": "yelp_academic_dataset",
            },
        )
        return self

    def _parse_reviews(self, max_reviews: int | None) -> pl.DataFrame:
        """Parse yelp_academic_dataset_review.json into a DataFrame."""
        review_path = self.data_dir / "yelp_academic_dataset_review.json"
        if not review_path.exists():
            raise FileNotFoundError(
                f"Review file not found: {review_path}\n"
                "Download the Yelp dataset from https://www.yelp.com/dataset "
                "and place yelp_academic_dataset_review.json in the data directory."
            )

        user_ids = []
        item_ids = []
        ratings = []
        timestamps = []

        with open(review_path, encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Parsing Yelp reviews")):
                if max_reviews and i >= max_reviews:
                    break
                review = json.loads(line)
                user_ids.append(review["user_id"])
                item_ids.append(review["business_id"])
                ratings.append(float(review["stars"]))
                # Yelp uses date strings, convert to simple ordinal
                date_str = review.get("date", "2020-01-01")
                timestamps.append(
                    int(date_str.replace("-", ""))
                )

        df = pl.DataFrame({
            "user_id_str": user_ids,
            "item_id_str": item_ids,
            "rating": ratings,
            "timestamp": timestamps,
        })

        # Encode string IDs to integers
        unique_users = df["user_id_str"].unique().sort().to_list()
        unique_items = df["item_id_str"].unique().sort().to_list()
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}

        df = df.with_columns(
            pl.col("user_id_str").replace_strict(user_map).cast(pl.Int64).alias("user_id"),
            pl.col("item_id_str").replace_strict(item_map).cast(pl.Int64).alias("item_id"),
        ).select(["user_id", "item_id", "rating", "timestamp"])

        return df

    def _load_business_features(self) -> pl.DataFrame | None:
        """Load business metadata if available."""
        biz_path = self.data_dir / "yelp_academic_dataset_business.json"
        if not biz_path.exists():
            return None

        items = []
        with open(biz_path, encoding="utf-8") as f:
            for line in f:
                biz = json.loads(line)
                items.append({
                    "item_id_str": biz["business_id"],
                    "name": biz.get("name", ""),
                    "city": biz.get("city", ""),
                    "state": biz.get("state", ""),
                    "categories": biz.get("categories", ""),
                    "stars": biz.get("stars", 0.0),
                    "review_count": biz.get("review_count", 0),
                })

        return pl.DataFrame(items)

    @property
    def data(self) -> InteractionData:
        """Access loaded data."""
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        return self._data

    def preprocess(self, min_user: int = 5, min_item: int = 5) -> YelpDataset:
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
