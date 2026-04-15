"""MovieLens dataset loader."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from recllm.data.base import InteractionData
from recllm.data.splitting import random_split, temporal_split, leave_one_out_split

# Dataset URLs and metadata
_MOVIELENS_CONFIGS = {
    "100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ratings_file": "ml-100k/u.data",
        "separator": "\t",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
    },
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ratings_file": "ml-1m/ratings.dat",
        "separator": "::",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
    },
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "ratings_file": "ml-25m/ratings.csv",
        "separator": ",",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
    },
}


class MovieLens:
    """MovieLens dataset loader with download, caching, and splitting.

    Supports MovieLens 100K, 1M, and 25M variants.

    Example:
        >>> data = MovieLens("1m").load().split(strategy="temporal")
        >>> print(data.train.n_interactions)
    """

    def __init__(
        self,
        version: str = "1m",
        data_dir: str = "./data",
    ):
        """Initialize MovieLens loader.

        Args:
            version: Dataset version. One of "100k", "1m", "25m".
            data_dir: Directory for downloading and caching data.
        """
        if version not in _MOVIELENS_CONFIGS:
            raise ValueError(
                f"Unknown MovieLens version '{version}'. "
                f"Choose from: {list(_MOVIELENS_CONFIGS.keys())}"
            )
        self.version = version
        self.config = _MOVIELENS_CONFIGS[version]
        self.data_dir = Path(data_dir) / f"movielens-{version}"
        self._data: InteractionData | None = None

    def load(self) -> MovieLens:
        """Download (if needed) and load the dataset.

        Returns:
            Self for method chaining.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.data_dir / "ratings.parquet"

        if cache_path.exists():
            df = pl.read_parquet(cache_path)
        else:
            df = self._download_and_parse()
            df.write_parquet(cache_path)

        # Load item metadata if available
        item_features = self._load_item_features()

        self._data = InteractionData(
            interactions=df,
            item_features=item_features,
            metadata={
                "name": f"MovieLens-{self.version.upper()}",
                "version": self.version,
                "source": self.config["url"],
            },
        )
        return self

    def split(
        self,
        strategy: str = "random",
        test_ratio: float = 0.1,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> InteractionData:
        """Split the loaded dataset.

        Args:
            strategy: "random", "temporal", or "leave_one_out".
            test_ratio: Fraction for test set (not used in leave_one_out).
            val_ratio: Fraction for validation set (not used in leave_one_out).
            seed: Random seed (only used for random split).

        Returns:
            InteractionData with train, val, test populated.
        """
        if self._data is None:
            raise RuntimeError("Call .load() before .split()")

        if strategy == "random":
            return random_split(self._data, test_ratio, val_ratio, seed)
        elif strategy == "temporal":
            return temporal_split(self._data, test_ratio, val_ratio)
        elif strategy == "leave_one_out":
            return leave_one_out_split(self._data, time_order=True)
        else:
            raise ValueError(f"Unknown split strategy '{strategy}'")

    def preprocess(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
    ) -> MovieLens:
        """Apply preprocessing filters.

        Args:
            min_user_interactions: Remove users with fewer interactions.
            min_item_interactions: Remove items with fewer interactions.

        Returns:
            Self for method chaining.
        """
        if self._data is None:
            raise RuntimeError("Call .load() before .preprocess()")
        self._data = self._data.filter_by_min_interactions(
            min_user=min_user_interactions,
            min_item=min_item_interactions,
        )
        return self

    def _download_and_parse(self) -> pl.DataFrame:
        """Download the dataset zip and parse ratings."""
        url = self.config["url"]
        print(f"Downloading MovieLens {self.version} from {url}...")

        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        content = io.BytesIO()
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
                pbar.update(len(chunk))

        content.seek(0)
        with zipfile.ZipFile(content) as zf:
            ratings_file = self.config["ratings_file"]
            with zf.open(ratings_file) as f:
                raw_data = f.read().decode("utf-8", errors="replace")

        # Parse based on version
        sep = self.config["separator"]
        columns = self.config["columns"]

        if self.version == "25m":
            # CSV with header
            df = pl.read_csv(
                io.StringIO(raw_data),
                has_header=True,
                separator=",",
            ).rename({"userId": "user_id", "movieId": "item_id"})
        else:
            # Custom separator, no header
            lines = raw_data.strip().split("\n")
            rows = [line.split(sep)[:4] for line in lines]
            df = pl.DataFrame(
                {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            ).cast({"user_id": pl.Int64, "item_id": pl.Int64, "rating": pl.Float64, "timestamp": pl.Int64})

        return df.select(["user_id", "item_id", "rating", "timestamp"])

    def _load_item_features(self) -> pl.DataFrame | None:
        """Load item metadata (movies.csv / u.item) if available in cache."""
        # Will be implemented when item features are needed for LLM enhancement
        return None
