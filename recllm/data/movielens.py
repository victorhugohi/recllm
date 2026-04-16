"""MovieLens dataset loader."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from recllm.data.base import InteractionData
from recllm.data.splitting import leave_one_out_split, random_split, temporal_split

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
        # Cache zip for item feature extraction later
        zip_cache = self.data_dir / f"ml-{self.version}.zip"
        if not zip_cache.exists():
            zip_cache.write_bytes(content.getvalue())

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
            ).cast({
                "user_id": pl.Int64,
                "item_id": pl.Int64,
                "rating": pl.Float64,
                "timestamp": pl.Int64,
            })

        return df.select(["user_id", "item_id", "rating", "timestamp"])

    def _load_item_features(self) -> pl.DataFrame | None:
        """Load item metadata (movie titles, genres) from the dataset.

        Returns:
            DataFrame with columns [item_id, title, genres] or None.
        """
        cache_path = self.data_dir / "items.parquet"
        if cache_path.exists():
            return pl.read_parquet(cache_path)

        # Check if we have the zip cached
        zip_cache = self.data_dir / f"ml-{self.version}.zip"
        if not zip_cache.exists():
            # Re-download (item features are in the same zip)
            return None

        try:
            import zipfile

            with zipfile.ZipFile(zip_cache) as zf:
                if self.version == "100k":
                    return self._parse_100k_items(zf, cache_path)
                elif self.version == "1m":
                    return self._parse_1m_items(zf, cache_path)
                elif self.version == "25m":
                    return self._parse_25m_items(zf, cache_path)
        except Exception:
            return None
        return None

    def _parse_100k_items(self, zf, cache_path: Path) -> pl.DataFrame | None:
        """Parse u.item from ML-100K (pipe-separated, latin-1)."""
        try:
            with zf.open("ml-100k/u.item") as f:
                raw = f.read().decode("latin-1")
        except KeyError:
            return None

        genre_names = [
            "unknown", "Action", "Adventure", "Animation", "Children's",
            "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
            "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
            "Sci-Fi", "Thriller", "War", "Western",
        ]

        rows = []
        for line in raw.strip().split("\n"):
            parts = line.split("|")
            if len(parts) < 5:
                continue
            item_id = int(parts[0])
            title = parts[1]
            # Genres are binary columns at positions 5-23
            genres = []
            for i, gname in enumerate(genre_names):
                if len(parts) > 5 + i and parts[5 + i].strip() == "1":
                    genres.append(gname)
            rows.append({
                "item_id": item_id,
                "title": title,
                "genres": "|".join(genres) if genres else "unknown",
            })

        df = pl.DataFrame(rows)
        df.write_parquet(cache_path)
        return df

    def _parse_1m_items(self, zf, cache_path: Path) -> pl.DataFrame | None:
        """Parse movies.dat from ML-1M (::separator, latin-1)."""
        try:
            with zf.open("ml-1m/movies.dat") as f:
                raw = f.read().decode("latin-1")
        except KeyError:
            return None

        rows = []
        for line in raw.strip().split("\n"):
            parts = line.split("::")
            if len(parts) >= 3:
                rows.append({
                    "item_id": int(parts[0]),
                    "title": parts[1],
                    "genres": parts[2],
                })

        df = pl.DataFrame(rows)
        df.write_parquet(cache_path)
        return df

    def _parse_25m_items(self, zf, cache_path: Path) -> pl.DataFrame | None:
        """Parse movies.csv from ML-25M."""
        try:
            with zf.open("ml-25m/movies.csv") as f:
                raw = f.read().decode("utf-8")
        except KeyError:
            return None

        import io
        df = pl.read_csv(io.StringIO(raw), has_header=True)
        df = df.rename({"movieId": "item_id"})
        if "title" not in df.columns:
            return None
        df = df.select(["item_id", "title", "genres"])
        df.write_parquet(cache_path)
        return df
