"""Book-Crossing dataset loader.

Supports the Book-Crossing dataset (Ziegler et al. 2005) for book recommendation.
Contains explicit ratings (1-10) and implicit interactions (0 = implicit).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

from recllm.data.base import InteractionData

_BOOKCROSSING_URL = (
    "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
)


class BookCrossing:
    """Book-Crossing dataset loader.

    Downloads and parses the BX-Book-Ratings CSV from the Book-Crossing
    dataset. Supports filtering to explicit-only ratings (1-10) or
    including implicit interactions (rating=0).

    Args:
        cache_dir: Directory for downloaded and processed files.
        explicit_only: If True, keep only explicit ratings (1-10).

    Example:
        >>> data = BookCrossing().load()
        >>> print(data.data.summary())
    """

    def __init__(
        self,
        cache_dir: str | Path = "data_cache/bookcrossing",
        explicit_only: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.explicit_only = explicit_only
        self._data: InteractionData | None = None

    def load(self, max_ratings: int | None = None) -> BookCrossing:
        """Download (if needed) and load the dataset.

        Args:
            max_ratings: Maximum number of ratings to load (None = all).

        Returns:
            Self for method chaining.
        """
        suffix = "explicit" if self.explicit_only else "all"
        parquet_path = self.cache_dir / f"interactions_{suffix}.parquet"

        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            if max_ratings:
                df = df.head(max_ratings)
        else:
            df = self._download_and_parse()
            df.write_parquet(parquet_path)
            if max_ratings:
                df = df.head(max_ratings)

        item_features = self._parse_book_features()

        self._data = InteractionData(
            interactions=df,
            item_features=item_features,
            metadata={
                "name": "Book-Crossing",
                "source": "BX-CSV-Dump",
                "explicit_only": self.explicit_only,
            },
        )
        return self

    def _download_and_parse(self) -> pl.DataFrame:
        """Download the zip and parse ratings CSV."""
        zip_path = self.cache_dir / "BX-CSV-Dump.zip"

        if not zip_path.exists():
            response = requests.get(_BOOKCROSSING_URL, stream=True, timeout=120)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc="Downloading Book-Crossing",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Parse ratings from zip
        with zipfile.ZipFile(zip_path) as zf:
            # Find the ratings file (BX-Book-Ratings.csv)
            ratings_file = None
            for name in zf.namelist():
                if "Rating" in name and name.endswith(".csv"):
                    ratings_file = name
                    break
            if ratings_file is None:
                raise FileNotFoundError("BX-Book-Ratings.csv not found in zip")

            with zf.open(ratings_file) as f:
                raw = f.read().decode("latin-1")

        # CSV uses ";" separator with quotes
        df = pl.read_csv(
            io.StringIO(raw),
            separator=";",
            has_header=True,
            quote_char='"',
            infer_schema_length=10000,
        )

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip('"').replace("-", "_")
            if "user" in lower:
                col_map[col] = "user_id_raw"
            elif "isbn" in lower:
                col_map[col] = "item_id_str"
            elif "rating" in lower:
                col_map[col] = "rating"
        df = df.rename(col_map)

        # Cast rating to float
        df = df.with_columns(pl.col("rating").cast(pl.Float64))

        # Filter explicit only if requested
        if self.explicit_only:
            df = df.filter(pl.col("rating") > 0)

        # Encode string IDs to integers
        unique_users = df["user_id_raw"].unique().sort().to_list()
        unique_items = df["item_id_str"].unique().sort().to_list()
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}

        df = df.select([
            pl.col("user_id_raw").replace_strict(user_map).cast(pl.Int64).alias("user_id"),
            pl.col("item_id_str").replace_strict(item_map).cast(pl.Int64).alias("item_id"),
            pl.col("rating"),
        ])

        return df

    def _parse_book_features(self) -> pl.DataFrame | None:
        """Parse book metadata from BX-Books.csv if available in cache."""
        zip_path = self.cache_dir / "BX-CSV-Dump.zip"
        if not zip_path.exists():
            return None

        try:
            with zipfile.ZipFile(zip_path) as zf:
                books_file = None
                for name in zf.namelist():
                    if "Book" in name and "Rating" not in name and name.endswith(".csv"):
                        books_file = name
                        break
                if books_file is None:
                    return None

                with zf.open(books_file) as f:
                    raw = f.read().decode("latin-1")

            df = pl.read_csv(
                io.StringIO(raw),
                separator=";",
                has_header=True,
                quote_char='"',
                infer_schema_length=10000,
                truncate_ragged_lines=True,
            )

            # Keep relevant columns
            cols = df.columns
            rename = {}
            for c in cols:
                lower = c.lower().strip('"')
                if "isbn" in lower:
                    rename[c] = "isbn"
                elif "title" in lower:
                    rename[c] = "title"
                elif "author" in lower:
                    rename[c] = "author"
                elif "year" in lower:
                    rename[c] = "year"
                elif "publisher" in lower:
                    rename[c] = "publisher"

            df = df.rename(rename)
            keep = [c for c in ["isbn", "title", "author", "year", "publisher"] if c in df.columns]
            return df.select(keep) if keep else None
        except Exception:
            return None

    @property
    def data(self) -> InteractionData:
        """Access loaded data."""
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        return self._data

    def preprocess(self, min_user: int = 5, min_item: int = 5) -> BookCrossing:
        """Filter users and items with fewer than min interactions."""
        if self._data is None:
            raise RuntimeError("Call .load() first.")
        self._data = self._data.filter_by_min_interactions(min_user, min_item)
        return self
