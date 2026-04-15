# ADR-001: Polars over Pandas for Data Layer

## Status
**Accepted** (Phase 1)

## Context
RecLLM's data layer needs to load, preprocess, and split large recommendation datasets (MovieLens 25M: 25M interactions, Amazon Reviews: 233M interactions). The choice of DataFrame library affects performance, memory usage, and API ergonomics.

The two primary options are:
- **Pandas:** The de facto standard in Python data science. Universal familiarity.
- **Polars:** Rust-based DataFrame library with Apache Arrow backend. 3-10x faster than pandas.

## Decision
**Use Polars internally with pandas compatibility wrappers.**

RecLLM uses Polars DataFrames internally for all data operations but exposes `.to_pandas()` convenience methods and accepts pandas DataFrames as input (auto-converted via Arrow).

## Rationale

### Performance
- Polars is 3-10x faster than pandas for common RecSys operations (groupby, join, filter)
- Lazy evaluation enables query optimization before execution
- Zero-copy interop with Apache Arrow (FAISS, PyTorch, sentence-transformers all support Arrow)
- Native multi-threaded execution (pandas is single-threaded by default)
- Memory-efficient: Arrow columnar format + string deduplication

### Recommendation-Specific Benefits
- **Temporal splitting:** Polars' lazy sort + filter on timestamp columns is significantly faster for leave-one-out and temporal train/test splits on large datasets
- **Negative sampling:** Polars' anti-join is cleaner and faster than pandas merge workarounds
- **User/item encoding:** Polars' categorical type with physical representation is natural for ID encoding

### Compatibility Strategy
```python
# RecLLM accepts both
dataset = recllm.data.load_movielens("25m")  # Returns Polars DataFrame
pandas_df = dataset.to_pandas()               # Easy conversion
dataset2 = recllm.data.from_pandas(my_df)     # Accepts pandas input
```

### Precedent
- KerasRS (Google, 2026) uses tf.data but supports pandas input
- Polars is increasingly adopted in ML pipelines (scikit-learn 1.4+ accepts Polars)

## Consequences

### Positive
- Faster data loading and preprocessing for large datasets
- Lower memory footprint (important when GPU memory is shared with LLM)
- Future-proof: Polars adoption is accelerating

### Negative
- Less familiar to most Python data scientists (mitigated by pandas wrappers)
- Smaller ecosystem of tutorials and StackOverflow answers
- Some RecBole interop may require intermediate pandas conversion

### Risks
- Polars API is still evolving (breaking changes possible) -- mitigated by pinning versions
- Users may expect pandas-native returns -- mitigated by clear documentation + `.to_pandas()`

## Alternatives Considered
- **Pandas only:** Simpler but too slow for 25M+ interaction datasets
- **Polars only (no pandas compat):** Cleaner but would alienate most users
- **DuckDB:** Excellent for analytics but less natural for ML pipeline integration
