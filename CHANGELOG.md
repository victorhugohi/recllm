# Changelog

All notable changes to RecLLM are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] -- 2026-04-17

First public release. Library is feature-complete for the three
LLM-RecSys integration patterns described in the accompanying thesis
(Phase 2 deliverable), with end-to-end validation on MovieLens-100K.

### Added

**Baseline models (6):**
- `PopularityBaseline` -- non-personalized top-K from item frequency.
- `BPR` (Rendle 2009) -- matrix factorization with pairwise ranking loss.
- `NCF` / NeuMF (He et al. 2017) -- GMF + MLP fusion; BCE or BPR loss.
- `DeepFM` (Guo et al. 2017) -- FM + deep with shared embeddings.
- `SASRec` (Kang and McAuley 2018) -- causal self-attention sequential.
- `LightGCN` (He et al. 2020) -- light graph convolution on bipartite graph.

**LLM-RecSys patterns (3):**
- `FeatureEnhancer` -- LLM-as-Feature-Enhancer with disk-persistent cache
  for generated text and optional embeddings.
- `LLMRanker` -- pointwise, pairwise, and listwise re-ranking modes
  (TALLRec / Hou et al. formulations).
- `LLMExplainer` -- conversational, analytical, and brief explanation
  styles with cache.

**LLM backends (3):**
- `OllamaClient` -- local inference via Ollama HTTP API, now with a
  `think` flag for qwen3 / deepseek-r1 style reasoning models.
- `OpenAIClient` -- OpenAI-compatible endpoint (works with Azure, vLLM,
  Groq, etc.).
- `LlamaCppClient` -- direct GGUF inference via `llama-cpp-python`.

**Datasets (4):**
- `MovieLens` 100K / 1M / 25M with automatic download, ratings parsing,
  and item feature extraction (titles + genres).
- `AmazonReviews` across 16 product categories.
- `Yelp` academic dataset (manual download required).
- `BookCrossing` (Ziegler et al. 2005) with explicit/implicit filtering.

**Evaluation:**
- Standard metrics: NDCG@K, HR@K, MRR, Recall@K, Precision@K, Coverage,
  Novelty, Diversity.
- Significance testing: paired t-test, Wilcoxon signed-rank, bootstrap
  confidence intervals, Cohen's d, rank-biserial effect sizes.
- Visualization: `plot_model_comparison`, `plot_training_curves`,
  `plot_metric_heatmap`, and `results_to_latex` with best-value bolding.

**Pipeline:**
- `RecommendationPipeline` -- data split -> enhance -> train -> evaluate.
- YAML / JSON declarative configs via `load_config()` and
  `run_from_config()` for reproducible experiments.

**Tooling:**
- Streamlit interactive dashboard (`app/app.py`) with six pages.
- Examples: `movielens_benchmark.py` (6-model comparison with
  significance) and `llm_experiment.py` (end-to-end LLM enhancement).
- GitHub Actions CI across Python 3.10-3.12 (ruff + mypy + pytest).
- 85 passing tests, ruff-clean.

### Notes

- qwen3:8b does not expose the Ollama embeddings endpoint -- use
  `nomic-embed-text` (or similar) when embeddings are required, and set
  `FeatureEnhancer(embed=False)` when only text descriptions are needed.
- Reasoning models (qwen3, deepseek-r1) should be constructed with
  `OllamaClient(..., think=False)` to avoid long chain-of-thought
  traces triggering request timeouts.
