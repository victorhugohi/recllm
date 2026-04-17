# RecLLM Architecture Specification

> **Status:** Implemented -- core architecture realized, 85 tests passing, v0.1.0 release
> **Repository:** https://github.com/victorhugohi/recllm
> **Last updated:** 2026-04-17

## 1. Scope and goals

RecLLM is a unified, open-source Python library for LLM-enhanced
recommendation systems with first-class support for local inference.
The library abstracts three LLM-RecSys integration patterns identified
in the literature (Feature Enhancer, Re-Ranker, Explainer) into
composable modules on top of six classical recommender models, so a
single YAML-configurable pipeline can compare "pure" baselines against
LLM-augmented variants.

Design priorities in order:

1. **Pedagogical clarity.** The library is a doctoral artefact and must
   be readable by graduate students and researchers, not just consumed.
   Public APIs follow scikit-learn conventions so prior mental models
   transfer.
2. **Reproducibility.** All experiments can be reconstructed from a
   declarative YAML / JSON config, a seed, and dataset identifiers. No
   hidden state in the pipeline.
3. **Local-first.** Ollama and `llama-cpp-python` are first-class LLM
   backends; cloud APIs are optional and live behind a provider-neutral
   interface.
4. **Honesty about limits.** Disk caching for LLM output, explicit GPU
   memory sharing, and small-scale defaults protect users from silent
   cost or OOM surprises.

## 2. Module structure

```
recllm/
|-- __init__.py
|-- data/
|   |-- __init__.py
|   |-- base.py            # InteractionData, encoding, filter helpers
|   |-- splitting.py       # random, temporal, leave-one-out
|   |-- preprocessing.py   # normalization, min-interaction filters
|   |-- movielens.py       # MovieLens 100K / 1M / 25M + item features
|   |-- amazon.py          # Amazon Reviews (16 categories)
|   |-- yelp.py            # Yelp Academic Dataset (manual download)
|   +-- bookcrossing.py    # BookCrossing (Ziegler 2005)
|-- models/
|   |-- __init__.py
|   |-- base.py            # BaseModel ABC
|   |-- popularity.py      # Non-personalized top-K baseline
|   |-- bpr.py             # Bayesian Personalized Ranking (Rendle 2009)
|   |-- ncf.py             # NCF / NeuMF (He 2017)
|   |-- deepfm.py          # DeepFM (Guo 2017)
|   |-- sasrec.py          # Self-Attentive Sequential (Kang 2018)
|   +-- lightgcn.py        # LightGCN (He 2020)
|-- llm/
|   |-- __init__.py
|   |-- base.py            # LLMClient ABC
|   |-- ollama.py          # OllamaClient (incl. `think` flag)
|   |-- llamacpp.py        # LlamaCppClient (direct GGUF)
|   +-- openai_client.py   # OpenAIClient (OpenAI-compatible APIs)
|-- enhance/
|   |-- __init__.py
|   |-- feature_enhancer.py  # Pattern 1: LLM-as-Feature-Enhancer
|   |-- ranker.py            # Pattern 2: LLM Re-Ranker (3 modes)
|   +-- explainer.py         # Pattern 3: LLM Explainer (3 styles)
|-- pipeline/
|   |-- __init__.py
|   |-- recommendation.py  # RecommendationPipeline.run / compare
|   +-- config.py          # load_config / run_from_config (YAML, JSON)
|-- eval/
|   |-- __init__.py
|   |-- metrics.py         # NDCG@K, HR@K, MRR, P@K, R@K, Coverage,
|   |                      # Novelty, Diversity, compute_metrics
|   |-- significance.py    # paired t-test, Wilcoxon, bootstrap CI
|   +-- visualization.py   # bar/line/heatmap plots, LaTeX export
+-- utils/
    |-- __init__.py
    |-- hardware.py        # profile_hardware, model-selection guidance
    +-- reproducibility.py # set_seed, config hashing
```

Out of tree but ship with the repo:

```
app/app.py                 # Streamlit dashboard (6 pages)
examples/
|-- movielens_benchmark.py # 6-model benchmark + significance
|-- llm_experiment.py      # End-to-end 3-pattern LLM experiment
+-- experiment_config.yaml # Declarative config example
```

## 3. Implementation status (v0.1.0)

| Module | File | Key class / function | Tests |
|--------|------|----------------------|-------|
| `data/base.py` | InteractionData, encoding, filters | done | 6 |
| `data/splitting.py` | random, temporal, leave-one-out | done | 3 |
| `data/movielens.py` | MovieLens loader with item features | done | - |
| `data/amazon.py` | Amazon Reviews loader | done | - |
| `data/yelp.py` | YelpDataset loader | done | - |
| `data/bookcrossing.py` | BookCrossing loader | done | - |
| `models/popularity.py` | PopularityBaseline | done | 4 |
| `models/bpr.py` | BPR | done | - |
| `models/ncf.py` | NCF / NeuMF | done | - |
| `models/deepfm.py` | DeepFM | done | 5 |
| `models/sasrec.py` | SASRec | done | - |
| `models/lightgcn.py` | LightGCN | done | - |
| `llm/base.py` | LLMClient ABC | done | - |
| `llm/ollama.py` | OllamaClient (with `think` flag) | done | - |
| `llm/openai_client.py` | OpenAIClient | done | - |
| `llm/llamacpp.py` | LlamaCppClient | done | - |
| `enhance/feature_enhancer.py` | FeatureEnhancer + cache | done | 6 |
| `enhance/ranker.py` | LLMRanker (3 modes) | done | 7 |
| `enhance/explainer.py` | LLMExplainer (3 styles) | done | 8 |
| `pipeline/recommendation.py` | RecommendationPipeline | done | 3 |
| `pipeline/config.py` | load_config / run_from_config | done | 10 |
| `eval/metrics.py` | standard metrics | done | 12 |
| `eval/significance.py` | paired t-test, Wilcoxon, bootstrap | done | 10 |
| `eval/visualization.py` | plots + LaTeX export | done | 4 |
| `utils/reproducibility.py` | set_seed | done | - |
| `utils/hardware.py` | profile_hardware | done | - |

**Total: 85 tests passing** | CI: GitHub Actions (Python 3.10-3.12) |
ruff clean | `pyproject.toml` version 0.1.0

## 4. Key interfaces

### BaseModel

```python
class BaseModel(ABC):
    def fit(self, train_data: InteractionData, epochs: int = 20,
            val_data: InteractionData | None = None) -> Self: ...
    def predict(self, user_ids: np.ndarray,
                item_ids: np.ndarray) -> np.ndarray: ...
    def recommend(self, user_id: int, n: int = 10,
                  exclude_seen: bool = True,
                  seen_items: set[int] | None = None
                  ) -> list[tuple[int, float]]: ...
    def evaluate(self, test_data: InteractionData,
                 metrics: list[str]) -> dict[str, float]: ...
```

### LLMClient

```python
class LLMClient(ABC):
    @property
    def model_name(self) -> str: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def is_available(self) -> bool: ...
    def unload(self) -> None: ...
```

### FeatureEnhancer (Pattern 1)

```python
class FeatureEnhancer:
    def __init__(self, llm: LLMClient, cache_dir: str = "llm_cache",
                 batch_size: int = 4) -> None: ...
    def enhance_items(self, data: InteractionData,
                      feature_col: str | None = None,
                      embed: bool = True) -> EnhancedFeatures: ...
    def enhance_users(self, data: InteractionData,
                      max_history: int = 20,
                      embed: bool = True) -> EnhancedFeatures: ...
```

### LLMRanker (Pattern 2)

```python
class LLMRanker:
    def __init__(self, llm: LLMClient,
                 mode: str = "listwise",          # or "pointwise", "pairwise"
                 max_candidates: int = 20) -> None: ...
    def rerank(self, user_history: list[str],
               candidates: list[str],
               candidate_ids: list[int] | None = None
               ) -> list[tuple[int, float]]: ...
```

### LLMExplainer (Pattern 3)

```python
class LLMExplainer:
    def __init__(self, llm: LLMClient,
                 style: str = "conversational",   # or "analytical", "brief"
                 cache_dir: str | Path | None = None) -> None: ...
    def explain(self, user_history: list[str],
                recommended_item: str, score: float = 0.0,
                custom_context: str | None = None) -> str: ...
    def explain_batch(self, user_history: list[str],
                      recommended_items: list[str],
                      scores: list[float] | None = None) -> list[str]: ...
```

### RecommendationPipeline

```python
class RecommendationPipeline:
    def __init__(self, model: BaseModel, seed: int = 42,
                 eval_metrics: list[str] = ("ndcg@10", "hr@10")): ...
    def run(self, data: InteractionData, epochs: int = 20,
            enhancer: FeatureEnhancer | None = None) -> PipelineResult: ...
    def compare(self, models: dict[str, BaseModel],
                data: InteractionData) -> dict[str, PipelineResult]: ...
```

### Declarative config

```python
from recllm.pipeline import load_config, run_from_config

cfg = load_config("experiments/ml100k_bpr.yaml")
result = run_from_config(cfg)       # also accepts a path string directly
```

## 5. Architecture Decision Records

- [x] [[adrs/ADR-001-polars-over-pandas|ADR-001]]: Polars over Pandas -- **Accepted**
- [x] [[adrs/ADR-002-ollama-primary-local|ADR-002]]: Ollama as primary local inference -- **Accepted**
- [x] [[adrs/ADR-003-sklearn-api-over-config|ADR-003]]: scikit-learn-style API as core, YAML config as wrapper -- **Accepted**
- [x] [[adrs/ADR-004-llm-feature-caching|ADR-004]]: Disk caching for LLM outputs -- **Accepted**
- [x] [[adrs/ADR-005-gpu-memory-sharing|ADR-005]]: Sequential GPU memory management -- **Accepted**

## 6. Validated operating envelope (v0.1.0)

Evidence to date (`results/llm_experiment.json`, MovieLens-100K,
2026-04-17, Ollama `qwen3:8b` with `think=False`, `num_ctx=2048`):

- Popularity baseline NDCG@10 = 0.0754; BPR NDCG@10 = 0.0770 on the 10 %
  random-split held-out set.
- FeatureEnhancer: ~80 s per item (text-only), disk-cached. Embeddings
  require a separate embedding-capable Ollama model (e.g.
  `nomic-embed-text`); `qwen3:8b` does not expose `/api/embeddings`.
- LLMRanker listwise: ~6 s per user re-rank over 10 BPR candidates.
- LLMExplainer: conversational ~16 s, analytical ~120 s (post-prompt
  tightening), brief ~11 s per recommendation.

These are local single-machine numbers and are intended as a sanity
floor rather than a benchmark. Thesis Phase 3 will run the full
factorial (backends x patterns x models x datasets).

## 7. Known limitations

- **Embedding coverage.** Not every Ollama model exposes embeddings;
  `FeatureEnhancer` falls back to text-only with `embed=False`.
- **Reasoning models.** qwen3 / deepseek-r1 default to emitting long
  `<think>` traces; always construct `OllamaClient(..., think=False)`
  unless chain-of-thought is desired.
- **Single-machine assumption.** The pipeline does not orchestrate
  distributed training or multi-GPU inference; ADR-005 describes the
  time-multiplex strategy used instead.
- **Yelp license.** The Yelp loader requires a manual download step per
  the dataset's terms of service.
