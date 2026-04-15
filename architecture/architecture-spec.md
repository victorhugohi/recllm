# RecLLM Architecture Specification

> **Status:** Implemented -- core architecture realized, 56 tests passing
> **Repository:** https://github.com/victorhugohi/recllm

## Module Structure

```
recllm/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── base.py          # Base dataset classes
│   ├── movielens.py     # MovieLens loader
│   ├── amazon.py        # Amazon Reviews loader
│   ├── yelp.py          # Yelp Open loader
│   ├── bookcrossing.py  # BookCrossing loader
│   ├── splitting.py     # Temporal, random, leave-one-out
│   └── preprocessing.py # Normalization, filtering, encoding
├── models/
│   ├── __init__.py
│   ├── base.py          # BaseModel ABC (fit/predict/recommend/evaluate)
│   ├── ncf.py           # Neural Collaborative Filtering
│   ├── sasrec.py        # Self-Attentive Sequential Recommendation
│   ├── lightgcn.py      # Light Graph Convolution Network
│   ├── bpr.py           # Bayesian Personalized Ranking
│   ├── als.py           # Alternating Least Squares
│   ├── deepfm.py        # DeepFM
│   └── popularity.py    # Popularity baseline
├── llm/
│   ├── __init__.py
│   ├── base.py          # LLMClient ABC
│   ├── ollama.py        # OllamaClient
│   ├── llamacpp.py      # LlamaCppClient
│   ├── openai.py        # OpenAIClient
│   ├── anthropic.py     # AnthropicClient
│   ├── prompts.py       # Prompt template management
│   ├── parsing.py       # Response parsing utilities
│   └── budget.py        # Token budget management
├── enhance/
│   ├── __init__.py
│   ├── feature.py       # FeatureEnhancer
│   ├── ranker.py        # LLM Re-Ranker
│   ├── explainer.py     # Explanation Generator
│   └── cache.py         # Feature caching system
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py      # RecommendationPipeline
│   └── config.py        # YAML/JSON config loader
├── eval/
│   ├── __init__.py
│   ├── metrics.py       # NDCG, HR, MRR, Precision, Recall, MAP
│   ├── stats.py         # Statistical significance tests
│   ├── visualization.py # Matplotlib visualization
│   └── export.py        # CSV, LaTeX table export
└── utils/
    ├── __init__.py
    ├── hardware.py      # Hardware profiling, model selection guidance
    └── reproducibility.py # Seeding, config serialization
```

## Implementation Status

| Module | File | Status | Tests |
|--------|------|--------|-------|
| `data/base.py` | InteractionData, encode_ids, filter | Done | 6 |
| `data/splitting.py` | random, temporal, leave-one-out | Done | 3 |
| `data/movielens.py` | MovieLens 100k/1m/25m loader | Done | - |
| `models/popularity.py` | PopularityBaseline | Done | 4 |
| `models/bpr.py` | BPR (Rendle 2009) | Done | - |
| `models/ncf.py` | NCF/NeuMF (He 2017) | Done | - |
| `models/sasrec.py` | SASRec (Kang 2018) | Done | - |
| `models/lightgcn.py` | LightGCN (He 2020) | Done | - |
| `llm/base.py` | LLMClient ABC | Done | - |
| `llm/ollama.py` | OllamaClient | Done | - |
| `enhance/feature_enhancer.py` | FeatureEnhancer + cache | Done | 6 |
| `enhance/ranker.py` | LLMRanker (3 modes) | Done | 7 |
| `enhance/explainer.py` | LLMExplainer (3 styles) | Done | 8 |
| `pipeline/recommendation.py` | Pipeline + compare() | Done | 3 |
| `eval/metrics.py` | NDCG, HR, MRR, P, R @K | Done | 12 |
| `utils/reproducibility.py` | set_seed | Done | - |
| `utils/hardware.py` | profile_hardware | Done | - |

**Total: 56 tests passing** | CI: GitHub Actions (Python 3.10-3.12)

## Key Interfaces

### BaseModel
```python
class BaseModel(ABC):
    def fit(self, train_data: InteractionData, epochs: int = 20) -> Self
    def predict(self, user_ids, item_ids) -> np.ndarray
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]
    def evaluate(self, test_data: InteractionData, metrics: list[str]) -> dict
```

### LLMClient
```python
class LLMClient(ABC):
    def generate(self, prompt: str, **kwargs) -> str
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]
    def embed(self, texts: list[str]) -> np.ndarray
```

### FeatureEnhancer
```python
class FeatureEnhancer:
    def __init__(self, llm: LLMClient, cache_dir: str = "llm_cache")
    def enhance_items(self, data, feature_col=None, embed=True) -> EnhancedFeatures
    def enhance_users(self, data, max_history=20, embed=True) -> EnhancedFeatures
```

### LLMRanker
```python
class LLMRanker:
    def __init__(self, llm: LLMClient, mode="listwise")  # pointwise, pairwise, listwise
    def rerank(self, user_history, candidates, candidate_ids=None) -> list[tuple[int, float]]
```

### LLMExplainer
```python
class LLMExplainer:
    def __init__(self, llm: LLMClient, style="conversational")  # analytical, brief
    def explain(self, user_history, recommended_item, score) -> str
    def explain_batch(self, user_history, items, scores) -> list[str]
```

### RecommendationPipeline
```python
class RecommendationPipeline:
    def __init__(self, model: BaseModel, seed=42, eval_metrics=["ndcg@10", "hr@10"])
    def run(self, data, epochs=20, enhancer=None) -> PipelineResult
    def compare(self, models: dict[str, BaseModel], data) -> dict[str, PipelineResult]
```

## Architecture Decision Records (ADRs)

- [x] [[adrs/ADR-001-polars-over-pandas|ADR-001]]: Polars over Pandas for data layer -- **Accepted**
- [x] [[adrs/ADR-002-ollama-primary-local|ADR-002]]: Ollama as primary local inference interface -- **Accepted**
- [x] [[adrs/ADR-003-sklearn-api-over-config|ADR-003]]: Scikit-learn-inspired API over config-based -- **Accepted**
- [x] [[adrs/ADR-004-llm-feature-caching|ADR-004]]: Disk-based caching for LLM-generated features -- **Accepted**
- [x] [[adrs/ADR-005-gpu-memory-sharing|ADR-005]]: Sequential GPU memory management (time-multiplex) -- **Accepted**
