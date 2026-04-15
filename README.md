# RecLLM

**Open-source Python library for LLM-enhanced recommendation systems with local model inference.**

RecLLM provides a unified, modular framework for integrating Large Language Models into recommendation pipelines. It supports three LLM-RecSys integration patterns (Feature Enhancement, Ranking, Explanation) with a focus on local inference via Ollama/llama.cpp, making it accessible on consumer hardware.

## Key Features

- **Scikit-learn-style API**: `fit()` -> `predict()` -> `recommend()` -> `evaluate()`
- **Three LLM integration patterns**: Feature Enhancement (RLMRec/KAR-style), Ranking, Explanation
- **Local-first inference**: Ollama as primary backend, with cloud API support
- **Polars-powered data layer**: Fast DataFrame operations with pandas/NumPy interop
- **Built-in evaluation**: NDCG, Hit Rate, MRR, Precision, Recall at K
- **Reproducibility**: Seed management, config serialization, content-addressed caching
- **GPU memory management**: Sequential time-multiplexing between PyTorch and LLM

## Installation

```bash
# From source (development)
git clone https://github.com/victorhugohi/recllm.git
cd recllm
pip install -e ".[dev]"

# Core only (no PyTorch)
pip install -e .
```

## Quick Start

```python
from recllm.models.popularity import PopularityBaseline
from recllm.pipeline import RecommendationPipeline
from recllm.data.movielens import MovieLensLoader

# Load and split data
data = MovieLensLoader("100k").load().preprocess()

# Train and evaluate
model = PopularityBaseline()
pipeline = RecommendationPipeline(model, seed=42)
result = pipeline.run(data, epochs=1)
print(result.summary())
```

### With LLM Feature Enhancement

```python
from recllm.llm.ollama import OllamaClient
from recllm.enhance import FeatureEnhancer

# Connect to local Ollama instance
llm = OllamaClient(model="mistral:7b")

# Enhance item features with LLM-generated descriptions
enhancer = FeatureEnhancer(llm, cache_dir="llm_cache")

pipeline = RecommendationPipeline(model, seed=42)
result = pipeline.run(
    data,
    enhancer=enhancer,
    enhance_items=True,
    feature_col="title",
)
```

### Model Comparison

```python
from recllm.models.popularity import PopularityBaseline
from recllm.models import BPR, NCF

models = {
    "Popularity": PopularityBaseline(),
    "BPR": BPR(embedding_dim=64),
    "NCF": NCF(gmf_dim=32, mlp_dims=[64, 32, 16]),
}

pipeline = RecommendationPipeline(models["Popularity"])
results = pipeline.compare(models, data, epochs=20)

for name, result in results.items():
    print(f"{name}: {result.metrics}")
```

## Architecture

RecLLM uses a 5-layer modular architecture:

| Layer | Module | Purpose |
|-------|--------|---------|
| Data | `recllm.data` | Polars-based loading, splitting, preprocessing |
| Model | `recllm.models` | Recommendation algorithms (Popularity, BPR, NCF) |
| LLM | `recllm.llm` | Backend-agnostic LLM clients (Ollama, OpenAI) |
| Enhancement | `recllm.enhance` | LLM-RecSys integration patterns |
| Evaluation | `recllm.eval` | Ranking metrics and experiment pipelines |

## Models

| Model | Type | Reference |
|-------|------|-----------|
| PopularityBaseline | Non-personalized | - |
| BPR | Matrix Factorization | Rendle et al. (2009) |
| NCF (NeuMF) | Deep Learning | He et al. (2017) |

## Hardware Requirements

RecLLM is designed for consumer hardware:

| Tier | GPU | Recommended Models |
|------|-----|--------------------|
| Minimal | GTX 1050 (4GB) | Phi-3-mini (Q4), TinyLlama |
| Standard | RTX 3060 (12GB) | Mistral 7B (Q4), Llama 3.1 8B (Q4) |
| Full | RTX 4090 (24GB) | Mistral 7B (FP16), Llama 3.1 8B (Q8) |

## Status

**Alpha** -- under active development as part of a doctoral thesis at UMSA, La Paz, Bolivia.

Thesis: *Design and Implementation of an Open-Source Python Library for LLM-Enhanced Recommendation Systems with Local Model Inference*

## License

MIT
