# RecLLM

A unified Python library for LLM-enhanced recommendation systems with local model inference.

## Installation

```bash
pip install recllm
```

## Quick Start

```python
from recllm.data import MovieLens
from recllm.models import PopularityBaseline

data = MovieLens("1m").load().split(strategy="random")
model = PopularityBaseline().fit(data.train)
results = model.evaluate(data.test, metrics=["ndcg@10", "hr@10"])
print(results)
```

## Status

Alpha - under active development as part of a doctoral thesis at UMSA, La Paz, Bolivia.
