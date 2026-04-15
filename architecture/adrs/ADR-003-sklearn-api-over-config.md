# ADR-003: Scikit-learn-Inspired API over Configuration-Based (RecBole-Style)

## Status
**Accepted** (Phase 1)

## Context
Two dominant API paradigms exist in RecSys libraries:
1. **Configuration-based (RecBole):** Users write YAML config files; the framework handles everything
2. **Pythonic/scikit-learn (Surprise, LightFM):** Users write Python code with fit/predict/evaluate methods

RecLLM must choose a primary API style.

## Decision
**Scikit-learn-inspired Pythonic API as the primary interface, with optional YAML pipeline configuration for reproducible experiments.**

```python
# Primary API: Pythonic, explicit
model = recllm.models.NCF(embedding_dim=64, layers=[128, 64, 32])
model.fit(train_data, epochs=20)
predictions = model.predict(user_ids, item_ids)
metrics = model.evaluate(test_data, metrics=["ndcg@10", "hr@10"])

# Optional: YAML pipeline for reproducibility
pipeline = recllm.pipeline.from_config("experiment.yaml")
pipeline.run()
```

## Rationale

### Why Scikit-learn Over Config-Based

1. **Lower learning curve:** Every Python ML practitioner knows `fit/predict/evaluate`. RecBole's config system requires learning a custom DSL.
2. **IDE support:** Python code gets autocomplete, type checking, inline documentation. YAML configs get none.
3. **Composability:** Python objects compose naturally. Config-based systems require custom composition mechanisms.
4. **Debuggability:** Python code can be stepped through in a debugger. Config-driven execution is opaque.
5. **Progressive complexity:** Simple tasks need ~10 lines of Python. Complex pipelines use YAML configs as an optional layer on top.

### Design Principle: Progressive Complexity

| Use Case | Lines of Code | API Level |
|----------|---------------|-----------|
| Train + evaluate basic model | ~10 | Python API |
| Add LLM feature enhancement | ~15 | Python API |
| Full pipeline with LLM + eval | ~25 | Python API |
| Reproducible experiment suite | YAML config | Pipeline config |
| Factorial benchmark | YAML + CLI | Pipeline config |

### Precedent
- **KerasRS (Google):** Keras-style layer composition -- most modern approach
- **Surprise:** scikit-learn fit/predict -- proven usability
- **LightFM:** scikit-learn fit/predict -- proven usability
- **scikit-learn itself:** The gold standard for ML API design

### Why Keep YAML as Optional
- Reproducibility: a single config file captures an entire experiment
- RecBole compatibility: researchers familiar with config-driven workflows can use RecLLM similarly
- CI/CD: config files are version-controllable experiment definitions

## Consequences

### Positive
- Broader adoption: accessible to any Python developer
- Better documentation: API docs auto-generated from type hints and docstrings
- Easier testing: Python objects are directly testable

### Negative
- Complex experiments need more code than RecBole's pure-config approach
- Users coming from RecBole may initially find it less familiar
