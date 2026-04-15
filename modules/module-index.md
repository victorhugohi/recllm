# RecLLM Module Index

**Repository:** https://github.com/victorhugohi/recllm
**Tests:** 56 passing | **CI:** GitHub Actions (Python 3.10-3.12)

## Implemented Modules

### Layer 1: Data (`recllm.data`)
- [x] `InteractionData` -- Polars-based container with pandas/NumPy interop
- [x] `MovieLensLoader` -- 100k, 1m, 25m with builder pattern
- [x] `random_split`, `temporal_split`, `leave_one_out_split`
- [x] `filter_by_min_interactions`, `encode_ids`
- [ ] Amazon Reviews, Yelp, BookCrossing loaders

### Layer 2: Models (`recllm.models`)
- [x] `PopularityBaseline` -- non-personalized frequency baseline
- [x] `BPR` -- Bayesian Personalized Ranking (Rendle et al. 2009)
- [x] `NCF` (NeuMF) -- GMF+MLP fusion (He et al. 2017), BCE and BPR loss
- [x] `SASRec` -- Self-Attentive Sequential Rec (Kang & McAuley 2018)
- [x] `LightGCN` -- Graph Convolution (He et al. 2020)
- [ ] `DeepFM` -- factorization machines + deep
- All follow scikit-learn API: `fit()` -> `predict()` -> `recommend()` -> `evaluate()`

### Layer 3: LLM Integration (`recllm.llm`)
- [x] `LLMClient` ABC -- backend-agnostic interface (generate, embed, generate_batch)
- [x] `OllamaClient` -- HTTP API, concurrent batch, GPU unload (ADR-005)
- [ ] `OpenAIClient`, `AnthropicClient`, `LlamaCppClient`

### Layer 4: Enhancement (`recllm.enhance`)
- [x] `FeatureEnhancer` -- LLM-as-Feature-Enhancer (RLMRec/KAR pattern), disk caching (ADR-004)
- [x] `LLMRanker` -- LLM-as-Ranker: pointwise, pairwise, listwise modes (TALLRec/Hou)
- [x] `LLMExplainer` -- LLM-as-Explainer: conversational, analytical, brief styles
- [x] `EnhancedFeatures` -- container with texts + embeddings + `to_numpy()`

### Layer 5: Pipeline (`recllm.pipeline`)
- [x] `RecommendationPipeline` -- seed -> split -> enhance -> train -> evaluate
- [x] `PipelineResult` -- metrics, timing, model, enhanced features
- [x] `compare()` -- multi-model benchmarking with shared splits
- [ ] YAML/JSON declarative configuration

### Layer 6: Evaluation (`recllm.eval`)
- [x] `ndcg_at_k`, `hit_rate_at_k`, `mrr`, `precision_at_k`, `recall_at_k`
- [x] `compute_metrics` -- full-ranking per-user evaluation
- [ ] Statistical significance tests (paired t-test, Wilcoxon)
- [ ] Visualization and export utilities

### Utilities (`recllm.utils`)
- [x] `set_seed` -- Python, NumPy, PyTorch seed management
- [x] `profile_hardware` -- GPU/CPU detection with model recommendations

## Remaining Implementation (Phase 2)
- DeepFM model
- Additional dataset loaders (Amazon, Yelp, BookCrossing)
- Cloud LLM clients (OpenAI, Anthropic)
- LlamaCppClient for direct GGUF inference
- Statistical significance testing
- YAML pipeline configuration
- End-to-end MovieLens experiments with Ollama

## Reference Implementations
- RecBole: algorithm implementations, evaluation protocol
- KerasRS: API design patterns
- sentence-transformers: embedding generation
