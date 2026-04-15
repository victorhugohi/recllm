# ADR-005: GPU Memory Management for Shared PyTorch + LLM Inference

## Status
**Accepted** (Phase 1)

## Context
On consumer hardware (GTX 1050 4GB, RTX 3060 12GB), GPU memory is shared between:
1. **PyTorch:** RecSys model training/inference (embeddings, GNN computations)
2. **LLM inference:** Ollama/llama.cpp running local models (model weights, KV-cache)

Both compete for the same VRAM. Without coordination, out-of-memory errors are likely.

## Decision
**Sequential execution with explicit memory boundaries. No concurrent GPU sharing by default.**

### Strategy: Time-Multiplex, Not Space-Share

```
Phase 1: LLM Feature Generation (Ollama uses GPU)
   |-- Generate item features (LLM has full GPU)
   |-- Generate user profiles (LLM has full GPU)
   |-- Cache all features to disk
   |
Phase 2: RecSys Model Training (PyTorch uses GPU)
   |-- Load cached LLM features from disk
   |-- Train NCF/SASRec/LightGCN with enhanced features
   |-- PyTorch has full GPU
   |
Phase 3: LLM Re-Ranking (Optional, Ollama uses GPU)
   |-- PyTorch model generates candidate list (CPU or small GPU footprint)
   |-- Ollama re-ranks candidates (LLM has GPU)
```

### Implementation

```python
# Pipeline automatically manages GPU allocation
pipeline = recllm.pipeline.RecommendationPipeline(
    memory_strategy="sequential",  # Default: time-multiplex
    # memory_strategy="shared",    # Advanced: concurrent (requires config)
)

# Or manual control
with recllm.utils.gpu_context("llm"):
    features = enhancer.enhance_items(data)
    
with recllm.utils.gpu_context("pytorch"):
    model.fit(enhanced_data, epochs=20)
```

## Rationale

### Why Sequential Over Concurrent
1. **Simplicity:** No need for complex memory budgeting
2. **Reliability:** OOM errors are the #1 user experience killer on consumer GPUs
3. **Performance:** On GTX 1050 (4GB), there simply isn't room for both simultaneously
4. **Caching makes it work:** LLM features are generated once and cached (ADR-004), so the LLM doesn't need to run during training

### Memory Budget Analysis

**GTX 1050 (4GB):**
| Component | VRAM | Scenario |
|-----------|------|----------|
| Phi-3 Mini Q4_K_M | ~2.2 GB | LLM phase |
| KV-cache (2K context) | ~0.3 GB | LLM phase |
| OS/driver overhead | ~0.5 GB | Always |
| Available for PyTorch | ~1.0 GB | -- |
| NCF embeddings (62K items) | ~0.5 GB | Training phase |
| LightGCN + graph | ~1.5 GB | Training phase |

Conclusion: Sequential execution is **required** on GTX 1050.

**RTX 3060 (12GB):**
| Component | VRAM | Scenario |
|-----------|------|----------|
| Mistral 7B Q4_K_M | ~4.1 GB | LLM phase |
| KV-cache (4K context) | ~0.5 GB | LLM phase |
| OS/driver overhead | ~0.5 GB | Always |
| Available for PyTorch | ~6.9 GB | -- |

Conclusion: Sequential is recommended but concurrent is technically feasible with careful budgeting.

### Advanced: Concurrent Mode
For users with RTX 3060+ who need LLM re-ranking at inference time:
```python
# Explicit memory budgets
pipeline = recllm.pipeline.RecommendationPipeline(
    memory_strategy="shared",
    llm_gpu_memory_limit="4GB",    # Cap Ollama's GPU usage
    pytorch_gpu_memory_limit="6GB", # Cap PyTorch's GPU usage
)
```

Ollama supports `OLLAMA_MAX_VRAM` environment variable for limiting GPU memory. PyTorch supports `torch.cuda.set_per_process_memory_fraction()`.

## Consequences

### Positive
- No OOM errors on consumer hardware
- Simple mental model for users
- Caching strategy (ADR-004) makes sequential execution practical

### Negative
- LLM re-ranking at inference time requires loading/unloading models (adds latency)
- Users with high-end GPUs are under-utilizing hardware in default mode
- Concurrent mode is advanced and may be fragile

### Mitigation
- Hardware profiling utility: `recllm.utils.hardware.profile()` reports available VRAM and recommends models + strategy
- Clear documentation on memory requirements per model/quantization
- Graceful degradation: if GPU OOM, automatically fall back to CPU for LLM inference
