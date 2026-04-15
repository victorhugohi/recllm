# ADR-004: Caching Strategy for LLM-Generated Features

## Status
**Accepted** (Phase 1)

## Context
LLM-generated features (semantic profiles, knowledge augmentation, explanations) are expensive to compute:
- Feature enhancement for 62K items (MovieLens) with Mistral-7B at ~0.5s/item = ~8.6 hours
- Re-generating on every run is impractical
- Features are deterministic given the same prompt + model + temperature=0

A caching strategy is essential for practical local inference.

## Decision
**Disk-based cache with content-addressed keys. Cache invalidation on prompt template or model change.**

```python
enhancer = recllm.enhance.FeatureEnhancer(
    llm=llm_client,
    cache_dir="./cache/features",   # Persistent disk cache
    cache_key_includes=["model", "prompt_template", "item_hash"],
)

# First run: generates all features (~hours)
enhanced = enhancer.enhance_items(data)

# Subsequent runs: loads from cache (~seconds)
enhanced = enhancer.enhance_items(data)

# Force regeneration
enhanced = enhancer.enhance_items(data, force_refresh=True)
```

## Design

### Cache Key Structure
```
cache_key = hash(model_name + model_quant + prompt_template_hash + item_content_hash)
```

Example: `cache/features/mistral-7b-q4/feature_enhance_v2/item_12345.json`

### Cache Format
- JSON files per item/user (human-readable, debuggable)
- Optional: SQLite database for large catalogs (faster lookup)
- Metadata sidecar: model version, prompt template, generation timestamp, token count

### Invalidation Rules
| Change | Action |
|--------|--------|
| Same model, same prompt, same data | Cache hit (use cached) |
| Different prompt template | Cache miss (regenerate) |
| Different model or quantization | Cache miss (regenerate) |
| New items added to catalog | Generate only new items |
| Item metadata updated | Regenerate affected items |
| Force refresh flag | Regenerate all |

### Batch Processing with Progress
```python
# Generates features with progress bar, checkpointing every 100 items
enhanced = enhancer.enhance_items(data, batch_size=100, checkpoint=True)
# If interrupted, resumes from last checkpoint
```

## Rationale

### Why Disk Cache (Not Memory)
- Features for 62K items at ~1KB each = ~62MB -- fits in memory but should persist across sessions
- Users shouldn't wait 8+ hours every time they restart a notebook
- Disk cache survives process crashes, IDE restarts, system reboots

### Why Content-Addressed (Not Timestamp)
- Same prompt + same model + same item = same output (temperature=0)
- Enables sharing caches across experiments
- Enables comparing outputs from different models on same items

## Consequences

### Positive
- LLM feature generation is a one-time cost per model/prompt/dataset combination
- Experiments become reproducible (cached features are deterministic)
- Enables iterating on RecSys model architecture without regenerating LLM features
- Progress checkpointing prevents loss from interruptions

### Negative
- Disk space: ~62MB per dataset per model per prompt template (manageable)
- Cache management complexity (users need to understand invalidation rules)
- Risk of stale cache if users modify prompts without clearing cache
