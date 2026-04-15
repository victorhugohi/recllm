# ADR-002: Ollama as Primary Local Inference Interface

## Status
**Accepted** (Phase 1)

## Context
RecLLM needs a local LLM inference backend for its local-first philosophy. Three options exist:
1. **Ollama:** User-friendly wrapper around llama.cpp with REST API
2. **llama.cpp (direct):** Low-level C/C++ with Python bindings
3. **vLLM:** Production serving engine with PagedAttention

## Decision
**Ollama is the primary local inference interface. llama.cpp is the advanced alternative. vLLM is optional for production.**

```python
# Default: Ollama (one-line setup)
llm = recllm.llm.OllamaClient(model="mistral:7b-q4_K_M")

# Advanced: llama.cpp (more control)
llm = recllm.llm.LlamaCppClient(model_path="./models/mistral-7b.Q4_K_M.gguf")

# Production: vLLM (optional dependency)
llm = recllm.llm.VLLMClient(model="mistralai/Mistral-7B-v0.3")
```

## Rationale

### Why Ollama as Primary
1. **Lowest barrier to entry:** `ollama pull mistral` -- one command to download and run any model
2. **OpenAI-compatible API:** `/v1/chat/completions` endpoint means RecLLM's OllamaClient and OpenAIClient share most code
3. **Automatic model management:** Downloads, updates, storage handled by Ollama
4. **Automatic GPU detection:** CUDA, Metal, ROCm -- no manual configuration
5. **Structured output:** JSON mode and grammar-constrained generation support
6. **Embeddings endpoint:** Can replace sentence-transformers for some use cases
7. **Active development:** Fast-growing project with broad community support

### Why Not Ollama Only
- **llama.cpp:** Offers finer control (batch size, context length, GPU layers, KV-cache quantization) needed for performance benchmarking
- **vLLM:** Better throughput for multi-user production scenarios with continuous batching

### Hardware Validation
Tested on prior work (AcademIA/SiReCA prototype):
- Ollama + SmolLM2 1.7B on GTX 1050 4GB: functional for feature generation
- REST API overhead is negligible (~1ms) compared to inference time (~50-200ms per generation)

## Consequences

### Positive
- Users can start with local LLMs in minutes, not hours
- Same model works across operating systems (Ollama handles platform differences)
- Easy model switching: change `model="mistral"` to `model="llama3.1"` -- no code changes

### Negative
- Ollama adds a dependency (Go binary, separate process)
- Single-request concurrency by default (improving in newer versions)
- Less control than raw llama.cpp for benchmarking

### Backend Abstraction
All backends implement the same `LLMClient` ABC:
```python
class LLMClient(ABC):
    def generate(self, prompt: str, **kwargs) -> str
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]
    def embed(self, texts: list[str]) -> np.ndarray
```
Switching backends is a one-line configuration change, not a code change.
