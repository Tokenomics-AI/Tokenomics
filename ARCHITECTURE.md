# Tokenomics Platform Architecture

## Overview

The Tokenomics platform is designed to maximize value from token budgets in LLM calls through three core components:

1. **Smart Memory Layer**: Caching and retrieval
2. **Token-Aware Orchestrator**: Dynamic budget allocation
3. **Bandit Optimizer**: Adaptive strategy selection

## Component Details

### Smart Memory Layer

Located in `tokenomics/memory/`:

- **`cache.py`**: Exact match caching with LRU/time-based eviction
- **`vector_store.py`**: Semantic search using FAISS or ChromaDB
- **`memory_layer.py`**: Unified interface combining both caches

**Features**:
- Two-layer caching (exact + semantic)
- Configurable eviction policies
- Vector embeddings for semantic similarity
- TTL support for time-based expiry

### Token-Aware Orchestrator

Located in `tokenomics/orchestrator/`:

- **`orchestrator.py`**: Query planning and token allocation

**Features**:
- Query complexity analysis
- Dynamic token allocation (knapsack optimization)
- Prompt compression
- Multi-model routing

### Bandit Optimizer

Located in `tokenomics/bandit/`:

- **`bandit.py`**: Multi-armed bandit implementation

**Features**:
- UCB, epsilon-greedy, and Thompson Sampling algorithms
- Strategy configuration (model, tokens, temperature, etc.)
- Reward computation based on quality vs. token cost
- Contextual bandits by query type

### LLM Providers

Located in `tokenomics/llm_providers/`:

- **`base.py`**: Abstract base class
- **`gemini.py`**: Google Gemini via Vertex AI
- **`openai_provider.py`**: OpenAI API
- **`vllm_provider.py`**: Self-hosted vLLM

## Data Flow

```
User Query
    ↓
[Memory Layer] Check cache (exact → semantic)
    ↓ (cache miss)
[Bandit Optimizer] Select strategy
    ↓
[Orchestrator] Create query plan & allocate tokens
    ↓
[LLM Provider] Generate response
    ↓
[Memory Layer] Store result
    ↓
[Bandit Optimizer] Update statistics
    ↓
Return Response
```

## Configuration

Configuration is managed through `TokenomicsConfig` in `tokenomics/config.py`:

- LLM provider settings
- Memory layer settings
- Orchestrator parameters
- Bandit algorithm configuration

Can be loaded from environment variables or passed directly.

## Extension Points

### Adding New LLM Providers

1. Inherit from `LLMProvider` in `tokenomics/llm_providers/base.py`
2. Implement `generate()` and `count_tokens()` methods
3. Add to `tokenomics/llm_providers/__init__.py`

### Adding New Bandit Algorithms

1. Add algorithm enum to `BanditAlgorithm`
2. Implement `select_arm_<algorithm>()` method
3. Update `select_strategy()` to use new algorithm

### Custom Vector Stores

1. Inherit from `VectorStore` in `tokenomics/memory/vector_store.py`
2. Implement required methods
3. Add factory logic in `SmartMemoryLayer`

## Performance Considerations

- **Caching**: Reduces token usage by 70-95% for repeated queries
- **Token Allocation**: Optimizes context vs. response token split
- **Multi-Model Routing**: Can reduce costs by 30-80% for simple queries
- **Bandit Learning**: Improves efficiency over time through exploration

## Future Enhancements

- Context compression using LLM summarization
- Learned token allocation policies
- Quality scoring integration (BLEU, ROUGE, LLM-as-judge)
- Distributed caching (Redis)
- Production deployment configurations

