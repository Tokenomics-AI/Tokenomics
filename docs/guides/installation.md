# Quick Start Guide

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:

Create a `.env` file in the project root:

```env
# For Google Gemini (Vertex AI)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# OR for OpenAI
OPENAI_API_KEY=your-api-key

# Optional: Configure provider
LLM_PROVIDER=gemini  # or "openai" or "vllm"
LLM_MODEL=gemini-2.0-flash-exp
```

## Basic Usage

```python
from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

# Create configuration
config = TokenomicsConfig.from_env()

# Initialize platform
platform = TokenomicsPlatform(config=config)

# Query
result = platform.query("What is machine learning?")
print(result["response"])
print(f"Tokens used: {result['tokens_used']}")
```

## Running Examples

### Basic Usage
```bash
python examples/basic_usage.py
```

### Benchmarking
```bash
python examples/benchmark.py
```

### Advanced Usage
```bash
python examples/advanced_usage.py
```

## Running Tests

```bash
pytest tests/
```

## Key Features

### 1. Caching
The platform automatically caches queries and responses:
- **Exact cache**: Identical queries return cached responses instantly
- **Semantic cache**: Similar queries retrieve relevant context

### 2. Token Optimization
The orchestrator allocates tokens optimally:
- Analyzes query complexity
- Allocates budget across system prompt, context, and response
- Compresses low-value content

### 3. Adaptive Learning
The bandit optimizer learns the best strategies:
- Tracks performance of different configurations
- Balances exploration vs. exploitation
- Improves efficiency over time

## Configuration

Customize the platform through `TokenomicsConfig`:

```python
from tokenomics.config import TokenomicsConfig

config = TokenomicsConfig(
    llm=LLMConfig(
        provider="gemini",
        model="gemini-2.0-flash-exp",
    ),
    memory=MemoryConfig(
        cache_size=1000,
        similarity_threshold=0.85,
    ),
    orchestrator=OrchestratorConfig(
        default_token_budget=4000,
    ),
    bandit=BanditConfig(
        algorithm="ucb",
        reward_lambda=0.001,
    ),
)

platform = TokenomicsPlatform(config=config)
```

## Monitoring

Get platform statistics:

```python
stats = platform.get_stats()
print(stats)
```

This includes:
- Cache hit rates
- Token usage
- Bandit performance
- Strategy selection frequencies

## Next Steps

- See `ARCHITECTURE.md` for detailed architecture
- Check `examples/` for more usage patterns
- Review `tokenomics/config.py` for all configuration options

