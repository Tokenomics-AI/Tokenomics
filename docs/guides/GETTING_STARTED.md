# Getting Started - Next Steps

## 1. Install Dependencies

First, install all required packages:

```bash
pip install -r requirements.txt
```

**Note**: Some dependencies are optional:
- For **Gemini**: `google-cloud-aiplatform` (already in requirements.txt)
- For **OpenAI**: `openai` (already in requirements.txt)
- For **FAISS**: `faiss-cpu` (already in requirements.txt)
- For **ChromaDB**: `chromadb` (already in requirements.txt)

## 2. Set Up Environment Variables

Create a `.env` file in the project root (copy from `env.template`):

### Option A: Google Gemini (Vertex AI)

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/your-credentials.json
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-exp
```

**To get Google Cloud credentials:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project or select existing one
3. Enable Vertex AI API
4. Create a service account and download JSON key
5. Set `GOOGLE_APPLICATION_CREDENTIALS` to the JSON file path

### Option B: OpenAI

```env
OPENAI_API_KEY=sk-your-api-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
```

**To get OpenAI API key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login
3. Navigate to API Keys section
4. Create a new secret key

### Option C: vLLM (Self-hosted)

```env
LLM_PROVIDER=vllm
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
```

**Note**: Requires a running vLLM server at `http://localhost:8000` (or configure `base_url`)

## 3. Verify Installation

Run a quick test to verify everything works:

```bash
python -c "from tokenomics.core import TokenomicsPlatform; print('✓ Installation successful')"
```

## 4. Run Basic Example

Start with the basic usage example:

```bash
python examples/basic_usage.py
```

This will:
- Initialize the platform
- Process a few sample queries
- Show cache hits/misses
- Display token usage statistics

**Expected output**: You should see queries being processed, cache hits on duplicates, and token usage metrics.

## 5. Run Benchmark

Test the platform's efficiency:

```bash
python examples/benchmark.py
```

This compares:
- Performance WITH caching vs. WITHOUT caching
- Token savings
- Latency improvements

**Expected results**: You should see significant token savings (70-95% for repeated queries) and reduced latency.

## 6. Run Tests

Verify all components work correctly:

```bash
pytest tests/
```

Or run specific test files:
```bash
pytest tests/test_memory.py
pytest tests/test_orchestrator.py
pytest tests/test_bandit.py
```

## 7. Customize for Your Use Case

### Modify Configuration

Edit `tokenomics/config.py` or create custom config:

```python
from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig, MemoryConfig, OrchestratorConfig

config = TokenomicsConfig(
    memory=MemoryConfig(
        cache_size=2000,  # Larger cache
        similarity_threshold=0.9,  # Stricter semantic matching
    ),
    orchestrator=OrchestratorConfig(
        default_token_budget=5000,  # Higher budget
    ),
)

platform = TokenomicsPlatform(config=config)
```

### Add Custom Strategies

```python
from tokenomics.bandit import Strategy

custom_strategy = Strategy(
    arm_id="my_strategy",
    model="gemini-pro",
    max_tokens=3000,
    temperature=0.8,
    memory_mode="rich",
    rerank=True,
)

platform.bandit.add_strategy(custom_strategy)
```

## 8. Integrate into Your Project

### Basic Integration

```python
from tokenomics.core import TokenomicsPlatform

platform = TokenomicsPlatform()

# Process queries
result = platform.query("Your question here")
print(result["response"])
```

### With Custom Settings

```python
from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

config = TokenomicsConfig.from_env()
config.memory.cache_size = 5000
config.orchestrator.default_token_budget = 8000

platform = TokenomicsPlatform(config=config)

# Your queries
results = []
for query in your_queries:
    result = platform.query(query)
    results.append(result)

# Check statistics
stats = platform.get_stats()
print(f"Cache hit rate: {stats['memory']['size']}")
print(f"Best strategy: {platform.bandit.get_best_strategy().arm_id}")
```

## 9. Monitor Performance

Track platform performance:

```python
# After processing queries
stats = platform.get_stats()

print("Memory Stats:")
print(f"  Cache size: {stats['memory']['size']}")
print(f"  Total tokens saved: {stats['memory']['total_tokens_saved']}")

print("\nBandit Stats:")
print(f"  Total pulls: {stats['bandit']['total_pulls']}")
for arm_id, arm_stats in stats['bandit']['arms'].items():
    print(f"  {arm_id}: {arm_stats['pulls']} pulls, avg reward: {arm_stats['average_reward']:.4f}")
```

## 10. Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'vertexai'`
- **Solution**: Install Google Cloud AI Platform: `pip install google-cloud-aiplatform`

**Issue**: `ValueError: project_id must be provided`
- **Solution**: Set `GOOGLE_CLOUD_PROJECT` in `.env` or pass `project_id` to config

**Issue**: `ImportError: faiss not found`
- **Solution**: Install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU)

**Issue**: Semantic cache not working
- **Solution**: Ensure `sentence-transformers` is installed: `pip install sentence-transformers`

**Issue**: Token counting inaccurate
- **Solution**: Install `tiktoken`: `pip install tiktoken`

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
```

## Next Steps After Setup

1. **Experiment with different strategies**: Try various model/token/temperature combinations
2. **Tune bandit parameters**: Adjust `reward_lambda` to balance quality vs. cost
3. **Benchmark on your data**: Test with your actual queries and measure improvements
4. **Add quality scoring**: Integrate BLEU/ROUGE or LLM-as-judge for better rewards
5. **Scale up**: Consider Redis for distributed caching, production vector DBs

## Need Help?

- Check `ARCHITECTURE.md` for detailed component documentation
- See `IMPLEMENTATION_NOTES.md` for design decisions and limitations
- Review `examples/` for more usage patterns
- Check test files in `tests/` for implementation examples

