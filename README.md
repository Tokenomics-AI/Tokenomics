# Tokenomics

> **Intelligent LLM Token Optimization Platform**  
> Reduce LLM costs by **90.7%** while maintaining response quality through smart caching, adaptive routing, and compression.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tested](https://img.shields.io/badge/tested-50%20queries-brightgreen.svg)]()

---

## Proven Results

We ran a comprehensive test with **50 diverse queries** across 7 categories. Here are the results:

| Metric | Value |
|--------|-------|
| **Cost Savings** | **90.7%** |
| Baseline Cost (50 queries) | $0.150788 |
| Optimized Cost (50 queries) | $0.013999 |
| **Total Saved** | **$0.136788** |
| Cache Hit Rate | 20% |
| Complexity Accuracy | 90% |

### Savings Breakdown by Component

```
Memory Layer (Cache):     $0.033  (22%)
Bandit Optimizer:         $0.103  (75%)
Compression:              $0.004  (3%)
─────────────────────────────────────────
TOTAL SAVINGS:            $0.137  (90.7%)
```

📄 **Full Test Results:** [COMPREHENSIVE_E2E_TEST_RESULTS.md](COMPREHENSIVE_E2E_TEST_RESULTS.md)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  1. MEMORY LAYER                                                  │
│     ├── Exact Cache: Hash-based O(1) lookup                       │
│     ├── Semantic Cache: Vector similarity search                  │
│     └── Context Injection: Enrich prompts with similar responses  │
│                                                                   │
│     Result: 20% cache hit rate → 0 tokens for cached queries      │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  2. TOKEN ORCHESTRATOR                                            │
│     ├── Complexity Analysis: Simple/Medium/Complex (90% accuracy) │
│     ├── Knapsack Optimization: Maximize utility per token         │
│     └── Token Budget Allocation: Smart distribution               │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  3. BANDIT OPTIMIZER                                              │
│     ├── Strategy Selection: cheap/balanced/premium                │
│     ├── UCB Algorithm: Exploration vs exploitation                │
│     └── Cost-Aware Routing: Balance quality and cost              │
│                                                                   │
│     Result: Learns to use cheaper models → 75% of savings         │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  4. COMPRESSION (LLMLingua)                                       │
│     ├── Long Query Detection: >500 chars or >150 tokens           │
│     └── Intelligent Compression: Preserve meaning, reduce tokens  │
│                                                                   │
│     Result: 376 tokens saved on 5 long queries                    │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│  5. CASCADING INFERENCE                                           │
│     ├── Start with cheap model (gpt-4o-mini)                      │
│     ├── Quality check (threshold: 0.85)                           │
│     └── Escalate to premium (gpt-4o) if needed                    │
│                                                                   │
│     Result: Quality maintained while minimizing cost              │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      OPTIMIZED RESPONSE                           │
│                  90.7% cheaper, same quality                      │
└───────────────────────────────────────────────────────────────────┘
```

---

## Learning & Adaptation

Tokenomics gets smarter over time through multiple ML components:

### Pre-trained ML Models

| Model | Purpose | Accuracy |
|-------|---------|----------|
| **Complexity Classifier** | Categorize queries as simple/medium/complex | 90% |
| **Token Predictor** | Estimate response length before calling LLM | 72% |
| **Escalation Predictor** | Predict when quality escalation is needed | 85% |

### Continuous Learning

```
┌─────────────────────────────────────────────────────────────────┐
│  BANDIT OPTIMIZER (UCB Algorithm)                               │
│                                                                 │
│  Learns from every query:                                       │
│  ├── Tracks cost vs quality for each strategy                   │
│  ├── Balances exploration (try new) vs exploitation (use best)  │
│  └── Adapts routing based on actual outcomes                    │
│                                                                 │
│  Result: Routing improves automatically with usage              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Collection Pipeline

The platform collects training data from real queries to retrain models:

```bash
# Collect training data
python scripts/collect_training_data.py

# Retrain models with new data
python scripts/train_ml_models.py

# Evaluate model performance
python scripts/evaluate_ml_models.py
```

**Pre-trained models:** `models/` directory contains ready-to-use `.pkl` files.

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/tokenomics.git
cd tokenomics
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Configure

```bash
cp env.template .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```python
from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
import os
from dotenv import load_dotenv

load_dotenv()

config = TokenomicsConfig.from_env()
platform = TokenomicsPlatform(config=config)

# Run optimized query
result = platform.query("What is machine learning?")
print(f"Response: {result['response']}")
print(f"Tokens: {result['tokens_used']}")
print(f"Cache hit: {result['cache_hit']}")
```

### 4. Try the Playground

```bash
python app.py
# Open http://localhost:5000/playground
```

---

## Components

| Component | Purpose | Key Metric |
|-----------|---------|------------|
| **Memory Layer** | Exact + semantic caching | 20% cache hit rate |
| **Token Orchestrator** | Complexity analysis & budget allocation | 90% accuracy |
| **Bandit Optimizer** | Cost-aware model routing | 75% of savings |
| **LLMLingua Compression** | Reduce long query tokens | 376 tokens saved |
| **Cascading Inference** | Quality-protected model selection | Maintains quality |
| **Token Predictor** | ML-based response length prediction | 72% accuracy |

---

## Test Categories

Our comprehensive test included:

| Category | Count | Purpose | Result |
|----------|-------|---------|--------|
| Simple queries | 10 | Test cheap strategy | ✅ Low cost |
| Medium queries | 10 | Test balanced strategy | ✅ Appropriate routing |
| Complex queries | 10 | Test premium + cascading | ✅ Quality maintained |
| Exact duplicates | 5 | Test exact cache | ✅ **100% hit rate** |
| Semantic variations | 5 | Test semantic cache | ✅ 40% hit rate |
| Long queries (>500 chars) | 5 | Test compression | ✅ All compressed |
| Mixed scenarios | 5 | Edge cases | ✅ Handled correctly |

---

## Project Structure

```
tokenomics/
├── tokenomics/              # Core platform
│   ├── core.py              # Main entry point
│   ├── memory/              # Caching layer
│   ├── orchestrator/        # Token allocation
│   ├── bandit/              # Strategy selection
│   ├── compression/         # LLMLingua integration
│   └── ml/                  # Token prediction
├── templates/               # Web UI
├── static/                  # CSS/JS
├── app.py                   # Flask server
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
└── examples/                # Usage examples
```

---

## Documentation

- **[Architecture](docs/architecture/overview.md)** - System design
- **[Installation](docs/guides/installation.md)** - Detailed setup
- **[Configuration](docs/guides/configuration.md)** - All options
- **[Test Results](COMPREHENSIVE_E2E_TEST_RESULTS.md)** - Full proof of work

---

## API Reference

### `platform.query()`

```python
result = platform.query(
    query="Your question here",
    token_budget=4000,          # Optional: max tokens
    use_cache=True,             # Use memory layer
    use_bandit=True,            # Use strategy selection
    use_compression=True,       # Compress long queries
    use_cost_aware_routing=True # Optimize for cost
)
```

**Returns:**
```python
{
    "response": "...",           # LLM response
    "tokens_used": 250,          # Total tokens
    "cache_hit": False,          # Was this cached?
    "strategy": "cheap",         # Strategy used
    "model": "gpt-4o-mini",      # Model used
    "latency_ms": 1234,          # Response time
}
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Acknowledgments

- [RouterBench](https://github.com/withmartian/routerbench) - Routing methodology
- [LLMLingua](https://github.com/microsoft/LLMLingua) - Compression
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search

---

**Built for the LLM optimization community**
