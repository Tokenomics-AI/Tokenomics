# Tokenomics Platform - Comprehensive Diagnostic Test Documentation

## Overview

This document provides complete documentation for the Tokenomics Platform Diagnostic Test Suite. It explains every metric, component, and behavior tested, enabling developers and stakeholders to understand platform performance and validate functionality.

## Table of Contents

1. [Test Architecture](#test-architecture)
2. [Component Reference](#component-reference)
3. [Metric Definitions](#metric-definitions)
4. [Test Categories](#test-categories)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Test Architecture

### Data Flow

```
Query Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Exact Cache │──│Semantic Cache│──│   Context Injection     │ │
│  │  (Hash Map) │  │   (FAISS)    │  │ (LLMLingua Compression) │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              User Preference Learning                        ││
│  │  (Tone Detection, Format Detection, Confidence Tracking)    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                                │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │  Complexity  │──│ Token Budget  │──│ Knapsack Optimization│  │
│  │   Analysis   │  │  Allocation   │  │  (Component Tokens)  │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │           LLMLingua Query Compression (>200 tokens)         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 BANDIT OPTIMIZER + ROUTERBENCH                   │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │   Strategy   │──│ Cost-Quality  │──│   Reward Learning    │  │
│  │  Selection   │  │    Routing    │  │   (UCB Algorithm)    │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│                                                                  │
│  Strategies: cheap (gpt-4o-mini) | balanced | premium (gpt-4o) │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM PROVIDER                               │
│            (OpenAI / Gemini / vLLM)                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUALITY JUDGE                               │
│         (Compares Optimized vs Baseline Response)               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Response Output + Metrics
```

### Test Execution Flow

1. **Initialize Platform**: Load configuration, instantiate all components
2. **Component Health Check**: Verify each component is operational
3. **Load Test Dataset**: 52 queries across 8 categories
4. **For Each Query**:
   - Run through optimized pipeline
   - Collect all metrics
   - Store result
5. **Aggregate Results**: Calculate summary statistics
6. **Generate Reports**: JSON, CSV, Markdown, HTML

---

## Component Reference

### 1. Memory Layer

The Smart Memory Layer provides intelligent caching and context management.

#### 1.1 Exact Cache

**What it does**: Stores and retrieves responses for identical queries using hash-based lookup.

**How it works**:
- Query is hashed using SHA-256
- Hash is used as key in LRU cache
- Exact match returns cached response immediately

**Trigger conditions**:
- Query string matches exactly (case-sensitive)
- Cache entry hasn't expired (TTL)

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `exact_cache_hits` | Number of exact matches | Increases over time |
| `cache_type: "exact"` | Query was exact match | For repeated queries |

#### 1.2 Semantic Cache (Vector Store)

**What it does**: Finds semantically similar queries using vector embeddings.

**How it works**:
1. Query is embedded using sentence-transformers
2. FAISS index searches for nearest neighbors
3. Similarity score determines action:
   - **>0.92**: Semantic direct return (use cached response)
   - **0.75-0.92**: Context injection (add cached context)
   - **<0.75**: Cache miss (no semantic match)

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `semantic_direct_hits` | High similarity matches | For paraphrased queries |
| `semantic_context_hits` | Medium similarity (context added) | For related queries |
| `similarity` | Cosine similarity score | 0.75-1.0 |

#### 1.3 Context Injection

**What it does**: Enriches new queries with relevant context from similar past queries.

**How it works**:
1. Find semantically similar queries (0.75-0.92 similarity)
2. Retrieve their responses as context
3. Compress context using LLMLingua-2
4. Inject compressed context into prompt

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `context_injected` | Context was added to prompt | true for medium similarity |
| `context_tokens_added` | Tokens from retrieved context | 50-500 |
| `context_compressed` | Context was compressed | true when context retrieved |
| `context_compression_ratio` | compressed/original | 0.3-0.5 |

#### 1.4 LLMLingua Compression

**What it does**: Compresses long contexts and queries using Microsoft's LLMLingua-2 model.

**How it works**:
- Uses BERT-based model to identify token importance
- Removes low-importance tokens while preserving meaning
- Achieves 40-70% compression with minimal quality loss

**Trigger conditions**:
- **Context compression**: When context is retrieved from semantic cache
- **Query compression**: When query > 200 tokens OR > 800 characters

**Metrics**:
| Metric | Description | Formula | Good Value |
|--------|-------------|---------|------------|
| `context_compressed` | Context was compressed | Boolean | true for context injection |
| `context_original_tokens` | Pre-compression size | Token count | Varies |
| `context_compressed_tokens` | Post-compression size | Token count | 30-60% of original |
| `context_compression_ratio` | Compression efficiency | compressed / original | 0.3-0.5 |
| `query_compressed` | Query was compressed | Boolean | true for long queries |
| `query_original_tokens` | Original query tokens | Token count | >200 triggers compression |
| `query_compressed_tokens` | Compressed query tokens | Token count | <100 typically |
| `query_compression_ratio` | Query compression efficiency | compressed / original | 0.2-0.4 |
| `total_compression_savings` | Tokens saved by compression | original - compressed | 100-500 |

#### 1.5 User Preference Learning

**What it does**: Learns user's preferred tone and format from interactions.

**How it works**:
- Analyzes query patterns (formal/casual, list/code/paragraph)
- Builds confidence over multiple interactions
- Applies learned preferences to future responses

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `preference_used` | Preferences applied | true after 3+ queries |
| `preference_tone` | Detected tone | neutral/formal/casual/simple |
| `preference_format` | Detected format | paragraph/list/code/concise |

---

### 2. Orchestrator

The Token-Aware Orchestrator manages token allocation and prompt construction.

#### 2.1 Complexity Analysis

**What it does**: Classifies query complexity to determine resource allocation.

**How it works**:
- Analyzes query length, keywords, structure
- Categories: `simple`, `medium`, `complex`

**Complexity indicators**:
| Complexity | Characteristics | Token Budget |
|------------|-----------------|--------------|
| Simple | Short, factual questions | 500-1000 |
| Medium | Explanations, comparisons | 1000-2000 |
| Complex | Analysis, multi-part questions | 2000-4000 |

#### 2.2 Token Budget Allocation

**What it does**: Allocates token budget across prompt components.

**How it works**:
- Uses knapsack optimization
- Balances: system prompt, query, context, response reserve

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `complexity` | Detected complexity level | simple/medium/complex |
| `token_budget` | Total allocated budget | 500-4000 |
| `max_response_tokens` | Max output tokens | 100-500 |
| `token_efficiency` | output_tokens / input_tokens | 0.5-2.0 |

---

### 3. Bandit Optimizer + RouterBench

The Multi-Armed Bandit optimizes strategy selection using cost-quality routing.

#### 3.1 Strategy Selection

**Available strategies**:
| Strategy | Model | Max Tokens | Temperature | Use Case |
|----------|-------|------------|-------------|----------|
| `cheap` | gpt-4o-mini | 300 | 0.3 | Simple factual queries |
| `balanced` | gpt-4o-mini | 500 | 0.5 | General queries |
| `premium` | gpt-4o | 1000 | 0.7 | Complex analysis |

**Selection algorithm**: UCB (Upper Confidence Bound)
- Balances exploration (trying new strategies) vs exploitation (using best known)
- Updates based on reward feedback

#### 3.2 RouterBench Cost-Quality Routing

**What it does**: Optimizes the cost-quality tradeoff for each query.

**Reward formula**:
```
reward = quality_score - (lambda * cost)
```

Where:
- `quality_score`: Response quality (0-1)
- `lambda`: Cost sensitivity parameter (default: 0.001)
- `cost`: Actual API cost in USD

**Metrics**:
| Metric | Description | Formula | Good Value |
|--------|-------------|---------|------------|
| `strategy_selected` | Chosen strategy arm | cheap/balanced/premium | Matches complexity |
| `strategy_reward` | Calculated reward | quality - lambda*cost | 0.9-1.0 |
| `cost_per_query` | API cost in USD | tokens * price_per_token | $0.00001-$0.001 |
| `routerbench_efficiency` | Overall efficiency | quality / cost | Higher is better |

---

### 4. Quality Judge

Evaluates response quality by comparing optimized vs baseline.

**How it works**:
1. Run query through optimized pipeline
2. Run same query without optimization (baseline)
3. Compare responses using LLM-as-judge
4. Return winner and confidence

**Metrics**:
| Metric | Description | Good Value |
|--------|-------------|------------|
| `quality_winner` | Which response won | optimized/baseline/equivalent |
| `quality_confidence` | Judge certainty | 0.8-1.0 |
| `quality_explanation` | Reasoning | Descriptive text |

---

## Metric Definitions

### Summary Metrics

| Metric | Formula | Description | Good Value |
|--------|---------|-------------|------------|
| `success_rate` | (successful / total) * 100 | % of queries that completed | >95% |
| `cache_hit_rate` | (exact + semantic_direct) / total * 100 | % returning cached response | Increases over time |
| `average_savings_percentage` | (tokens_saved / baseline_tokens) * 100 | Average token reduction | 10-50% |
| `total_compressions` | context_compressions + query_compressions | Total LLMLingua uses | >0 for long content |
| `average_reward` | mean(all rewards) | Bandit performance | >0.9 |

### Per-Query Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `latency_ms` | Float | Total query processing time |
| `input_tokens` | Int | Tokens sent to LLM |
| `output_tokens` | Int | Tokens received from LLM |
| `total_tokens` | Int | input + output tokens |
| `tokens_saved` | Int | Estimated baseline - actual |
| `savings_percentage` | Float | % tokens saved |

---

## Test Categories

### Category 1: Exact Cache (6 queries)

**Purpose**: Validate exact match caching

**Test design**:
- 3 unique queries
- Each repeated once

**Expected behavior**:
- First occurrence: cache miss, store response
- Repeat: exact cache hit, zero LLM tokens

**Success indicators**:
- `cache_type: "exact"` for repeats
- `latency_ms` < 100ms for cache hits

### Category 2: Semantic Cache (8 queries)

**Purpose**: Validate semantic similarity matching

**Test design**:
- Seed queries followed by similar variants
- Different similarity levels

**Expected behavior**:
- High similarity (>0.85): semantic direct hit
- Medium similarity (0.75-0.85): context injection
- Low similarity (<0.75): cache miss

**Success indicators**:
- `similarity` values in expected ranges
- Context compression for medium similarity

### Category 3: LLMLingua Compression (8 queries)

**Purpose**: Validate compression functionality

**Test design**:
- 4 long queries (>800 chars)
- 2 medium queries (should NOT compress)
- 2 context compression tests

**Expected behavior**:
- Long queries: `query_compressed: true`
- Compression ratio: 0.3-0.5
- Tokens saved: 100-500 per query

**Success indicators**:
- `total_compressions` > 4
- `compression_tokens_saved` > 0
- Compression ratios < 0.6

### Category 4: Orchestrator (8 queries)

**Purpose**: Validate complexity analysis and token allocation

**Test design**:
- 3 simple queries ("Hi", "2+2?")
- 2 medium queries (comparisons)
- 2 complex queries (detailed analysis)
- 1 constrained query

**Expected behavior**:
- Simple: `complexity: "simple"`, small budget
- Complex: `complexity: "complex"`, large budget

**Success indicators**:
- Correct complexity classification
- Token usage < budget

### Category 5: Bandit + RouterBench (8 queries)

**Purpose**: Validate strategy selection and cost routing

**Test design**:
- 3 cheap strategy triggers
- 2 balanced triggers
- 2 premium triggers
- 1 exploration test

**Expected behavior**:
- Simple factual: cheap strategy
- Complex analysis: premium strategy
- Positive rewards for all

**Success indicators**:
- All strategies used
- `average_reward` > 0.9
- Cost increases with complexity

### Category 6: Preference Learning (6 queries)

**Purpose**: Validate tone and format detection

**Test design**:
- Formal vs casual tone queries
- List vs code vs paragraph format hints

**Expected behavior**:
- `preference_tone` detected
- `preference_format` detected

**Success indicators**:
- Multiple unique tones detected
- Multiple unique formats detected

### Category 7: Edge Cases (4 queries)

**Purpose**: Validate error handling

**Test design**:
- Empty query
- Minimal query ("x")
- Special characters
- Repetitive content

**Expected behavior**:
- Graceful handling, no crashes
- Appropriate error messages for invalid input

### Category 8: Quality (4 queries)

**Purpose**: Validate quality judge

**Test design**:
- Creative, simplified, technical, comprehensive queries

**Expected behavior**:
- Quality comparison runs
- `quality_winner` populated
- High confidence scores

---

## Interpreting Results

### Overall Health Indicators

| Indicator | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| Success Rate | >95% | 80-95% | <80% |
| Cache Hit Rate | >10% | 5-10% | <5% |
| LLMLingua Active | Yes | Fallback | Error |
| Avg Reward | >0.9 | 0.8-0.9 | <0.8 |

### What Good Results Look Like

1. **All components healthy** in health check
2. **LLMLingua showing compressions** for long queries
3. **Exact cache hits** on repeat queries
4. **Semantic matches** for similar queries
5. **All strategies used** by bandit
6. **Positive token savings** in most categories

### Warning Signs

1. **0 compressions** but LLMLingua available
   - Check compression thresholds
   - Verify queries are long enough

2. **No cache hits** after repeated queries
   - Check exact cache initialization
   - Verify hash consistency

3. **All cheap strategy**
   - Query complexity not varying enough
   - Check complexity analysis thresholds

4. **Negative savings**
   - Context injection adding more than saving
   - Review similarity thresholds

---

## Troubleshooting Guide

### LLMLingua Not Compressing

**Symptoms**: `total_compressions: 0` despite long queries

**Causes**:
1. LLMLingua not initialized
2. Queries below threshold (need >200 tokens or >800 chars)
3. Model loading failure

**Solutions**:
1. Check component health for LLMLingua status
2. Review `llmlingua_compressor.py` initialization
3. Verify model download completed

### Cache Not Working

**Symptoms**: No cache hits even for identical queries

**Causes**:
1. `use_cache=False` in query call
2. Cache TTL expired
3. Hash collision (rare)

**Solutions**:
1. Verify cache configuration
2. Check exact_cache.stats() for entries
3. Enable debug logging

### Poor Token Savings

**Symptoms**: `average_savings_percentage` < 5%

**Causes**:
1. Queries too short to optimize
2. Context injection costs exceeding benefits
3. Strategy selection not optimal

**Solutions**:
1. Review query distribution
2. Adjust similarity thresholds
3. Tune bandit exploration rate

### High Latency

**Symptoms**: `average_latency_ms` > 10000

**Causes**:
1. Model download on first use
2. Network issues
3. Rate limiting

**Solutions**:
1. Pre-warm models
2. Check API connectivity
3. Add retry logic with backoff

---

## Report Formats

### JSON Report
- **Use for**: Programmatic analysis, data pipelines
- **Contains**: Complete raw data, all metrics

### CSV Report
- **Use for**: Spreadsheet analysis, quick overview
- **Contains**: One row per query, key metrics

### Markdown Report
- **Use for**: Documentation, sharing
- **Contains**: Formatted summary, tables, details

### HTML Report
- **Use for**: Presentations, visual analysis
- **Contains**: Interactive dashboard, charts

---

## Appendix: Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend | gemini |
| `LLM_MODEL` | Model name | gemini-1.5-flash |
| `ENABLE_LLMLINGUA` | Enable compression | true |
| `LLMLINGUA_COMPRESSION_RATIO` | Target ratio | 0.4 |
| `ORCHESTRATOR_DEFAULT_BUDGET` | Token budget | 4000 |
| `BANDIT_ALGORITHM` | UCB/epsilon/thompson | ucb |
| `BANDIT_EXPLORATION_RATE` | Exploration rate | 0.1 |

### Test Execution

```bash
# Run in WSL with LLMLingua
wsl -d Ubuntu -e bash -c "cd /mnt/d/Nayan/Tokenomics/Prototype && \
    source .venv-wsl/bin/activate && \
    python3 test_comprehensive_showcase.py"
```

---

*Generated for Tokenomics Platform v1.0*



