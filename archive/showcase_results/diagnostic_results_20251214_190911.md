# Tokenomics Platform - Comprehensive Diagnostic Test Results

**Test Date:** 2025-12-14 18:54:42
**Duration:** 869.5 seconds

## Executive Summary

This diagnostic test validates all components of the Tokenomics platform using 52 carefully designed queries.

### Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Success Rate | 7.7% | ⚠️ |
| Cache Hit Rate | 100.0% | ✅ |
| LLMLingua Compressions | 0 | ⚠️ |
| Tokens Saved | 1,257 | ✅ |
| Average Reward | 0.0000 | ⚠️ |

## Component Health

| Component | Status | Details |
|-----------|--------|--------|
| Exact Cache | ✅ OK | entries: 0, max_size: 1000 |
| Semantic Cache | ✅ OK | type: FAISSVectorStore |
| LLMLingua | ✅ OK | model: microsoft/llmlingua-2-bert-base-multilingua |
| User Preferences | ✅ OK | confidence: 0.5 |
| Memory Layer | ✅ OK | type: SmartMemoryLayer |
| Orchestrator | ✅ OK | default_budget: 4000, max_context: 8000 |
| Bandit Optimizer | ✅ OK | num_strategies: 3, strategies: ['cheap', 'balanced |
| Quality Judge | ✅ OK | model: gpt-4o |
| LLM Provider | ✅ OK | type: OpenAIProvider |

## Cache Performance

The memory layer implements a tiered caching system:

| Cache Type | Count | Description |
|------------|-------|-------------|
| Exact Match | 3 | Identical query found in cache |
| Semantic Direct | 1 | High similarity (>0.85) - direct return |
| Context Injection | 0 | Medium similarity (0.75-0.85) - context added |
| Cache Miss | 48 | No match found - full LLM call |

## LLMLingua Compression

**Status:** ✅ Active
**Model:** microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank

| Metric | Value |
|--------|-------|
| Context Compressions | 0 |
| Query Compressions | 0 |
| Total Compressions | 0 |
| Avg Context Ratio | 100.00% |
| Avg Query Ratio | 100.00% |
| Tokens Saved | 0 |

## Bandit Optimizer + RouterBench

Strategy selection based on query complexity and cost-quality routing:

| Strategy | Uses | Description |
|----------|------|-------------|
| Cheap | 4 | gpt-4o-mini, low cost, fast |
| Balanced | 0 | gpt-4o-mini, balanced settings |
| Premium | 0 | gpt-4o, high quality |

**Average Reward:** 0.0000
**Total Cost:** $0.000000

## Results by Category

| Category | Success | Cache Hits | Compressions | Tokens Saved |
|----------|---------|------------|--------------|-------------|
| bandit | 0/8 | 0 | 0 | 0 |
| compression | 0/8 | 0 | 0 | 0 |
| edge_case | 0/4 | 0 | 0 | 0 |
| exact_cache | 3/6 | 3 | 0 | 940 |
| orchestrator | 0/8 | 0 | 0 | 0 |
| preferences | 0/6 | 0 | 0 | 0 |
| quality | 0/4 | 0 | 0 | 0 |
| semantic_cache | 1/8 | 1 | 0 | 317 |

## Detailed Query Results

<details>
<summary>Click to expand all queries</summary>

### Query 1: exact_cache/first_occurrence
- **Query:** `What is Python programming language?...`
- **Status:** ❌
- **Cache:** miss
