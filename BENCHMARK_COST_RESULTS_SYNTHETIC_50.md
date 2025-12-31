# Cost Benchmark (Synthetic-50)

**Generated:** 2025-12-31 05:24:55  
**Git Commit:** `5a1aac1`

---

## Benchmark Configuration

| Setting | Value |
|---------|-------|
| Baseline Model | `gpt-4o` |
| Tokenomics Default Model | `gpt-4o-mini` |
| Tokenomics Premium Model | `gpt-4o` |
| Temperature | 0.3 (baseline) / varies (tokenomics) |
| Max Tokens | 512 (baseline) / varies (tokenomics) |
| Cache | Disabled (baseline) / Enabled (tokenomics) |
| Compression | Disabled (baseline) / Enabled (tokenomics) |
| Routing | Disabled (baseline) / Bandit UCB (tokenomics) |

**Cache Rules:**
- BASELINE: Cache completely disabled (no read, no write)
- TOKENOMICS: Cache starts empty (cold), warms across prompts

---

## Workload Summary

50 synthetic queries originally used for decision accuracy validation. Categories include simple, medium, complex queries, exact duplicates for cache testing, semantic variations, long queries for compression testing, and mixed scenarios.

| Category | Count |
|----------|-------|
| complex | 10 |
| exact_duplicate | 5 |
| long_query | 5 |
| medium | 10 |
| mixed | 5 |
| semantic_variation | 5 |
| simple | 10 |
| **Total** | **50** |

---

## Aggregate Summary

### Cost & Token Savings

| Metric | Value |
|--------|-------|
| Total Baseline Tokens | 23,039 |
| Total Tokenomics Tokens | 12,548 |
| **Mean Token Savings** | **33.7%** |
| Median Token Savings | 36.0% |
| Total Baseline Cost | $0.217602 |
| Total Tokenomics Cost | $0.013989 |
| **Total Cost Savings** | **$0.203614** |
| **Mean Cost Savings** | **90.7%** |
| Median Cost Savings | 96.0% |

### Cache Performance

| Metric | Value |
|--------|-------|
| Cache Hit Rate | 20.0% |
| Exact Cache Hits | 6 |
| Semantic Cache Hits | 4 |
| Cache Misses | 40 |

### Routing Distribution

| Strategy | Count | Percentage |
|----------|-------|------------|
| Cheap (gpt-4o-mini) | 28 | 56.0% |
| Balanced (gpt-4o-mini) | 12 | 24.0% |
| Premium (gpt-4o) | 10 | 20.0% |

### Quality & Reliability

| Metric | Value |
|--------|-------|
| Successful Prompts | 50/50 |
| Quality Failures | 3 |
| Mean Latency Delta | -1524ms |

---

## Breakdown by Complexity

| Complexity | Count | Median Cost Savings | Cache Hit Rate | Escalation Rate |
|------------|-------|---------------------|----------------|-----------------|
| complex | 16 | 97.1% | 0.0% | 0.0% |
| medium | 16 | 95.3% | 25.0% | 0.0% |
| simple | 18 | 94.0% | 33.3% | 0.0% |

---

## Breakdown by Category

| Category | Count | Median Cost Savings | Cache Hit Rate |
|----------|-------|---------------------|----------------|
| complex | 10 | 97.1% | 0.0% |
| exact_duplicate | 5 | 100.0% | 100.0% |
| long_query | 5 | 97.6% | 0.0% |
| medium | 10 | 94.6% | 0.0% |
| mixed | 5 | 95.4% | 20.0% |
| semantic_variation | 5 | 100.0% | 80.0% |
| simple | 10 | 90.8% | 0.0% |

---

## Breakdown by Routing Outcome

| Outcome | Count | Percentage | Median Cost Savings |
|---------|-------|------------|---------------------|
| Cheaper Model Used | 46 | 92.0% | 96.5% |
| Baseline Model Used | 4 | 8.0% | 28.3% |
| Not Escalated | 50 | 100.0% | 96.0% |

---

## Per-Prompt Results

| ID | Category | Baseline Tokens | Tokenomics Tokens | Token Savings | Baseline Cost | Tokenomics Cost | Cost Savings | Cache | Strategy |
|----|----------|-----------------|-------------------|---------------|---------------|-----------------|--------------|-------|----------|
| 1 | simple | 22 | 22 | 0.0% | $0.000115 | $0.000115 | 0.0% | miss | premium |
| 2 | simple | 21 | 21 | 0.0% | $0.000105 | $0.000006 | 94.0% | miss | cheap |
| 3 | simple | 31 | 48 | -54.8% | $0.000213 | $0.000023 | 89.2% | miss | premium |
| 4 | simple | 163 | 112 | 31.3% | $0.001532 | $0.000061 | 96.0% | miss | cheap |
| 5 | simple | 18 | 25 | -38.9% | $0.000097 | $0.000010 | 89.7% | miss | balanced |
| 6 | simple | 25 | 48 | -92.0% | $0.000138 | $0.000022 | 84.0% | miss | balanced |
| 7 | simple | 23 | 23 | 0.0% | $0.000117 | $0.000117 | 0.0% | miss | cheap |
| 8 | simple | 56 | 72 | -28.6% | $0.000463 | $0.000037 | 91.9% | miss | cheap |
| 9 | simple | 30 | 20 | 33.3% | $0.000203 | $0.000006 | 97.0% | miss | cheap |
| 10 | simple | 49 | 49 | 0.0% | $0.000385 | $0.000023 | 94.0% | miss | cheap |
| 11 | medium | 567 | 454 | 19.9% | $0.005505 | $0.000262 | 95.2% | miss | premium |
| 12 | medium | 555 | 442 | 20.4% | $0.005415 | $0.000257 | 95.3% | miss | balanced |
| 13 | medium | 334 | 401 | -20.1% | $0.003197 | $0.000232 | 92.7% | miss | cheap |
| 14 | medium | 422 | 172 | 59.2% | $0.004077 | $0.000095 | 97.7% | miss | cheap |
| 15 | medium | 397 | 574 | -44.6% | $0.003850 | $0.000337 | 91.2% | miss | cheap |
| 16 | medium | 501 | 409 | 18.4% | $0.004860 | $0.000236 | 95.1% | miss | premium |
| 17 | medium | 617 | 458 | 25.8% | $0.006035 | $0.000267 | 95.6% | miss | cheap |
| 18 | medium | 330 | 444 | -34.5% | $0.003180 | $0.000259 | 91.8% | miss | cheap |
| 19 | medium | 360 | 406 | -12.8% | $0.003480 | $0.000236 | 93.2% | miss | balanced |
| 20 | medium | 540 | 539 | 0.2% | $0.005273 | $0.000316 | 94.0% | miss | premium |
| 21 | complex | 903 | 458 | 49.3% | $0.008618 | $0.000250 | 97.1% | miss | balanced |
| 22 | complex | 937 | 482 | 48.6% | $0.009025 | $0.000269 | 97.0% | miss | premium |
| 23 | complex | 853 | 490 | 42.6% | $0.008200 | $0.000274 | 96.7% | miss | cheap |
| 24 | complex | 768 | 471 | 38.7% | $0.007305 | $0.000260 | 96.4% | miss | balanced |
| 25 | complex | 807 | 294 | 63.6% | $0.007710 | $0.000155 | 98.0% | miss | cheap |
| 26 | complex | 986 | 420 | 57.4% | $0.009470 | $0.000229 | 97.6% | miss | premium |
| 27 | complex | 851 | 588 | 30.9% | $0.008120 | $0.000329 | 95.9% | miss | premium |
| 28 | complex | 1060 | 450 | 57.5% | $0.010217 | $0.000247 | 97.6% | miss | balanced |
| 29 | complex | 862 | 517 | 40.0% | $0.008297 | $0.000291 | 96.5% | miss | balanced |
| 30 | complex | 964 | 376 | 61.0% | $0.009250 | $0.000202 | 97.8% | miss | premium |
| 31 | exact_duplicate | 22 | 0 | 100.0% | $0.000115 | $0.000000 | 100.0% | exact | cheap |
| 32 | exact_duplicate | 21 | 0 | 100.0% | $0.000105 | $0.000000 | 100.0% | exact | cheap |
| 33 | exact_duplicate | 405 | 0 | 100.0% | $0.003930 | $0.000000 | 100.0% | exact | cheap |
| 34 | exact_duplicate | 51 | 0 | 100.0% | $0.000405 | $0.000000 | 100.0% | exact | cheap |
| 35 | exact_duplicate | 472 | 0 | 100.0% | $0.004570 | $0.000000 | 100.0% | exact | cheap |
| 36 | semantic_variation | 26 | 26 | 0.0% | $0.000140 | $0.000008 | 94.0% | miss | cheap |
| 37 | semantic_variation | 23 | 0 | 100.0% | $0.000117 | $0.000000 | 100.0% | semantic_direct | cheap |
| 38 | semantic_variation | 512 | 0 | 100.0% | $0.004985 | $0.000000 | 100.0% | semantic_direct | cheap |
| 39 | semantic_variation | 145 | 132 | 9.0% | $0.001352 | $0.000057 | 95.8% | context | cheap |
| 40 | semantic_variation | 445 | 0 | 100.0% | $0.004315 | $0.000000 | 100.0% | semantic_direct | cheap |
| 41 | long_query | 1231 | 534 | 56.6% | $0.011297 | $0.004905 | 56.6% | miss | premium |
| 42 | long_query | 893 | 332 | 62.8% | $0.008037 | $0.000177 | 97.8% | miss | balanced |
| 43 | long_query | 1125 | 449 | 60.1% | $0.010305 | $0.000246 | 97.6% | miss | balanced |
| 44 | long_query | 992 | 379 | 61.8% | $0.008900 | $0.000204 | 97.7% | miss | balanced |
| 45 | long_query | 992 | 482 | 51.4% | $0.008960 | $0.000264 | 97.0% | miss | balanced |
| 46 | mixed | 22 | 0 | 100.0% | $0.000115 | $0.000000 | 100.0% | exact | cheap |
| 47 | mixed | 32 | 33 | -3.1% | $0.000215 | $0.000014 | 93.7% | miss | cheap |
| 48 | mixed | 703 | 253 | 64.0% | $0.006820 | $0.002320 | 66.0% | miss | cheap |
| 49 | mixed | 360 | 277 | 23.1% | $0.003465 | $0.000158 | 95.4% | miss | cheap |
| 50 | mixed | 515 | 366 | 28.9% | $0.005000 | $0.000211 | 95.8% | miss | cheap |

---

## Caveats

1. **Baseline Configuration**: Baseline runs with cache completely disabled (no read, no write). This ensures baseline costs are not artificially reduced by prior tokenomics runs.

2. **Cache Cold Start**: Tokenomics cache starts empty at benchmark start. The cache is NOT cleared between prompts, allowing natural cache warming.

3. **Workload Dependence**: Results are specific to this workload (50 prompts). This is a synthetic workload, not a production distribution. Production savings depend on actual query patterns, repetition, and complexity distribution.

4. **Cache Benefits**: Caching benefits may be limited if prompts are not repetitive. This synthetic workload includes intentional duplicates to test caching. Cache hit rates depend on query similarity and repetition patterns.

5. **Quality Metric**: Quality is measured using `minlen` check. This is a proxy metric, not a comprehensive quality evaluation.

6. **Latency Variance**: Latency measurements include API round-trip time and may vary based on network conditions and API load.

7. **Model Pricing**: Costs are estimated based on OpenAI's published pricing as of the benchmark date. Actual costs may vary.

---

## Reproduction Steps

```bash
# From repository root
cd /path/to/Tokenomics

# Ensure .env has OPENAI_API_KEY set
source venv/bin/activate

# Run benchmark
python scripts/run_cost_benchmark.py \
    --workload benchmarks/synthetic_accuracy_50.json \
    --output BENCHMARK_COST_RESULTS_SYNTHETIC_50.md \
    --quality_check minlen \
    --seed 42 --include_breakdowns
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Valid OpenAI API key

**Expected Runtime:** ~250 seconds (varies with API latency)

---

*Report generated by `scripts/run_cost_benchmark.py`*
