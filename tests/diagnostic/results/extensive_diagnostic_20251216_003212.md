# Tokenomics Platform - Extensive Diagnostic Test Report

**Generated:** 2025-12-16T00:32:12.365719

**Platform Version:** 1.0

**LLM Provider:** openai (gpt-4o-mini)

**Test Duration:** 0.0 seconds


## Executive Summary

| Metric | Value |
|--------|-------|
| Total Queries | 32 |
| Passed | 17 (53.1%) |
| Failed | 15 |
| Total Tokens Used | 9,033 |
| Total Savings | 7,665 tokens |
| Savings Rate | 45.9% |


## Token Savings Breakdown

| Component | Tokens Saved | Percentage | Description |
|-----------|--------------|------------|-------------|
| Memory Layer | 4,142 | 54.0% | Tokens saved from exact and semantic cache hits |
| Orchestrator | 3,435 | 44.8% | Tokens saved from optimized token allocation |
| Bandit | 88 | 1.1% | Tokens saved from strategy selection |
| Compression | 1,854 | 24.2% | Tokens saved from LLMLingua compression |


## Component Analysis

### Memory Layer [ISSUES]

- **Queries Processed:** 32
- **Tokens Saved:** 4,142


**Metrics:**
- exact_cache_hits: 1
- semantic_direct_hits: 6
- semantic_context_hits: 4
- total_cache_hits: 11
- cache_hit_rate: 34.4%
- tokens_saved_from_cache: 4142


**Issues Found:**
- Cache type validation failed for 7 queries

### Compression [ISSUES]

- **Queries Processed:** 32
- **Tokens Saved:** 1,854


**Metrics:**
- queries_compressed: 1
- contexts_compressed: 4
- total_compression_events: 5
- total_tokens_saved: 1854


**Issues Found:**
- Compression expected but not triggered for 1 queries

### Orchestrator [ISSUES]

- **Queries Processed:** 32
- **Tokens Saved:** 3,435


**Metrics:**
- simple_queries: 30
- medium_queries: 0
- complex_queries: 2
- avg_token_budget: 3031.25
- total_output_tokens: 7281
- tokens_saved: 3435


**Issues Found:**
- Complexity mismatch for 3 queries

### Bandit [ISSUES]

- **Queries Processed:** 32
- **Tokens Saved:** 88


**Metrics:**
- cheap_selections: 32
- balanced_selections: 0
- premium_selections: 0
- avg_reward: 0.9223
- tokens_saved: 88


**Issues Found:**
- Strategy selection mismatch for 2 queries

### Quality Judge [OK]

- **Queries Processed:** 25
- **Tokens Saved:** 0


**Metrics:**
- comparisons_run: 25
- optimized_wins: 18
- equivalent_results: 5
- baseline_wins: 2
- quality_maintained_rate: 92.0%
- avg_confidence: 0.86

### Preferences [ISSUES]

- **Queries Processed:** 32
- **Tokens Saved:** 0


**Metrics:**
- tone_detections: 30
- format_detections: 17
- preferences_learned: 47


**Issues Found:**
- Tone detection failed for 3 queries


## Issues Found

| Test ID | Phase | Issue |
|---------|-------|-------|
| final_exact_1 | 10_final_validation | Cache type mismatch: expected 'exact', got 'None' |
| final_semantic_1 | 10_final_validation | Cache type mismatch: expected 'semantic_direct', got 'context' |
| final_semantic_1 | 10_final_validation | Similarity too low: expected >= 0.85, got 0.822 |
| seed_rest | 1_cache_seeding | Complexity mismatch: expected 'medium', got 'simple' |
| exact_python_1 | 2_exact_cache | Cache type mismatch: expected 'exact', got 'semantic_direct' |
| exact_python_2 | 2_exact_cache | Cache type mismatch: expected 'exact', got 'semantic_direct' |
| context_python_packages | 4_context_injection | Cache type mismatch: expected 'context', got 'None' |
| context_ml_vs_dl | 4_context_injection | Cache type mismatch: expected 'context', got 'None' |
| compress_long_query | 5_compression | Expected compression but none occurred |
| compress_follow_up | 5_compression | Cache type mismatch: expected 'context', got 'None' |
| pref_formal | 6_preferences | Tone mismatch: expected 'formal', got 'simple' |
| pref_casual | 6_preferences | Tone mismatch: expected 'casual', got 'simple' |
| pref_technical | 6_preferences | Tone mismatch: expected 'technical', got 'simple' |
| complexity_medium_1 | 7_complexity | Complexity mismatch: expected 'medium', got 'simple' |
| complexity_medium_1 | 7_complexity | Strategy mismatch: expected 'balanced', got 'cheap' |
| complexity_medium_2 | 7_complexity | Complexity mismatch: expected 'medium', got 'simple' |
| complexity_complex_1 | 7_complexity | Strategy mismatch: expected 'premium', got 'cheap' |


## Phase-by-Phase Results

### 10_final_validation (0/2 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| final_exact_1 | none | 311 | cheap | 11 | FAIL |
| final_semantic_1 | context | 614 | cheap | 523 | FAIL |

### 1_cache_seeding (3/4 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| seed_python | semantic_direct | 0 | cheap | 311 | PASS |
| seed_ml | none | 167 | cheap | 76 | PASS |
| seed_rest | none | 313 | cheap | 212 | FAIL |
| seed_database | none | 313 | cheap | 88 | PASS |

### 2_exact_cache (1/3 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| exact_python_1 | semantic_direct | 0 | cheap | 311 | FAIL |
| exact_python_2 | semantic_direct | 0 | cheap | 311 | FAIL |
| exact_rest | exact | 0 | cheap | 313 | PASS |

### 3_semantic_direct (3/3 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| semantic_direct_python_1 | semantic_direct | 0 | cheap | 311 | PASS |
| semantic_direct_python_2 | semantic_direct | 0 | cheap | 311 | PASS |
| semantic_direct_ml | semantic_direct | 0 | cheap | 167 | PASS |

### 4_context_injection (1/3 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| context_python_packages | none | 316 | cheap | 212 | FAIL |
| context_rest_benefits | context | 622 | cheap | 531 | PASS |
| context_ml_vs_dl | none | 318 | cheap | 212 | FAIL |

### 5_compression (0/2 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| compress_long_query | none | 455 | cheap | 212 | FAIL |
| compress_follow_up | none | 315 | cheap | 212 | FAIL |

### 6_preferences (2/5 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| pref_formal | none | 317 | cheap | 120 | FAIL |
| pref_casual | none | 315 | cheap | 112 | FAIL |
| pref_technical | context | 618 | cheap | 527 | FAIL |
| pref_list | none | 315 | cheap | 212 | PASS |
| pref_code | none | 314 | cheap | 212 | PASS |

### 7_complexity (2/5 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| complexity_simple_1 | none | 311 | cheap | 210 | PASS |
| complexity_simple_2 | none | 236 | cheap | 6 | PASS |
| complexity_medium_1 | none | 318 | cheap | 212 | FAIL |
| complexity_medium_2 | none | 316 | cheap | 212 | FAIL |
| complexity_complex_1 | none | 358 | cheap | 300 | FAIL |

### 8_quality_judge (3/3 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| quality_test_1 | none | 314 | cheap | 212 | PASS |
| quality_test_2 | none | 315 | cheap | 56 | PASS |
| quality_test_3 | none | 313 | cheap | 212 | PASS |

### 9_token_budget (2/2 passed)

| Test ID | Cache | Tokens | Strategy | Savings | Passed |
|---------|-------|--------|----------|---------|--------|
| budget_low | none | 312 | cheap | 212 | PASS |
| budget_high | context | 617 | cheap | 526 | PASS |


## Conclusion

The Tokenomics Platform requires attention to several issues.

Token savings of **45.9%** demonstrates the platform's value proposition.