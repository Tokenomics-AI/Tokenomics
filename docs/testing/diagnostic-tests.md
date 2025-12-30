# Comprehensive Diagnostic Test Documentation

## Overview

This document provides a complete explanation of the Comprehensive Diagnostic Test for the Tokenomics Platform. This test validates **every component** and **every capability** of the platform, making it ideal for product showcase and validation.

**Purpose**: Validate that all platform components are working correctly and demonstrate the full range of capabilities and savings.

**Test Coverage**: 
- ✅ Memory Layer (exact cache, semantic cache, context injection, LLM Lingua, preferences)
- ✅ Orchestrator (complexity analysis, token allocation, compression, routing)
- ✅ Bandit Optimizer (strategy selection, RouterBench routing, learning)
- ✅ Quality Judge (if enabled)
- ✅ Component-level savings tracking

---

## Table of Contents

1. [Test Structure](#test-structure)
2. [Memory Layer Metrics](#memory-layer-metrics)
3. [Orchestrator Metrics](#orchestrator-metrics)
4. [Bandit Optimizer Metrics](#bandit-optimizer-metrics)
5. [Compression Metrics](#compression-metrics)
6. [Savings Metrics](#savings-metrics)
7. [Quality Metrics](#quality-metrics)
8. [Performance Metrics](#performance-metrics)
9. [Test Cases](#test-cases)
10. [Interpreting Results](#interpreting-results)

---

## Test Structure

### Test Categories

The diagnostic test includes test cases organized by category:

1. **Exact Cache**: Tests identical query caching
2. **Semantic Cache Direct**: Tests high-similarity direct returns
3. **Context Injection**: Tests medium-similarity context enhancement
4. **LLM Lingua Query**: Tests query compression
5. **LLM Lingua Context**: Tests context compression
6. **User Preferences**: Tests preference learning
7. **Query Complexity**: Tests complexity detection (simple/medium/complex)
8. **Bandit Selection**: Tests strategy selection
9. **RouterBench**: Tests cost-aware routing
10. **Token Budget**: Tests different budget scenarios
11. **Edge Cases**: Tests edge cases

### Test Execution Flow

```
1. Initialize Platform
   ↓
2. Run Test Cases (for each test case)
   ├─ Execute query through platform
   ├─ Collect all metrics
   ├─ Validate expectations
   └─ Store results
   ↓
3. Run Component Tests
   ├─ Memory Layer Tests
   ├─ Orchestrator Tests
   ├─ Bandit Optimizer Tests
   └─ LLM Lingua Tests
   ↓
4. Calculate Summary Statistics
   ↓
5. Save Results
```

---

## Memory Layer Metrics

The Memory Layer is the first line of defense against unnecessary token usage. It provides multiple tiers of caching and context retrieval.

### Cache Hit Metrics

#### `cache_hit` (boolean)
- **What it is**: Whether the query resulted in a cache hit (no LLM call needed or context-enhanced)
- **How it works**: 
  - `true`: Query matched cached content (exact or semantic)
  - `false`: Query required full LLM call
- **What it signifies**: 
  - `true` = Tokens saved (either 100% for direct hits or partial for context-enhanced)
  - `false` = Full LLM call required
- **Good value**: Higher is better (indicates effective caching)

#### `cache_type` (string)
- **What it is**: Type of cache hit that occurred
- **Possible values**:
  - `"exact"`: Identical query found in cache
  - `"semantic_direct"`: High similarity (>0.85) - cached response returned directly
  - `"context"`: Medium similarity (0.75-0.85) - cached context injected into prompt
  - `None`: No cache hit (cache miss)
- **How it works**:
  - Exact: Hash-based lookup for identical queries
  - Semantic Direct: Vector similarity search with threshold >0.85
  - Context: Vector similarity search with threshold 0.75-0.85, context compressed and injected
- **What it signifies**:
  - `"exact"`: 100% token savings (0 tokens used)
  - `"semantic_direct"`: ~100% token savings (0 tokens used, cached response returned)
  - `"context"`: Partial savings (context added to input, but reduces output tokens)
  - `None`: No savings from cache
- **Good value**: More `"exact"` and `"semantic_direct"` hits = better

#### `cache_tier` (string)
- **What it is**: Simplified cache classification for reporting
- **Possible values**: `"exact"`, `"semantic"`, `"capsule"`, `"none"`
- **Mapping**:
  - `cache_type="exact"` → `cache_tier="exact"`
  - `cache_type="semantic_direct"` → `cache_tier="semantic"`
  - `cache_type="context"` → `cache_tier="capsule"`
  - `cache_type=None` → `cache_tier="none"`
- **What it signifies**: Same as `cache_type`, but simplified for reporting

#### `similarity` (float, 0.0-1.0)
- **What it is**: Semantic similarity score for semantic cache matches
- **How it works**: 
  - Calculated using cosine similarity of query embeddings
  - Only present for semantic cache hits (not exact matches)
- **What it signifies**:
  - `>0.85`: High similarity - direct return (no LLM call)
  - `0.75-0.85`: Medium similarity - context injection
  - `<0.75`: Low similarity - no cache benefit
- **Good value**: Higher similarity = better match quality

### Memory Operations Metrics

#### `memory_metrics.exact_cache_hits` (integer)
- **What it is**: Count of exact cache hits in this query
- **How it works**: Incremented when identical query found
- **What it signifies**: Number of exact matches found
- **Good value**: 1 for exact hits, 0 otherwise

#### `memory_metrics.semantic_direct_hits` (integer)
- **What it is**: Count of semantic direct return hits
- **How it works**: Incremented when similarity >0.85
- **What it signifies**: High-quality semantic matches that bypass LLM
- **Good value**: 1 for semantic direct hits, 0 otherwise

#### `memory_metrics.semantic_context_hits` (integer)
- **What it is**: Count of context injection hits
- **How it works**: Incremented when similarity 0.75-0.85
- **What it signifies**: Medium-quality matches used as context
- **Good value**: 1 for context hits, 0 otherwise

#### `memory_metrics.semantic_matches_found` (integer)
- **What it is**: Total number of semantic matches found (regardless of threshold)
- **How it works**: Count of all matches from vector search
- **What it signifies**: How many similar queries exist in cache
- **Good value**: Higher = more cached knowledge available

#### `memory_metrics.top_similarity` (float)
- **What it is**: Highest similarity score from semantic search
- **How it works**: Maximum similarity across all matches
- **What it signifies**: Best match quality found
- **Good value**: Higher = better match quality

### Context Injection Metrics

#### `memory_metrics.context_injected` (boolean)
- **What it is**: Whether context was injected into the prompt
- **How it works**: `true` when medium-similarity matches are used as context
- **What it signifies**: Context-enhanced query (partial cache benefit)
- **Good value**: `true` indicates context reuse

#### `memory_metrics.context_tokens_added` (integer)
- **What it is**: Number of tokens added to input from cached context
- **How it works**: Count of tokens in compressed context
- **What it signifies**: Input token cost of context injection
- **Good value**: Lower is better (but must be balanced with output savings)

#### `memory_metrics.context_original_tokens` (integer)
- **What it is**: Original token count before compression
- **How it works**: Estimated from compressed size and compression ratio
- **What it signifies**: How much context was compressed
- **Good value**: Used to calculate compression savings

#### `memory_metrics.context_compressed_tokens` (integer)
- **What it is**: Token count after compression
- **How it works**: Actual token count of compressed context
- **What it signifies**: Final context size after LLM Lingua compression
- **Good value**: Lower is better (more compression)

#### `memory_metrics.context_similarity` (float)
- **What it is**: Similarity score of the context match
- **How it works**: Same as `similarity` but specifically for context injection
- **What it signifies**: Quality of context match
- **Good value**: Higher = more relevant context

### User Preference Metrics

#### `memory_metrics.preferences_used` (boolean)
- **What it is**: Whether user preferences were applied
- **How it works**: `true` when preference confidence >0.5
- **What it signifies**: Personalized response based on learned preferences
- **Good value**: `true` indicates personalization working

#### `memory_metrics.preference_tone` (string)
- **What it is**: Detected/learned tone preference
- **Possible values**: `"formal"`, `"casual"`, `"technical"`, `"simple"`, `"neutral"`
- **How it works**: Learned from query patterns
- **What it signifies**: User's communication style preference
- **Good value**: Matches user's actual style

#### `memory_metrics.preference_format` (string)
- **What it is**: Detected/learned format preference
- **Possible values**: `"list"`, `"paragraph"`, `"code"`, `"concise"`
- **How it works**: Learned from query patterns
- **What it signifies**: User's response format preference
- **Good value**: Matches user's actual preference

#### `memory_metrics.preference_confidence` (float, 0.0-1.0)
- **What it is**: Confidence in learned preferences
- **How it works**: Increases with consistent patterns
- **What it signifies**: How certain the system is about preferences
- **Good value**: Higher = more reliable preferences

#### `memory_metrics.preference_interaction_count` (integer)
- **What it is**: Number of interactions used to learn preferences
- **How it works**: Increments with each query
- **What it signifies**: Amount of learning data
- **Good value**: Higher = more learning data

### Memory Operations Tracking

#### `memory_metrics.memory_operations.exact_lookup` (boolean)
- **What it is**: Whether exact cache lookup was performed
- **How it works**: Always `true` if exact cache enabled
- **What it signifies**: Exact cache was checked

#### `memory_metrics.memory_operations.semantic_search` (boolean)
- **What it is**: Whether semantic search was performed
- **How it works**: `true` if semantic cache enabled and exact cache missed
- **What it signifies**: Vector search was executed

#### `memory_metrics.memory_operations.context_retrieval` (boolean)
- **What it is**: Whether context was retrieved
- **How it works**: `true` when medium-similarity matches found
- **What it signifies**: Context injection path was used

#### `memory_metrics.memory_operations.preference_retrieval` (boolean)
- **What it is**: Whether preferences were retrieved
- **How it works**: `true` when preferences exist and confidence >0.5
- **What it signifies**: Personalization was applied

#### `memory_metrics.memory_operations.context_compression` (boolean)
- **What it is**: Whether context was compressed
- **How it works**: `true` when context compression is applied
- **What it signifies**: LLM Lingua compression was used

---

## Orchestrator Metrics

The Orchestrator manages token budgets, analyzes query complexity, and allocates tokens optimally.

### Query Complexity Metrics

#### `query_complexity` / `query_type` (string)
- **What it is**: Detected complexity level of the query
- **Possible values**: `"simple"`, `"medium"`, `"complex"`
- **How it works**:
  - **Simple**: <20 tokens, <100 characters
  - **Medium**: 20-100 tokens, 100-500 characters
  - **Complex**: >100 tokens, >500 characters
- **What it signifies**:
  - Determines which strategy/model to use
  - Influences token budget allocation
- **Good value**: Should match actual query complexity

### Token Budget Metrics

#### `orchestrator_metrics.token_budget` (integer)
- **What it is**: Total token budget allocated for the query
- **How it works**: 
  - Uses provided `token_budget` parameter or default (4000)
  - Distributed across components
- **What it signifies**: Total tokens available for the query
- **Good value**: Appropriate for query complexity

#### `orchestrator_metrics.max_response_tokens` (integer)
- **What it is**: Maximum tokens allocated for response generation
- **How it works**: 
  - Set by selected strategy (cheap/balanced/premium)
  - Or calculated from token budget
- **What it signifies**: Upper limit on output tokens
- **Good value**: Balanced between quality and cost

### Token Allocation Metrics

#### `orchestrator_metrics.allocations` (list)
- **What it is**: Token allocations across components
- **Structure**: List of `{component: str, tokens: int}`
- **Components**:
  - `"system_prompt"`: System instructions
  - `"user_query"`: User's query
  - `"retrieved_context"`: Cached context (if any)
  - `"response"`: Response generation budget
- **How it works**: Knapsack optimization distributes budget
- **What it signifies**: How tokens are allocated across prompt components
- **Good value**: Balanced allocation based on utility

### Token Usage Metrics

#### `tokens_used` (integer)
- **What it is**: Total tokens consumed (input + output)
- **How it works**: Sum of `input_tokens` + `output_tokens`
- **What it signifies**: Total cost of the query
- **Good value**: Lower is better (but must maintain quality)

#### `input_tokens` (integer)
- **What it is**: Tokens in the prompt sent to LLM
- **How it works**: Count of tokens in final prompt (after compression)
- **What it signifies**: Input cost
- **Good value**: Lower is better (compression helps)

#### `output_tokens` (integer)
- **What it is**: Tokens in the LLM response
- **How it works**: Count of tokens in generated response
- **What it signifies**: Output cost
- **Good value**: Appropriate for query (not too short, not too long)

### Token Efficiency Metrics

#### `orchestrator_metrics.token_efficiency` (float)
- **What it is**: Ratio of output tokens to input tokens
- **Formula**: `output_tokens / input_tokens`
- **How it works**: Measures how efficiently tokens are used
- **What it signifies**:
  - `>1.0`: More output than input (good for generation)
  - `~1.0`: Balanced
  - `<1.0`: Less output than input (may indicate over-prompting)
- **Good value**: 0.5-2.0 (depends on use case)

### Model Selection Metrics

#### `model` (string)
- **What it is**: LLM model used for generation
- **How it works**: 
  - Selected by orchestrator based on complexity
  - Or overridden by bandit strategy
- **What it signifies**: Which model handled the query
- **Good value**: Appropriate for query complexity

---

## Bandit Optimizer Metrics

The Bandit Optimizer selects the best strategy (model, tokens, temperature) for each query using online learning.

### Strategy Selection Metrics

#### `strategy` / `strategy_arm` (string)
- **What it is**: Selected strategy arm ID
- **Possible values**: `"cheap"`, `"balanced"`, `"premium"`
- **How it works**: 
  - **Cheap**: Fast, low-cost model (gpt-4o-mini/gemini-flash), 300 tokens, low temp
  - **Balanced**: Standard model, 600 tokens, medium temp
  - **Premium**: Powerful model (gpt-4o/gemini-pro), 1000 tokens, higher temp
- **What it signifies**: Which strategy was selected for this query
- **Good value**: Should match query complexity

### Bandit Algorithm Metrics

#### `bandit_metrics.algorithm` (string)
- **What it is**: Bandit algorithm used
- **Possible values**: `"ucb"`, `"epsilon_greedy"`, `"thompson"`
- **How it works**:
  - **UCB**: Upper Confidence Bound - balances exploration/exploitation
  - **Epsilon-Greedy**: Random exploration with probability ε
  - **Thompson Sampling**: Bayesian approach
- **What it signifies**: Learning strategy
- **Good value**: UCB is default and works well

### Reward Metrics

#### `reward` (float)
- **What it is**: Reward signal for the selected strategy
- **How it works**: 
  - Standard: `quality_score - lambda * tokens_used`
  - RouterBench: `quality * (1 - cost_weight * cost) * (1 - latency_weight * latency)`
- **What it signifies**: How well the strategy performed
- **Good value**: Higher = better performance

### RouterBench Metrics

#### `bandit_metrics.routerbench.total_cost` (float)
- **What it is**: Total cost in USD across all queries for this strategy
- **How it works**: Sum of `(tokens / 1M) * cost_per_million` for all queries
- **What it signifies**: Cumulative cost for this strategy
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.total_tokens` (integer)
- **What it is**: Total tokens used across all queries
- **How it works**: Sum of tokens for all queries using this strategy
- **What it signifies**: Token consumption
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.total_latency_ms` (float)
- **What it is**: Total latency in milliseconds
- **How it works**: Sum of latencies for all queries
- **What it signifies**: Cumulative latency
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.total_quality` (float)
- **What it is**: Total quality score
- **How it works**: Sum of quality scores (0-1) for all queries
- **What it signifies**: Cumulative quality
- **Good value**: Higher is better

#### `bandit_metrics.routerbench.query_count` (integer)
- **What it is**: Number of queries using this strategy
- **How it works**: Count of queries
- **What it signifies**: Sample size for this strategy
- **Good value**: Higher = more data

#### `bandit_metrics.routerbench.avg_cost_per_query` (float)
- **What it is**: Average cost per query
- **Formula**: `total_cost / query_count`
- **What it signifies**: Cost efficiency
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.avg_tokens` (float)
- **What it is**: Average tokens per query
- **Formula**: `total_tokens / query_count`
- **What it signifies**: Token efficiency
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.avg_latency` (float)
- **What it is**: Average latency per query
- **Formula**: `total_latency_ms / query_count`
- **What it signifies**: Speed
- **Good value**: Lower is better

#### `bandit_metrics.routerbench.avg_quality` (float)
- **What it is**: Average quality per query
- **Formula**: `total_quality / query_count`
- **What it signifies**: Quality level
- **Good value**: Higher is better (closer to 1.0)

#### `bandit_metrics.routerbench.cost_quality_ratio` (float)
- **What it is**: Quality per unit cost
- **Formula**: `avg_quality / (avg_cost_per_query * 1000)`
- **What it signifies**: Cost-effectiveness
- **Good value**: Higher = better value

#### `bandit_metrics.routerbench.efficiency_score` (float)
- **What it is**: Combined efficiency score
- **Formula**: `(quality * 0.5) + (cost_efficiency * 0.3) + (speed * 0.2)`
- **What it signifies**: Overall strategy efficiency
- **Good value**: Higher = better overall performance

---

## Compression Metrics

Compression reduces token usage while preserving semantic meaning.

### Context Compression Metrics

#### `compression_metrics.context_compressed` (boolean)
- **What it is**: Whether context was compressed
- **How it works**: `true` when LLM Lingua compresses retrieved context
- **What it signifies**: Context compression was applied
- **Good value**: `true` for long contexts

#### `compression_metrics.context_original_tokens` (integer)
- **What it is**: Original token count before compression
- **How it works**: Count of tokens in original context
- **What it signifies**: How much context was compressed
- **Good value**: Used to calculate savings

#### `compression_metrics.context_compressed_tokens` (integer)
- **What it is**: Token count after compression
- **How it works**: Count of tokens in compressed context
- **What it signifies**: Final context size
- **Good value**: Lower is better

#### `compression_metrics.context_compression_ratio` (float)
- **What it is**: Compression ratio achieved
- **Formula**: `compressed_tokens / original_tokens`
- **How it works**: Ratio of compressed to original
- **What it signifies**: 
  - `0.4` = 40% of original (60% reduction)
  - Lower = more compression
- **Good value**: 0.3-0.5 (aggressive but preserves meaning)

### Query Compression Metrics

#### `compression_metrics.query_compressed` (boolean)
- **What it is**: Whether query was compressed
- **How it works**: `true` when query exceeds thresholds (>200 tokens or >800 chars)
- **What it signifies**: Query compression was applied
- **Good value**: `true` for long queries

#### `compression_metrics.query_original_tokens` (integer)
- **What it is**: Original query token count
- **How it works**: Count before compression
- **What it signifies**: Original query size
- **Good value**: Used to calculate savings

#### `compression_metrics.query_compressed_tokens` (integer)
- **What it is**: Compressed query token count
- **How it works**: Count after compression
- **What it signifies**: Final query size
- **Good value**: Lower is better

#### `compression_metrics.query_compression_ratio` (float)
- **What it is**: Query compression ratio
- **Formula**: `compressed_tokens / original_tokens`
- **What it signifies**: How much query was compressed
- **Good value**: 0.4-0.6 (preserves query intent)

### Total Compression Metrics

#### `compression_metrics.total_compression_savings` (integer)
- **What it is**: Total tokens saved from compression
- **Formula**: `(context_original - context_compressed) + (query_original - query_compressed)`
- **How it works**: Sum of all compression savings
- **What it signifies**: Total compression benefit
- **Good value**: Higher = more savings

---

## Savings Metrics

Savings metrics track how much the platform saves compared to baseline.

### Component Savings

#### `component_savings.memory_layer` (integer)
- **What it is**: Tokens saved by memory layer
- **How it works**:
  - **Exact/Semantic Direct Hit**: Full baseline tokens (100% savings)
  - **Context Hit**: Net savings (output reduction - context input cost)
  - **Cache Miss**: 0
- **What it signifies**: Memory layer contribution to savings
- **Good value**: Higher = more cache hits

#### `component_savings.orchestrator` (integer)
- **What it is**: Tokens saved by orchestrator
- **How it works**:
  - Output token reduction vs baseline
  - Compression savings (context + query)
  - Better token allocation
- **What it signifies**: Orchestrator contribution to savings
- **Good value**: Higher = better optimization

#### `component_savings.bandit` (integer)
- **What it is**: Tokens saved by bandit optimizer
- **How it works**:
  - Strategy selection savings (choosing cheaper models)
  - Max token limits (preventing over-generation)
  - Cost-aware routing
- **What it signifies**: Bandit contribution to savings
- **Good value**: Higher = better strategy selection

#### `component_savings.total_savings` (integer)
- **What it is**: Total tokens saved across all components
- **Formula**: `memory_layer + orchestrator + bandit`
- **What it signifies**: Overall platform savings
- **Good value**: Higher = more total savings

### Savings Percentage

#### `savings_percentage` (float, 0-100)
- **What it is**: Percentage of tokens saved
- **Formula**: `total_savings / (total_tokens + total_savings) * 100`
- **How it works**: Ratio of savings to total potential cost
- **What it signifies**: Overall efficiency
- **Good value**: 20-50% typical, higher is better

---

## Quality Metrics

### Quality Judge Metrics (if enabled)

#### `quality_judge.winner` (string)
- **What it is**: Which response was better
- **Possible values**: `"optimized"`, `"baseline"`, `"equivalent"`
- **How it works**: LLM judge compares optimized vs baseline
- **What it signifies**: 
  - `"optimized"`: Tokenomics platform produced better response
  - `"baseline"`: Baseline was better (rare)
  - `"equivalent"`: Similar quality
- **Good value**: `"optimized"` or `"equivalent"` indicates quality maintained

#### `quality_judge.explanation` (string)
- **What it is**: Explanation of judgment
- **How it works**: LLM explains why it chose the winner
- **What it signifies**: Reasoning behind quality assessment
- **Good value**: Clear explanation

#### `quality_judge.confidence` (float, 0.0-1.0)
- **What it is**: Confidence in judgment
- **How it works**: LLM's confidence score
- **What it signifies**: How certain the judge is
- **Good value**: Higher = more reliable judgment

### Baseline Comparison Metrics

#### `baseline_comparison.tokens_used` (integer)
- **What it is**: Tokens used by baseline (vanilla LLM)
- **How it works**: Count from baseline query
- **What it signifies**: What we're comparing against
- **Good value**: Used to calculate savings

#### `baseline_comparison.latency_ms` (float)
- **What it is**: Baseline latency
- **How it works**: Time for baseline query
- **What it signifies**: Baseline performance
- **Good value**: Used to compare latency

---

## Performance Metrics

### Latency Metrics

#### `elapsed_ms` (float)
- **What it is**: Total query processing time in milliseconds
- **How it works**: End-to-end time from query to response
- **What it signifies**: User-perceived latency
- **Good value**: Lower is better (<2000ms for cache hits, <5000ms for LLM calls)

#### `latency_ms` (float)
- **What it is**: LLM API latency
- **How it works**: Time for LLM API call only
- **What it signifies**: LLM provider performance
- **Good value**: Lower is better

### Summary Statistics

#### `summary.test_execution.total_tests` (integer)
- **What it is**: Total number of test cases
- **What it signifies**: Test coverage
- **Good value**: Higher = more comprehensive

#### `summary.test_execution.successful_tests` (integer)
- **What it is**: Number of successful tests
- **What it signifies**: Test reliability
- **Good value**: Should equal `total_tests`

#### `summary.cache_performance.cache_hit_rate` (float, 0-1)
- **What it is**: Percentage of queries that hit cache
- **Formula**: `cache_hits / total_queries`
- **What it signifies**: Cache effectiveness
- **Good value**: 20-40% typical, higher is better

#### `summary.token_usage.avg_tokens_per_query` (float)
- **What it is**: Average tokens per query
- **Formula**: `total_tokens / successful_tests`
- **What it signifies**: Average cost per query
- **Good value**: Lower is better

---

## Test Cases

### Exact Cache Tests

**Purpose**: Validate identical query caching

**Test Flow**:
1. First query: "What is Python?" → Cache miss, stores response
2. Second query: "What is Python?" → Exact cache hit, 0 tokens

**Expected Results**:
- First: `cache_hit=false`, `tokens_used > 0`
- Second: `cache_hit=true`, `cache_type="exact"`, `tokens_used=0`

### Semantic Cache Direct Tests

**Purpose**: Validate high-similarity direct returns

**Test Flow**:
1. Query: "Explain Python programming language" → Should match "What is Python?" with similarity >0.85
2. Returns cached response directly (no LLM call)

**Expected Results**:
- `cache_hit=true`, `cache_type="semantic_direct"`, `similarity > 0.85`, `tokens_used=0`

### Context Injection Tests

**Purpose**: Validate medium-similarity context enhancement

**Test Flow**:
1. Query: "How do I install Python packages?" → Should match "What is Python?" with similarity 0.75-0.85
2. Injects compressed context into prompt
3. Generates response with context

**Expected Results**:
- `cache_hit=true`, `cache_type="context"`, `similarity 0.75-0.85`
- `context_injected=true`, `context_tokens_added > 0`
- `tokens_used > 0` (but less than without context)

### LLM Lingua Query Compression Tests

**Purpose**: Validate query compression for long queries

**Test Flow**:
1. Query: Very long query (>200 tokens)
2. LLM Lingua compresses query
3. Uses compressed query in prompt

**Expected Results**:
- `query_compressed=true`
- `query_compressed_tokens < query_original_tokens`
- `total_compression_savings > 0`

### LLM Lingua Context Compression Tests

**Purpose**: Validate context compression

**Test Flow**:
1. Query retrieves long context
2. LLM Lingua compresses context
3. Uses compressed context in prompt

**Expected Results**:
- `context_compressed=true`
- `context_compressed_tokens < context_original_tokens`
- `compression_ratio ~0.4`

### User Preference Tests

**Purpose**: Validate preference learning

**Test Flow**:
1. Query with formal tone: "Could you please provide..."
2. System learns formal preference
3. Subsequent queries use formal preference

**Expected Results**:
- `preferences_used=true` (after learning)
- `preference_tone="formal"` (matches query)
- `preference_confidence` increases with more queries

### Query Complexity Tests

**Purpose**: Validate complexity detection

**Test Cases**:
- Simple: "What is API?" → `complexity="simple"`
- Medium: "How does authentication work?" → `complexity="medium"`
- Complex: "Design a comprehensive system..." → `complexity="complex"`

**Expected Results**:
- Complexity matches actual query complexity
- Appropriate strategy selected

### Bandit Selection Tests

**Purpose**: Validate strategy selection

**Test Flow**:
1. Simple query → Should select "cheap" strategy
2. Medium query → Should select "balanced" strategy
3. Complex query → Should select "premium" strategy

**Expected Results**:
- Strategy matches query complexity
- RouterBench metrics track performance

### RouterBench Tests

**Purpose**: Validate cost-aware routing

**Test Flow**:
1. Query with `use_cost_aware_routing=true`
2. Bandit selects strategy based on cost-quality tradeoff
3. RouterBench metrics updated

**Expected Results**:
- `bandit_metrics.routerbench` populated
- `efficiency_score` calculated
- Cost-quality optimization working

### Token Budget Tests

**Purpose**: Validate different budget scenarios

**Test Cases**:
- Low budget (1000 tokens)
- Medium budget (2000 tokens)
- High budget (4000 tokens)

**Expected Results**:
- Budget respected
- Allocations appropriate for budget
- Response quality maintained

---

## Interpreting Results

### Overall Health Check

**Good Platform Health**:
- ✅ Cache hit rate >20%
- ✅ Total savings >15%
- ✅ Quality maintained (judge winner = "optimized" or "equivalent")
- ✅ All components working (no errors)
- ✅ Compression working (compression savings >0)
- ✅ Bandit learning (strategy distribution appropriate)

### Component Health

**Memory Layer**:
- ✅ Exact cache hits >0
- ✅ Semantic cache hits >0
- ✅ Context injection working
- ✅ LLM Lingua compression working
- ✅ Preferences learning

**Orchestrator**:
- ✅ Complexity detection accurate
- ✅ Token allocation balanced
- ✅ Budget respected
- ✅ Model selection appropriate

**Bandit Optimizer**:
- ✅ Strategy selection working
- ✅ RouterBench metrics populated
- ✅ Learning over time (rewards improving)
- ✅ Cost-aware routing working

### Red Flags

**Issues to Watch For**:
- ❌ Cache hit rate <10% (cache not working well)
- ❌ Total savings <5% (optimization not effective)
- ❌ Quality judge winner = "baseline" frequently (quality degraded)
- ❌ Compression not working (compression_savings = 0)
- ❌ All queries using same strategy (bandit not learning)
- ❌ High error rate (component failures)

### Performance Benchmarks

**Typical Values**:
- Cache hit rate: 20-40%
- Total savings: 15-35%
- Avg tokens per query: 200-500 (with cache)
- Avg latency: 1000-3000ms (with cache), 3000-6000ms (without cache)
- Compression ratio: 0.3-0.5
- Quality maintained: >90% of queries

---

## Conclusion

This comprehensive diagnostic test validates every component and capability of the Tokenomics platform. Use it to:

1. **Validate Platform**: Ensure all components are working
2. **Showcase Capabilities**: Demonstrate full range of features
3. **Measure Performance**: Track savings and efficiency
4. **Identify Issues**: Find components that need attention
5. **Benchmark**: Compare against baseline and track improvements

For questions or issues, refer to the component-specific documentation or test results.

