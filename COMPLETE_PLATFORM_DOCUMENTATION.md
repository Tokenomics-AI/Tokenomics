# Tokenomics Platform - Complete Documentation

## 📖 How to Read This Documentation

**New to the platform?** Start here:
1. Read [Overview](#overview) - Understand what the platform does
2. Read [Complete Query Flow](#complete-query-flow) - See how a query is processed step-by-step
3. Look at [Flow Diagrams](#flow-diagrams) - Visual understanding
4. Read [Component Details](#component-details) - Deep dive into each component
5. Check [Examples](#examples) - See it in action

**Want to understand a specific component?**
- Jump to [Component Details](#component-details) section
- Each component has: Purpose, Key Methods, How It Works

**Debugging an issue?**
- Check [Troubleshooting](#troubleshooting) section
- Review [Flow Diagrams](#flow-diagrams) to see decision points

**Want to modify the platform?**
- Read [Data Structures](#data-structures) - Understand data formats
- Read [Configuration](#configuration) - See what can be configured
- Check [Key Files Reference](#key-files-reference) - Find the right file

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Complete Query Flow](#complete-query-flow)
4. [Flow Diagrams](#flow-diagrams)
5. [Component Details](#component-details)
6. [Data Structures](#data-structures)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Key Files Reference](#key-files-reference)

---

## Overview

### What is Tokenomics Platform?

Tokenomics Platform is an intelligent proxy layer between users and Large Language Models (LLMs) that **optimizes token usage while maintaining quality**. It acts as a smart intermediary that:

- **Remembers** previous queries and answers (caching)
- **Plans** optimal token allocation (orchestration)
- **Predicts** answer length (ML-based prediction)
- **Routes** queries to the best model (bandit optimization)
- **Compresses** long queries (LLMLingua-2)
- **Ensures** quality (cascading inference + quality judge)
- **Learns** from experience (data collection for ML)

### Core Problem It Solves

**Without Tokenomics:**
- Every query costs tokens (money)
- No memory of previous queries
- Fixed token budgets (waste or truncation)
- No learning from past queries
- One-size-fits-all model selection

**With Tokenomics:**
- Cache hits = 0 tokens (instant answers)
- Smart token allocation (no waste)
- Dynamic prediction (right-sized responses)
- Adaptive routing (best model for each query)
- Continuous learning (improves over time)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tokenomics Platform                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Memory     │  │ Orchestrator │  │    Bandit    │       │
│  │    Layer     │  │              │  │  Optimizer   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │                │
│         └─────────────────┼──────────────────┘                │
│                           │                                   │
│                  ┌────────▼────────┐                          │
│                  │  Core Platform  │                          │
│                  │   (Coordinator) │                          │
│                  └────────┬────────┘                          │
│                           │                                   │
│         ┌─────────────────┼─────────────────┐                 │
│         │                 │                 │                 │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐          │
│  │   Token     │  │ Compression  │  │   Quality   │          │
│  │ Predictor   │  │  (LLMLingua) │  │    Judge    │          │
│  └─────────────┘  └──────────────┘  └─────────────┘          │
│                                                               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  LLM Providers │
                    │ (OpenAI/Gemini)│
                    └────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key File |
|-----------|---------------|----------|
| **Core Platform** | Coordinates all components, main entry point | `tokenomics/core.py` |
| **Memory Layer** | Caching (exact + semantic), context injection | `tokenomics/memory/memory_layer.py` |
| **Orchestrator** | Token budget planning, complexity analysis | `tokenomics/orchestrator/orchestrator.py` |
| **Bandit Optimizer** | Strategy selection (cheap/balanced/premium) | `tokenomics/bandit/bandit.py` |
| **Token Predictor** | Predicts answer length (heuristic → ML) | `tokenomics/ml/token_predictor.py` |
| **Compression** | Compresses long queries | `tokenomics/compression/llmlingua_compressor.py` |
| **Quality Judge** | Compares answers, ensures quality | `tokenomics/judge/quality_judge.py` |
| **LLM Providers** | Interface to OpenAI/Gemini APIs | `tokenomics/llm_providers/` |

---

## Complete Query Flow

### Main Entry Point

**File**: `tokenomics/core.py`  
**Method**: `query()` (line 251)

```python
def query(
    self,
    query: str,
    token_budget: Optional[int] = None,
    use_cache: bool = True,
    use_bandit: bool = True,
    system_prompt: Optional[str] = None,
    use_compression: bool = True,
    use_cost_aware_routing: bool = True,
) -> Dict:
```

### Step-by-Step Flow

#### **STEP 1: Input Validation** (lines 282-291)

```python
# Validate query
if not query or not query.strip():
    raise ValueError("Query cannot be empty")
```

**What happens**: Checks if query is valid  
**If invalid**: Raises error immediately  
**If valid**: Continues to Step 2

---

#### **STEP 2: Memory Layer - Cache Check** (lines 334-460)

```python
if use_cache:
    cache_entry, compressed_context, preference_context, match_similarity, mem_ops = \
        self.memory.retrieve_compressed(query)
```

**What happens**:
1. **Exact Cache Check**: Hash query → check if exact match exists
2. **Semantic Cache Check**: Generate embedding → search similar queries
3. **Context Retrieval**: If similarity 0.70-0.85, retrieve context
4. **Preference Retrieval**: Get user preferences (tone, format)

**Cache Tiers**:
- **Exact Match** (similarity = 1.0): Instant return, 0 tokens
- **Semantic Direct** (similarity > 0.85): Return similar answer, 0 tokens
- **Context Injection** (similarity 0.70-0.85): Add context to prompt
- **Cache Miss** (similarity < 0.70): Continue to full processing

**If Cache Hit**:
```python
return {
    "response": cache_entry.response,
    "tokens_used": 0,  # FREE!
    "cache_hit": True,
    "cache_type": "exact" or "semantic_direct",
    ...
}
```

**If Cache Miss**: Continue to Step 3

---

#### **STEP 3: Complexity Analysis** (line 509)

```python
complexity = self.orchestrator.analyze_complexity(query).value
# Returns: "simple", "medium", or "complex"
```

**What happens**:
- Analyzes query length, keywords, structure
- Classifies as: simple / medium / complex
- Used for routing and prediction

**Complexity Rules**:
- **Simple**: Short queries, simple questions ("What is X?")
- **Medium**: Moderate length, requires explanation
- **Complex**: Long queries, technical topics, multi-part questions

---

#### **STEP 4: Query Planning** (lines 518-525)

```python
plan = self.orchestrator.plan_query(
    query=query,
    token_budget=token_budget or 4000,
    retrieved_context=[compressed_context] if compressed_context else None,
)
```

**What happens**:
1. **Token Budget**: Default 4000 tokens (or specified)
2. **Knapsack Optimization**: Allocates tokens to:
   - User query: X tokens
   - System prompt: 100 tokens
   - Context (if any): Y tokens
   - Response: Remaining tokens
3. **Context Quality Score**: 1.0 = full context, 0.0 = no context

**Returns**: `QueryPlan` object with allocations

---

#### **STEP 5: Bandit Routing** (lines 528-548)

```python
if use_bandit:
    strategy = self.bandit.select_strategy_cost_aware(
        query_complexity=complexity,
        context_quality_score=plan.context_quality_score,
    )
```

**What happens**:
- **Multi-Armed Bandit Algorithm** (UCB) selects best strategy:
  - `cheap`: gpt-4o-mini (fast, cheap, simple queries)
  - `balanced`: gpt-4o-mini with more tokens (medium queries)
  - `premium`: gpt-4o (expensive, best quality, complex queries)
- **RouterBench Cost-Quality Routing**: Balances cost vs quality
- **Learning**: Updates strategy performance after each query

**Strategy Selection Logic**:
```
Simple query + high context quality → cheap
Medium query → balanced
Complex query + low context quality → premium
```

---

#### **STEP 6: Cascading Inference Setup** (lines 550-600)

```python
if self.cascading_enabled and strategy.model == premium_model:
    # NEW: Predict escalation likelihood using bandit
    escalation_likelihood = bandit.predict_escalation_likelihood(
        query_complexity, context_quality_score, query_tokens
    )
    
    if escalation_likelihood >= threshold (0.7):
        # Skip cheap model, go straight to premium
        initial_model = premium_model
        should_cascade = False
    else:
        # Normal cascading: start with cheap
        initial_model = cheap_model
        should_cascade = True
```

**What happens**:
- **NEW**: Bandit predicts escalation likelihood based on historical data
- If prediction suggests escalation is likely (≥70% probability), skip cheap model
- Otherwise, use normal cascading: try cheap first, escalate if needed

**Escalation Prediction**:
- Uses historical escalation patterns from similar queries
- Considers: query complexity, context quality, query length
- Learns over time as more data is collected
- Configurable threshold (default: 0.7 = 70% likelihood)

**Cascading Logic** (when prediction says to try cheap):
```
1. Generate with cheap model
2. Quick quality check (heuristic, fast)
3. If quality < 0.85 threshold:
   → Escalate to premium model
   → Re-generate with premium
4. Return best response
5. Record outcome for learning
```

---

#### **STEP 7: Prompt Building** (lines 583-587)

```python
prompt = self.orchestrator.build_prompt(plan, system_prompt=system_prompt)
```

**What happens**:
- Combines: system prompt + context (if any) + user query
- Respects token allocations from plan
- Applies user preferences (tone, format)

---

#### **STEP 8: Query Compression** (lines 589-630)

```python
if use_compression:
    if query_tokens > 150 or query_chars > 500:
        compressed_query = self.memory.compress_query_if_needed(query)
```

**What happens**:
- **LLMLingua-2** compresses long queries
- Only compresses if query > 150 tokens OR > 500 characters
- Reduces token count while preserving meaning

**Compression Example**:
```
Original: "Write a detailed explanation of quantum computing, including qubits, superposition, entanglement..."
(200 tokens)

Compressed: "Explain quantum computing: qubits, superposition, entanglement..."
(80 tokens)

Savings: 120 tokens
```

---

#### **STEP 9: Token Prediction** (lines 632-660)

```python
if self.token_predictor:
    predicted_max_tokens = self.token_predictor.predict(
        query=query,
        complexity=complexity,
        query_tokens=query_token_count,
        query_embedding=query_embedding,
    )
    # Override strategy's max_tokens with prediction
    max_response_tokens = predicted_max_tokens
```

**What happens**:
- **Heuristic Mode** (current): Uses rules based on complexity + length
- **ML Mode** (future): Uses XGBoost model trained on historical data
- Predicts how many tokens the answer will need
- Overrides hardcoded max_tokens from strategy

**Prediction Logic** (Heuristic):
```python
base_tokens = {
    "simple": 200,
    "medium": 500,
    "complex": 800
}
length_factor = 1.0 + (query_tokens / 100) * 0.1
predicted = base_tokens[complexity] * length_factor * type_multiplier
```

**Data Collection**:
- Records: predicted_tokens, actual_tokens, query features
- Stores in `token_prediction_data.db`
- When 500+ samples → trains ML model

---

#### **STEP 10: LLM Generation** (lines 700-750)

```python
llm_response = self.llm_provider.generate(
    prompt=prompt,
    model=initial_model,  # From cascading (cheap or premium)
    max_tokens=predicted_max_tokens,
    temperature=strategy.temperature,
    ...
)
```

**What happens**:
- Calls OpenAI/Gemini API
- Uses predicted max_tokens (not hardcoded)
- Returns: response text, tokens used, latency

---

#### **STEP 11: Cascading Quality Check** (lines 750-800)

```python
if should_cascade and llm_response:
    quality_score = self.quality_judge.quick_quality_check(
        query, llm_response.text
    )
    if quality_score < self.cascading_quality_threshold:  # 0.85
        # Escalate to premium model
        llm_response = premium_model.generate(prompt, ...)
```

**What happens**:
- **Quick Quality Check**: Heuristic-based (fast, no LLM call)
- Checks: response length, completeness, keyword overlap
- If score < 0.85: Escalate to premium model
- If score >= 0.85: Use cheap model answer (saves money!)

**Quality Check Heuristic**:
```python
score = 0.0
# Length check (30%)
if response_length > min_length:
    score += 0.3
# Completeness indicators (40%)
if has_complete_indicators(response):
    score += 0.4
# Keyword overlap (30%)
overlap = calculate_keyword_overlap(query, response)
score += 0.3 * overlap
```

---

#### **STEP 12: Store Results** (lines 1100-1150)

```python
# Store in cache
if not cache_hit:
    self.memory.store(query, response, ...)

# Record training data
if self.token_predictor:
    self.token_predictor.record_prediction(
        query=query,
        predicted_tokens=predicted_max_tokens,
        actual_output_tokens=output_tokens,
        ...
    )

# Update bandit rewards
if strategy:
    reward = calculate_reward(quality, cost, latency)
    self.bandit.update_reward(strategy.arm_id, reward)
```

**What happens**:
1. **Cache Storage**: Stores in exact cache (hash) + semantic cache (embedding)
2. **Training Data**: Records prediction vs actual for ML learning
3. **Bandit Update**: Updates strategy performance statistics

---

#### **STEP 13: Return Response** (line 1148)

```python
return {
    "response": llm_response.text,
    "tokens_used": input_tokens + output_tokens,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "cache_hit": False,
    "cache_type": "none",
    "strategy": strategy.arm_id,
    "model": model_used,
    "latency_ms": latency,
    "component_savings": {
        "memory_layer": 0,  # Cache miss
        "orchestrator": orchestrator_savings,
        "bandit": bandit_savings,
        "total_savings": total,
    },
    "predicted_max_tokens": predicted_max_tokens,
    "cascading_escalated": escalated,
    ...
}
```

---

## Flow Diagrams

### Complete Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                 │
│              "Explain quantum computing"                      │
└───────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 1: Input Validation    │
        │   - Check query not empty     │
        │   - Strip whitespace           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 2: Memory Layer        │
        │   ┌─────────────────────┐     │
        │   │ Exact Cache Check   │     │
        │   └──────────┬──────────┘     │
        │              │                 │
        │              ▼                 │
        │   ┌─────────────────────┐     │
        │   │ Semantic Cache      │     │
        │   │ (Embedding Search)   │     │
        │   └──────────┬──────────┘     │
        │              │                 │
        │    ┌─────────┴─────────┐       │
        │    │                   │       │
        │    ▼                   ▼       │
        │  HIT                MISS       │
        │  │                   │         │
        │  │                   │         │
        │  ▼                   ▼         │
        │ Return            Continue     │
        │ (0 tokens)        to Step 3    │
        └─────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 3: Complexity Analysis │
        │   - Query length              │
        │   - Keywords                  │
        │   - Structure                 │
        │   Result: "complex"           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 4: Query Planning      │
        │   - Token budget: 4000        │
        │   - Knapsack optimization     │
        │   - Allocate:                 │
        │     * Query: 50 tokens        │
        │     * System: 100 tokens      │
        │     * Response: 3850 tokens   │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 5: Bandit Routing      │
        │   - Complexity: complex        │
        │   - Select: "premium"          │
        │   - Model: gpt-4o             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 6: Cascading Setup     │
        │   - Premium selected          │
        │   - Override: start cheap     │
        │   - Initial: gpt-4o-mini      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 7: Build Prompt        │
        │   - System prompt             │
        │   - User query                │
        │   - Context (if any)          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 8: Query Compression   │
        │   - Query: 50 tokens          │
        │   - Threshold: 150 tokens     │
        │   - Skip compression          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 9: Token Prediction    │
        │   - Complexity: complex       │
        │   - Query tokens: 50          │
        │   - Predicted: 1300 tokens    │
        │   - Override max_tokens       │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 10: LLM Generation     │
        │   - Model: gpt-4o-mini        │
        │   - Max tokens: 1300          │
        │   - Generate response         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 11: Quality Check      │
        │   - Quick check: 0.75         │
        │   - Threshold: 0.85           │
        │   - Quality < threshold       │
        │   - ESCALATE to gpt-4o        │
        │   - Re-generate response      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 12: Store Results      │
        │   - Cache: Store in DB        │
        │   - Training: Record data     │
        │   - Bandit: Update stats      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   STEP 13: Return Response    │
        │   - Response text             │
        │   - Tokens used               │
        │   - Metrics                   │
        └───────────────────────────────┘
```

### Cache Decision Tree

```
                    Query Received
                         │
                         ▼
            ┌────────────────────────┐
            │  Exact Cache Check     │
            │  (Hash-based lookup)   │
            └───────────┬────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
        FOUND                   NOT FOUND
            │                       │
            │                       ▼
            │          ┌────────────────────────┐
            │          │  Semantic Cache Check   │
            │          │  (Vector similarity)   │
            │          └───────────┬────────────┘
            │                      │
            │          ┌───────────┴───────────┐
            │          │                       │
            │          ▼                       ▼
            │      Similarity              Similarity
            │      > 0.85                  < 0.85
            │          │                       │
            │          │                       ▼
            │          │              ┌────────────────┐
            │          │              │  Cache Miss    │
            │          │              │  (Full LLM)    │
            │          │              └────────────────┘
            │          │
            │          ▼
            │  ┌──────────────────┐
            │  │  Direct Return   │
            │  │  (0 tokens)      │
            │  └──────────────────┘
            │
            ▼
    ┌──────────────────┐
    │  Direct Return   │
    │  (0 tokens)      │
    └──────────────────┘
```

### Cascading Inference Flow

```
                    Query + Strategy
                         │
                         ▼
            ┌────────────────────────┐
            │  Strategy Selected:     │
            │  Premium (gpt-4o)      │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  Cascading Enabled?    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌───────────────────────────┐
            │ Predict Escalation        │
            │ Likelihood (Bandit)       │
            └───────────┬───────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     ┌─────────────┐         ┌─────────────┐
     │ Likelihood  │         │ Likelihood  │
     │   < 0.7     │         │   >= 0.7    │
     │ (Try Cheap) │         │ (Skip Cheap)│
     └──────┬──────┘         └──────┬──────┘
            │                       │
            ▼                       ▼
     ┌───────────────┐       ┌───────────────┐
     │ Generate with │       │ Generate with │
     │  Cheap Model  │       │ Premium Model │
     └───────┬───────┘       └───────┬───────┘
             │                       │
             ▼                       │
     ┌───────────────┐               │
     │ Quality Check │               │
     │  (Heuristic)  │               │
     └───────┬───────┘               │
             │                       │
     ┌───────┴───────┐               │
     │               │               │
     ▼               ▼               │
┌─────────┐   ┌───────────────┐     │
│ Quality │   │ Quality <     │     │
│ >= 0.85 │   │  Threshold    │     │
└────┬────┘   └───────┬───────┘     │
     │                │             │
     │                ▼             │
     │        ┌───────────────┐    │
     │        │  Escalate to  │    │
     │        │ Premium Model │    │
     │        └───────┬───────┘    │
     │                │            │
     └────────┬───────┴────────────┘
              │
              ▼
      ┌───────────────┐
      │ Return Best   │
      │   Response    │
      │ Record Outcome│
      └───────────────┘
```

### Bandit Strategy Selection

```
                    Query + Complexity
                         │
                         ▼
            ┌────────────────────────┐
            │  Cost-Aware Routing?   │
            └───────────┬────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
          YES                      NO
            │                       │
            │                       ▼
            │              ┌──────────────┐
            │              │ Random/UCB    │
            │              │ Selection      │
            │              └──────────────┘
            │
            ▼
    ┌──────────────────────┐
    │  Calculate:          │
    │  - Query Complexity  │
    │  - Context Quality   │
    │  - Strategy Rewards   │
    └───────────┬──────────┘
                │
                ▼
    ┌──────────────────────┐
    │  Select Strategy:    │
    │  - Simple → cheap    │
    │  - Medium → balanced │
    │  - Complex → premium │
    └───────────┬──────────┘
                │
                ▼
    ┌──────────────────────┐
    │  Return Strategy     │
    │  (arm_id, model,     │
    │   max_tokens, etc.)  │
    └──────────────────────┘
```

---

## Component Details

### 1. Memory Layer (`tokenomics/memory/memory_layer.py`)

#### Purpose
Smart caching system with exact matching, semantic search, and context injection.

#### Key Methods

**`retrieve_compressed(query)`**:
- Checks exact cache (hash-based)
- Searches semantic cache (vector similarity)
- Retrieves context if similarity 0.70-0.85
- Returns: cache_entry, compressed_context, preferences, similarity, operations

**`store(query, response, ...)`**:
- Stores in exact cache (hash key)
- Stores in semantic cache (embedding vector)
- Saves to persistent storage (SQLite)

#### Cache Tiers

1. **Exact Cache** (similarity = 1.0)
   - Hash-based lookup
   - Instant return
   - 0 tokens

2. **Semantic Direct** (similarity > 0.85)
   - Vector similarity search
   - Return similar answer
   - 0 tokens

3. **Context Injection** (similarity 0.70-0.85)
   - Add compressed context to prompt
   - Reduces output tokens
   - Adds input tokens (net savings if similarity high)

4. **Cache Miss** (similarity < 0.70)
   - Full LLM call required

#### Data Storage

- **Exact Cache**: `MemoryCache` (in-memory + SQLite)
- **Semantic Cache**: `FAISSVectorStore` or `ChromaVectorStore`
- **Persistent**: `tokenomics_cache.db` (SQLite)

---

### 2. Orchestrator (`tokenomics/orchestrator/orchestrator.py`)

#### Purpose
Plans token allocation and analyzes query complexity.

#### Key Methods

**`analyze_complexity(query)`**:
```python
# Analyzes query to determine complexity
- Query length
- Keywords (explain, describe, write, etc.)
- Structure (question marks, multiple parts)
Returns: QueryComplexity.SIMPLE | MEDIUM | COMPLEX
```

**`plan_query(query, token_budget, retrieved_context)`**:
```python
# Creates query plan with token allocations
1. Calculate complexity
2. Allocate token budget using knapsack optimization:
   - User query: X tokens
   - System prompt: 100 tokens
   - Context: Y tokens (if provided)
   - Response: Remaining tokens
3. Calculate context quality score
Returns: QueryPlan object
```

**`build_prompt(plan, system_prompt)`**:
```python
# Builds final prompt
- System prompt (if provided)
- Compressed context (if any)
- User query
- Applies user preferences (tone, format)
Returns: Complete prompt string
```

#### Knapsack Optimization

Maximizes utility within token budget:

```python
Components:
- User query: utility=1.0, tokens=query_length
- System prompt: utility=1.0, tokens=100
- Response: utility=0.9, tokens=remaining_budget

Algorithm:
1. Calculate utility density (utility/tokens)
2. Allocate in order of highest density
3. Ensure response gets minimum ratio (30%)
```

---

### 3. Bandit Optimizer (`tokenomics/bandit/bandit.py`)

#### Purpose
Selects optimal strategy (model + parameters) using multi-armed bandit algorithm.

#### Strategies

1. **Cheap** (`arm_id="cheap"`)
   - Model: `gpt-4o-mini`
   - Max tokens: 300
   - Temperature: 0.7
   - Use case: Simple queries

2. **Balanced** (`arm_id="balanced"`)
   - Model: `gpt-4o-mini`
   - Max tokens: 600
   - Temperature: 0.7
   - Use case: Medium queries

3. **Premium** (`arm_id="premium"`)
   - Model: `gpt-4o`
   - Max tokens: 1200
   - Temperature: 0.7
   - Use case: Complex queries

#### Algorithm: UCB (Upper Confidence Bound)

```python
# Selects arm with highest UCB score
ucb_score = average_reward + c * sqrt(ln(total_pulls) / arm_pulls)

Where:
- average_reward: Historical average reward
- c: Exploration constant (default: sqrt(2))
- total_pulls: Total queries processed
- arm_pulls: Times this arm was selected
```

#### RouterBench Cost-Quality Routing

```python
# Calculates reward based on RouterBench metrics
reward = quality - (cost_lambda * cost) - (latency_lambda * latency)

Where:
- quality: 0.0-1.0 (from quality judge)
- cost: $ per query (based on model pricing)
- cost_lambda: Cost weight (default: 0.001)
- latency: ms
- latency_lambda: Latency weight (default: 0.0001)
```

#### Learning

After each query:
1. Calculate reward (quality - cost - latency)
2. Update arm statistics:
   - Total pulls
   - Average reward
   - Total reward
3. Save state to `bandit_state.json`

---

### 4. Token Predictor (`tokenomics/ml/token_predictor.py`)

#### Purpose
Predicts optimal max_tokens for a query (replaces hardcoded values).

#### Modes

**1. Heuristic Mode** (current):
```python
base_tokens = {
    "simple": 200,
    "medium": 500,
    "complex": 800
}
length_factor = 1.0 + (query_tokens / 100) * 0.1
type_multiplier = 1.0  # Can vary by query type
predicted = base_tokens[complexity] * length_factor * type_multiplier
```

**2. ML Mode** (when 500+ samples):
```python
# Uses XGBoost regressor
Features:
- query_length (tokens)
- complexity (encoded)
- embedding_vector (first 10 dims)
- query_type (if available)

Target:
- actual_output_tokens

Model: XGBoostRegressor
```

#### Data Collection

Stores in `token_prediction_data.db`:
- Query text
- Query length (tokens)
- Complexity
- Embedding vector (first 10 dimensions)
- Predicted tokens
- Actual output tokens
- Model used
- Timestamp

#### Training

```python
# When 500+ samples available
1. Load training data from database
2. Extract features (query_length, complexity, embedding)
3. Train XGBoost regressor
4. Save model
5. Switch to ML mode
```

---

### 5. Compression (`tokenomics/compression/llmlingua_compressor.py`)

#### Purpose
Compresses long queries using LLMLingua-2 to reduce token count.

#### When Compression Triggers

```python
if query_tokens > 150 OR query_chars > 500:
    compressed = compress(query)
```

#### Compression Process

1. **Load LLMLingua-2 Model**: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
2. **Compress Query**: Reduces tokens while preserving meaning
3. **Compression Ratio**: Default 0.4 (40% of original size)

#### Example

```
Original: "Write a detailed explanation of quantum computing, including qubits, superposition, entanglement, quantum gates, quantum algorithms like Shor's algorithm and Grover's algorithm, quantum error correction, and applications in cryptography and optimization problems."
(200 tokens)

Compressed: "Explain quantum computing: qubits, superposition, entanglement, quantum gates, Shor's and Grover's algorithms, quantum error correction, cryptography and optimization applications."
(80 tokens)

Savings: 120 tokens (60% reduction)
```

---

### 6. Quality Judge (`tokenomics/judge/quality_judge.py`)

#### Purpose
Evaluates answer quality and compares optimized vs baseline.

#### Methods

**`quick_quality_check(query, response)`** (for cascading):
```python
# Fast heuristic check (no LLM call)
score = 0.0
# Length check (30%)
if len(response) > min_length:
    score += 0.3
# Completeness (40%)
if has_complete_indicators(response):
    score += 0.4
# Keyword overlap (30%)
overlap = calculate_overlap(query, response)
score += 0.3 * overlap
return score  # 0.0-1.0
```

**`judge(optimized_response, baseline_response, query)`** (full comparison):
```python
# Uses LLM-as-judge (gpt-4o)
Prompt: "Compare these two answers..."
Returns:
- winner: "optimized" | "baseline" | "equivalent"
- confidence: 0.0-1.0
- explanation: Text explanation
- quality_score: 0.0-1.0
```

---

## Data Structures

### QueryPlan

```python
@dataclass
class QueryPlan:
    query: str
    complexity: QueryComplexity  # SIMPLE, MEDIUM, COMPLEX
    token_budget: int
    allocations: List[TokenAllocation]
    model: Optional[str]
    use_retrieval: bool
    retrieved_context: List[str]
    compressed_prompt: Optional[str]
    context_quality_score: float  # 1.0 = full context
    context_compression_ratio: Optional[float]
    context_original_tokens: int
    context_allocated_tokens: int
```

### Strategy

```python
@dataclass
class Strategy:
    arm_id: str  # "cheap", "balanced", "premium"
    model: str  # "gpt-4o-mini", "gpt-4o"
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    metadata: Dict = field(default_factory=dict)
```

### CacheEntry

```python
@dataclass
class CacheEntry:
    query_hash: str
    query: str
    response: str
    tokens_used: int
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
```

### Query Result (Return Type)

```python
{
    "response": str,  # LLM response text
    "tokens_used": int,  # Total tokens (input + output)
    "input_tokens": int,
    "output_tokens": int,
    "cache_hit": bool,
    "cache_type": str,  # "exact", "semantic_direct", "context", "none"
    "similarity": Optional[float],
    "latency_ms": float,
    "strategy": str,  # "cheap", "balanced", "premium"
    "model": str,  # Model used
    "plan": QueryPlan,
    "reward": Optional[float],
    "component_savings": {
        "memory_layer": int,
        "orchestrator": int,
        "bandit": int,
        "total_savings": int,
    },
    "predicted_max_tokens": Optional[int],
    "cascading_escalated": bool,
    "quality_judge": Optional[Dict],
    ...
}
```

---

## Configuration

### TokenomicsConfig

Located in `tokenomics/config.py`:

```python
class TokenomicsConfig:
    llm: LLMConfig
    memory: MemoryConfig
    orchestrator: OrchestratorConfig
    bandit: BanditConfig
    judge: JudgeConfig
    cascading: CascadingConfig
    log_level: str
    log_file: str
```

### Key Configuration Options

**Memory**:
- `use_exact_cache`: Enable exact matching
- `use_semantic_cache`: Enable semantic search
- `similarity_threshold`: 0.70 (minimum for context)
- `direct_return_threshold`: 0.85 (minimum for direct return)
- `persistent_cache_path`: "tokenomics_cache.db"

**Orchestrator**:
- `default_token_budget`: 4000
- `max_context_tokens`: 8000
- `use_knapsack_optimization`: True

**Bandit**:
- `algorithm`: "ucb" (UCB, epsilon_greedy, thompson)
- `exploration_rate`: 0.1
- `state_file`: "bandit_state.json"

**Cascading**:
- `enabled`: True
- `quality_threshold`: 0.85
- `cheap_model`: "gpt-4o-mini"
- `premium_model`: "gpt-4o"
- `use_lightweight_check`: True
- `use_escalation_prediction`: True (NEW: Use bandit to predict escalation)
- `escalation_prediction_threshold`: 0.7 (NEW: Skip cheap if likelihood ≥ 70%)
- `use_escalation_prediction`: True (NEW: Use bandit to predict escalation)
- `escalation_prediction_threshold`: 0.7 (NEW: Skip cheap if likelihood ≥ 70%)

---

## Examples

### Example 1: Cache Hit

```python
# First query
result1 = platform.query("What is machine learning?")
# Cache miss, uses LLM, stores in cache
# Tokens: 250

# Second query (exact match)
result2 = platform.query("What is machine learning?")
# Cache hit (exact)
# Tokens: 0 (FREE!)
# Latency: ~0ms (instant)
```

### Example 2: Semantic Cache

```python
# First query
result1 = platform.query("What is machine learning?")
# Stores in cache

# Second query (semantically similar)
result2 = platform.query("Tell me about machine learning")
# Semantic cache hit (similarity > 0.85)
# Tokens: 0 (FREE!)
```

### Example 3: Cascading Inference

```python
# Complex query
result = platform.query("Explain quantum computing in detail")

# Flow (with escalation prediction):
# 1. Bandit selects: premium (gpt-4o)
# 2. Escalation prediction: likelihood = 0.85 (≥ 0.7 threshold)
# 3. Skip cheap model, go straight to gpt-4o
# 4. Generate with gpt-4o
# 5. Return premium answer (saved one API call!)

# Flow (without prediction, or prediction < threshold):
# 1. Bandit selects: premium (gpt-4o)
# 2. Escalation prediction: likelihood = 0.5 (< 0.7 threshold)
# 3. Cascading: start with gpt-4o-mini
# 4. Generate with gpt-4o-mini
# 5. Quality check: 0.75 (< 0.85 threshold)
# 6. Escalate to gpt-4o
# 7. Re-generate with gpt-4o
# 8. Return premium answer
# 9. Record outcome for learning

# Result:
# - Cost: Saved API call when prediction is accurate
# - Quality: Ensured high quality
# - Learning: System improves predictions over time
```

### Example 4: Token Prediction

```python
# Query
result = platform.query("What is 2+2?")

# Token prediction:
# - Complexity: simple
# - Query tokens: 5
# - Predicted: 241 tokens
# - Strategy default: 300 tokens
# - Override: Use 241 tokens

# Actual: 8 tokens generated
# Recorded: (predicted: 241, actual: 8) for ML training
```

---

## Troubleshooting

### Issue: Cache Hit But Still Shows Tokens

**Cause**: Baseline comparison still runs (for A/B testing)

**Solution**: This is expected. Cache hit = 0 tokens for optimized path. Baseline uses tokens for comparison.

### Issue: Prediction Way Off

**Cause**: Using heuristic mode (rules-based)

**Solution**: 
- Collect more data (500+ samples)
- Train ML model: `platform.train_token_predictor()`
- ML model will improve accuracy

### Issue: Bandit Savings = 0

**Cause**: Bandit selected same model as baseline

**Solution**: This is normal. Bandit is learning. Check `bandit_state.json` to see strategy performance.

### Issue: Quality Below Threshold

**Cause**: Cascading quality check failed

**Solution**:
- Check if cascading escalated (should escalate to premium)
- If not escalating, check cascading configuration
- Lower quality threshold if too strict

### Issue: No Training Data Collected

**Cause**: Cache hits don't generate training data

**Solution**: 
- Training data only collected on cache misses
- Run more diverse queries
- Check `token_prediction_data.db` for samples

---

## Key Files Reference

| File | Purpose | Key Methods |
|------|---------|-------------|
| `tokenomics/core.py` | Main platform | `query()`, `compare_with_baseline()` |
| `tokenomics/memory/memory_layer.py` | Caching | `retrieve_compressed()`, `store()` |
| `tokenomics/orchestrator/orchestrator.py` | Planning | `plan_query()`, `analyze_complexity()` |
| `tokenomics/bandit/bandit.py` | Routing | `select_strategy_cost_aware()` |
| `tokenomics/ml/token_predictor.py` | Prediction | `predict()`, `record_prediction()` |
| `tokenomics/compression/llmlingua_compressor.py` | Compression | `compress_query_if_needed()` |
| `tokenomics/judge/quality_judge.py` | Quality | `judge()`, `quick_quality_check()` |

---

## Summary

The Tokenomics Platform is a **smart proxy** that:

1. **Remembers** (Memory Layer): Caches answers, finds similar queries
2. **Plans** (Orchestrator): Allocates tokens optimally
3. **Predicts** (Token Predictor): Guesses answer length
4. **Routes** (Bandit): Chooses best model
5. **Compresses** (LLMLingua): Shrinks long queries
6. **Ensures Quality** (Cascading + Judge): Maintains quality while saving cost
7. **Learns** (Data Collection): Improves over time

**Result**: Same quality answers, lower cost, faster responses (when cached).

---

*For questions or issues, check logs in `tokenomics.log` or examine component-specific files.*
