# A/B Comparison Test Report

**Generated:** 2025-12-18 13:25:42

---

## Executive Summary

- **Total Queries:** 55
- **Successful Queries:** 55
- **Failed Queries:** 0

### Overall Savings

- **Token Savings:** 2,386 tokens (12.9%)
- **Cost Savings:** $0.0019 (18.3%)
- **Latency Improvement:** -6717ms (-107.5%)

### Quality Preservation

- **Quality Preservation Rate:** 96.4%
- **Average Quality Confidence:** 0.89
- **Quality Winners:** Optimized: 25, Baseline: 2, Equivalent: 28

## Component Analysis

### Memory Layer

- **Exact Cache Hits:** 5
- **Semantic Cache Hits:** 6
- **Cache Misses:** 44
- **Cache Hit Rate:** 20.0%

- **Context Injection Usage:** 4
- **Preference Learning Usage:** 52

### Token Orchestrator

- **Complexity Distribution:** Simple: 0, Medium: 0, Complex: 0
- **Query Compression:** 0 queries compressed
- **Average Query Compression Ratio:** 1.000
- **Context Compression:** 4 contexts compressed
- **Average Context Compression Ratio:** 1.000

### Bandit Optimizer

- **Strategy Distribution:** Cheap: 43, Balanced: 12, Premium: 0
- **Context-Aware Routing:** 0 queries

## Savings Breakdown

### By Category

- **simple:** -15 tokens ($-0.0000) across 10 queries
- **medium:** 1,062 tokens ($0.0006) across 10 queries
- **complex:** 620 tokens ($0.0004) across 10 queries
- **duplicate:** 226 tokens ($0.0001) across 5 queries
- **paraphrase_original:** 659 tokens ($0.0004) across 5 queries
- **paraphrase:** 575 tokens ($0.0007) across 5 queries
- **context_heavy:** 2 tokens ($0.0000) across 5 queries
- **long:** -743 tokens ($-0.0003) across 5 queries

### By Cache Type

- **none:** 1,976 tokens ($0.0012) across 44 queries
- **exact:** 226 tokens ($0.0001) across 5 queries
- **semantic_direct:** 1,032 tokens ($0.0006) across 2 queries
- **context:** -848 tokens ($0.0000) across 4 queries

### By Strategy

- **cheap:** 3,657 tokens ($0.0026) across 43 queries
- **balanced:** -1,271 tokens ($-0.0006) across 12 queries

## Detailed Results

| Query | Category | Baseline Tokens | Optimized Tokens | Savings | Baseline Cost | Optimized Cost | Cost Savings | Quality Winner |
|-------|----------|-----------------|------------------|---------|---------------|----------------|--------------|----------------|
| What is the capital of France?... | simple | 21 | 21 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| How many days are in a week?... | simple | 24 | 24 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is 2 + 2?... | simple | 23 | 23 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is the largest planet in our solar system?... | simple | 66 | 66 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| Who wrote Romeo and Juliet?... | simple | 45 | 45 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is the speed of light?... | simple | 84 | 86 | -2 (-2.4%) | $0.0000 | $0.0000 | $-0.0000 | equivalent |
| What is the chemical symbol for gold?... | simple | 23 | 23 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| How many continents are there?... | simple | 74 | 78 | -4 (-5.4%) | $0.0000 | $0.0000 | $-0.0000 | equivalent |
| What is the smallest prime number?... | simple | 46 | 46 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is the boiling point of water in Celsius?... | simple | 47 | 56 | -9 (-19.1%) | $0.0000 | $0.0000 | $-0.0000 | optimized |
| Explain how photosynthesis works in plants.... | medium | 527 | 615 | -88 (-16.7%) | $0.0003 | $0.0004 | $-0.0001 | equivalent |
| What are the main differences between Python and J... | medium | 530 | 318 | 212 (40.0%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Describe the water cycle in nature.... | medium | 366 | 314 | 52 (14.2%) | $0.0002 | $0.0002 | $0.0000 | equivalent |
| How does a refrigerator work?... | medium | 382 | 313 | 69 (18.1%) | $0.0002 | $0.0002 | $0.0000 | optimized |
| What is the difference between HTTP and HTTPS?... | medium | 442 | 316 | 126 (28.5%) | $0.0003 | $0.0002 | $0.0001 | equivalent |
| Explain the concept of supply and demand in econom... | medium | 529 | 317 | 212 (40.1%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| How do vaccines work in the human body?... | medium | 443 | 316 | 127 (28.7%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| What is the difference between RAM and ROM?... | medium | 410 | 316 | 94 (22.9%) | $0.0002 | $0.0002 | $0.0001 | optimized |
| Explain the greenhouse effect.... | medium | 416 | 312 | 104 (25.0%) | $0.0002 | $0.0002 | $0.0001 | optimized |
| How does a computer process information?... | medium | 468 | 314 | 154 (32.9%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Design a microservices architecture for an e-comme... | complex | 553 | 641 | -88 (-15.9%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
| Explain the mathematical foundations of machine le... | complex | 551 | 339 | 212 (38.5%) | $0.0003 | $0.0002 | $0.0001 | equivalent |
| Create a comprehensive security strategy for a clo... | complex | 547 | 335 | 212 (38.8%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Design a distributed system for real-time analytic... | complex | 548 | 636 | -88 (-16.1%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
| Explain the principles of quantum computing, inclu... | complex | 551 | 339 | 212 (38.5%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Design a scalable database architecture for a soci... | complex | 548 | 636 | -88 (-16.1%) | $0.0003 | $0.0004 | $-0.0001 | equivalent |
| Create a comprehensive DevOps pipeline for a micro... | complex | 548 | 636 | -88 (-16.1%) | $0.0003 | $0.0004 | $-0.0001 | equivalent |
| Explain the architecture of a modern web browser, ... | complex | 543 | 331 | 212 (39.0%) | $0.0003 | $0.0002 | $0.0001 | equivalent |
| Design a machine learning pipeline for fraud detec... | complex | 546 | 634 | -88 (-16.1%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
| Explain the principles of distributed consensus al... | complex | 548 | 336 | 212 (38.7%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| What is the capital of France?... | duplicate | 21 | 0 | 21 (100.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| How many days are in a week?... | duplicate | 24 | 0 | 24 (100.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is 2 + 2?... | duplicate | 23 | 0 | 23 (100.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What is the largest planet in our solar system?... | duplicate | 113 | 0 | 113 (100.0%) | $0.0001 | $0.0000 | $0.0001 | baseline |
| Who wrote Romeo and Juliet?... | duplicate | 45 | 0 | 45 (100.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| How do I bake a cake?... | paraphrase_original | 526 | 314 | 212 (40.3%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Explain the theory of evolution.... | paraphrase_original | 515 | 313 | 202 (39.2%) | $0.0003 | $0.0002 | $0.0001 | equivalent |
| What causes climate change?... | paraphrase_original | 385 | 312 | 73 (19.0%) | $0.0002 | $0.0002 | $0.0000 | optimized |
| How does the internet work?... | paraphrase_original | 525 | 313 | 212 (40.4%) | $0.0003 | $0.0002 | $0.0001 | equivalent |
| What is artificial intelligence?... | paraphrase_original | 266 | 306 | -40 (-15.0%) | $0.0002 | $0.0002 | $-0.0000 | equivalent |
| What is the recipe for baking a cake?... | paraphrase | 528 | 0 | 528 (100.0%) | $0.0003 | $0.0000 | $0.0003 | optimized |
| Can you describe how evolution works?... | paraphrase | 482 | 617 | -135 (-28.0%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Why does the climate change?... | paraphrase | 504 | 0 | 504 (100.0%) | $0.0003 | $0.0000 | $0.0003 | optimized |
| Explain how the internet functions.... | paraphrase | 525 | 616 | -91 (-17.3%) | $0.0003 | $0.0002 | $0.0001 | optimized |
| Can you define what AI is?... | paraphrase | 273 | 504 | -231 (-84.6%) | $0.0002 | $0.0002 | $-0.0000 | optimized |
| What was the main goal of the Apollo program?... | context_heavy | 145 | 150 | -5 (-3.4%) | $0.0001 | $0.0001 | $-0.0000 | equivalent |
| How many Apollo missions landed on the moon?... | context_heavy | 84 | 83 | 1 (1.2%) | $0.0000 | $0.0000 | $0.0000 | optimized |
| Who was the first person to walk on the moon?... | context_heavy | 82 | 76 | 6 (7.3%) | $0.0000 | $0.0000 | $0.0000 | baseline |
| What was the name of the Apollo 11 command module?... | context_heavy | 33 | 33 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| What year did the Apollo program end?... | context_heavy | 42 | 42 | 0 (0.0%) | $0.0000 | $0.0000 | $0.0000 | equivalent |
| I need a comprehensive explanation of how modern w... | long | 596 | 684 | -88 (-14.8%) | $0.0003 | $0.0004 | $-0.0001 | equivalent |
| Explain in detail the architecture of distributed ... | long | 588 | 979 | -391 (-66.5%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
| I want a thorough guide on machine learning model ... | long | 583 | 671 | -88 (-15.1%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
| Provide a detailed explanation of cloud computing ... | long | 593 | 681 | -88 (-14.8%) | $0.0003 | $0.0004 | $-0.0001 | equivalent |
| I need comprehensive information about database de... | long | 588 | 676 | -88 (-15.0%) | $0.0003 | $0.0004 | $-0.0001 | optimized |
