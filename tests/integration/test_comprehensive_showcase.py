#!/usr/bin/env python3
"""
Comprehensive Diagnostic Test Suite for Tokenomics Platform

This showcase-quality test validates EVERY component of the platform:
- Smart Memory Layer (exact cache, semantic cache, context injection, preferences, LLMLingua compression)
- Token-Aware Orchestrator (complexity analysis, token allocation, knapsack optimization)
- Bandit Optimizer with RouterBench (strategy selection, cost-quality routing, reward learning)
- Quality Judge (response evaluation, baseline comparison)

Generates detailed reports in JSON, HTML, Markdown, and CSV formats with complete documentation.

Test Dataset: 52 queries across 8 categories designed to trigger specific features.
"""

import os
import sys
import json
import time
import csv
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import traceback

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for enum and datetime objects."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@dataclass
class QueryResult:
    """Comprehensive result of a single query test with all metrics."""
    query_id: int
    query: str
    category: str
    subcategory: str = ""
    expected_behaviors: List[str] = field(default_factory=list)
    
    # Response data
    response: str = ""
    success: bool = False
    error: Optional[str] = None
    
    # Timing
    latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    
    # Token metrics (optimized path)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Baseline metrics (for comparison)
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_total_tokens: int = 0
    
    # Savings calculations
    tokens_saved: int = 0
    savings_percentage: float = 0.0
    
    # Memory layer metrics
    cache_hit: bool = False
    cache_type: str = "miss"  # exact, semantic_direct, context, miss
    similarity: Optional[float] = None
    context_injected: bool = False
    context_tokens_added: int = 0
    preference_used: bool = False
    preference_tone: Optional[str] = None
    preference_format: Optional[str] = None
    
    # LLMLingua compression metrics (CRITICAL)
    context_compressed: bool = False
    context_original_tokens: int = 0
    context_compressed_tokens: int = 0
    context_compression_ratio: float = 1.0
    query_compressed: bool = False
    query_original_tokens: int = 0
    query_compressed_tokens: int = 0
    query_compression_ratio: float = 1.0
    total_compression_savings: int = 0
    
    # Orchestrator metrics
    complexity: str = ""
    token_budget: int = 0
    max_response_tokens: int = 0
    token_efficiency: float = 0.0
    model_used: str = ""
    
    # Bandit + RouterBench metrics
    strategy_selected: str = ""
    strategy_model: str = ""
    strategy_reward: Optional[float] = None
    cost_per_query: float = 0.0
    routerbench_efficiency: float = 0.0
    
    # Quality metrics
    quality_winner: str = ""  # optimized, baseline, equivalent
    quality_confidence: float = 0.0
    quality_explanation: str = ""
    
    # Component savings breakdown
    memory_savings: int = 0
    orchestrator_savings: int = 0
    bandit_savings: int = 0
    
    # Raw response data for debugging
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status of a platform component."""
    name: str
    available: bool
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSummary:
    """Comprehensive summary of all test results."""
    # Test info
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Overall metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0
    
    # Cache performance
    exact_cache_hits: int = 0
    semantic_direct_hits: int = 0
    semantic_context_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    # Token metrics
    total_optimized_tokens: int = 0
    total_baseline_tokens: int = 0
    total_tokens_saved: int = 0
    average_savings_percentage: float = 0.0
    
    # Latency
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # LLMLingua Compression
    llmlingua_available: bool = False
    llmlingua_model: str = ""
    context_compressions: int = 0
    query_compressions: int = 0
    total_compressions: int = 0
    average_context_compression_ratio: float = 1.0
    average_query_compression_ratio: float = 1.0
    compression_tokens_saved: int = 0
    
    # Orchestrator
    simple_queries: int = 0
    medium_queries: int = 0
    complex_queries: int = 0
    average_token_budget: float = 0.0
    average_token_efficiency: float = 0.0
    
    # Bandit + RouterBench
    cheap_strategy_uses: int = 0
    balanced_strategy_uses: int = 0
    premium_strategy_uses: int = 0
    average_reward: float = 0.0
    total_cost: float = 0.0
    average_cost_per_query: float = 0.0
    
    # Quality
    optimized_wins: int = 0
    baseline_wins: int = 0
    equivalent_results: int = 0
    average_quality_confidence: float = 0.0
    
    # Preference Learning
    preferences_detected: int = 0
    unique_tones: List[str] = field(default_factory=list)
    unique_formats: List[str] = field(default_factory=list)
    
    # By category breakdown
    category_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Component health
    component_health: List[Dict] = field(default_factory=list)


class ComprehensiveShowcaseTest:
    """Main test class for comprehensive platform validation."""
    
    def __init__(self):
        self.platform: Optional[TokenomicsPlatform] = None
        self.results: List[QueryResult] = []
        self.summary: TestSummary = TestSummary()
        self.component_health: List[ComponentHealth] = []
        
    def initialize_platform(self) -> bool:
        """Initialize the Tokenomics platform."""
        print("\n" + "=" * 70)
        print("TOKENOMICS PLATFORM - COMPREHENSIVE DIAGNOSTIC TEST SUITE")
        print("=" * 70)
        print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.summary.test_start = datetime.now()
        
        try:
            config = TokenomicsConfig.from_env()
            self.platform = TokenomicsPlatform(config=config)
            print("\n[OK] Platform initialized successfully")
            return True
        except Exception as e:
            print(f"\n[FAIL] Platform initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def check_component_health(self) -> List[ComponentHealth]:
        """Check health of all platform components."""
        print("\n" + "-" * 50)
        print("COMPONENT HEALTH CHECK")
        print("-" * 50)
        
        self.component_health = []
        
        # 1. Memory Layer
        if self.platform.memory:
            health = ComponentHealth(name="Memory Layer", available=True, status="OK")
            health.details = {"type": "SmartMemoryLayer"}
            print("[OK] Memory Layer: Available")
            
            # 1a. Exact Cache
            if self.platform.memory.exact_cache:
                stats = self.platform.memory.exact_cache.stats()
                self.component_health.append(ComponentHealth(
                    name="Exact Cache",
                    available=True,
                    status="OK",
                    details={"entries": stats.get('size', 0), "max_size": stats.get('max_size', 1000)}
                ))
                print(f"  [OK] Exact Cache: {stats.get('size', 0)} entries")
            else:
                self.component_health.append(ComponentHealth(name="Exact Cache", available=False, status="Disabled"))
                print("  [--] Exact Cache: Disabled")
            
            # 1b. Semantic Cache
            if self.platform.memory.vector_store:
                vs_type = type(self.platform.memory.vector_store).__name__
                self.component_health.append(ComponentHealth(
                    name="Semantic Cache",
                    available=True,
                    status="OK",
                    details={"type": vs_type}
                ))
                print(f"  [OK] Semantic Cache: {vs_type}")
            else:
                self.component_health.append(ComponentHealth(name="Semantic Cache", available=False, status="Disabled"))
                print("  [--] Semantic Cache: Disabled")
            
            # 1c. LLMLingua Compression
            if self.platform.memory.llmlingua:
                is_available = self.platform.memory.llmlingua.is_available()
                if is_available:
                    model_name = self.platform.memory.llmlingua.model_name
                    self.summary.llmlingua_available = True
                    self.summary.llmlingua_model = model_name
                    self.component_health.append(ComponentHealth(
                        name="LLMLingua",
                        available=True,
                        status="OK",
                        details={"model": model_name, "compression_ratio": 0.4}
                    ))
                    print(f"  [OK] LLMLingua: {model_name}")
                else:
                    error = self.platform.memory.llmlingua.get_init_error() or "Unknown error"
                    self.component_health.append(ComponentHealth(
                        name="LLMLingua",
                        available=False,
                        status="Unavailable",
                        details={"error": error[:100]}
                    ))
                    print(f"  [--] LLMLingua: Unavailable ({error[:50]}...)")
            else:
                self.component_health.append(ComponentHealth(name="LLMLingua", available=False, status="Not configured"))
                print("  [--] LLMLingua: Not configured")
            
            # 1d. User Preferences
            if self.platform.memory.user_preferences:
                self.component_health.append(ComponentHealth(
                    name="User Preferences",
                    available=True,
                    status="OK",
                    details={"confidence": self.platform.memory.user_preferences.confidence}
                ))
                print(f"  [OK] User Preferences: Active")
            
            self.component_health.append(health)
        else:
            self.component_health.append(ComponentHealth(name="Memory Layer", available=False, status="FAIL"))
            print("[FAIL] Memory Layer: Not available")
        
        # 2. Orchestrator
        if self.platform.orchestrator:
            budget = self.platform.orchestrator.default_token_budget
            max_ctx = self.platform.orchestrator.max_context_tokens
            self.component_health.append(ComponentHealth(
                name="Orchestrator",
                available=True,
                status="OK",
                details={"default_budget": budget, "max_context": max_ctx}
            ))
            print(f"[OK] Orchestrator: Budget={budget}, MaxContext={max_ctx}")
        else:
            self.component_health.append(ComponentHealth(name="Orchestrator", available=False, status="FAIL"))
            print("[FAIL] Orchestrator: Not available")
        
        # 3. Bandit Optimizer
        if self.platform.bandit:
            num_strategies = len(self.platform.bandit.arms)
            strategy_names = list(self.platform.bandit.arms.keys())
            self.component_health.append(ComponentHealth(
                name="Bandit Optimizer",
                available=True,
                status="OK",
                details={"num_strategies": num_strategies, "strategies": strategy_names, "algorithm": self.platform.bandit.algorithm.value}
            ))
            print(f"[OK] Bandit Optimizer: {num_strategies} strategies ({', '.join(strategy_names)})")
        else:
            self.component_health.append(ComponentHealth(name="Bandit Optimizer", available=False, status="FAIL"))
            print("[FAIL] Bandit Optimizer: Not available")
        
        # 4. Quality Judge
        if self.platform.quality_judge:
            self.component_health.append(ComponentHealth(
                name="Quality Judge",
                available=True,
                status="OK",
                details={"model": self.platform.config.judge.model}
            ))
            print(f"[OK] Quality Judge: {self.platform.config.judge.model}")
        else:
            self.component_health.append(ComponentHealth(name="Quality Judge", available=False, status="Not configured"))
            print("[--] Quality Judge: Not configured (testing only)")
        
        # 5. LLM Provider
        if self.platform.llm_provider:
            provider_name = type(self.platform.llm_provider).__name__
            self.component_health.append(ComponentHealth(
                name="LLM Provider",
                available=True,
                status="OK",
                details={"type": provider_name}
            ))
            print(f"[OK] LLM Provider: {provider_name}")
        else:
            self.component_health.append(ComponentHealth(name="LLM Provider", available=False, status="FAIL"))
            print("[FAIL] LLM Provider: Not available")
        
        # Store in summary
        self.summary.component_health = [asdict(h) for h in self.component_health]
        
        return self.component_health
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive test dataset with 52 queries across 8 categories."""
        print("\n" + "-" * 50)
        print("CREATING TEST DATASET (52 queries)")
        print("-" * 50)
        
        dataset = []
        
        # ==========================================================
        # CATEGORY 1: EXACT CACHE TESTING (6 queries)
        # Tests: exact match storage and retrieval
        # ==========================================================
        exact_cache_queries = [
            # First occurrence - should be cache miss, then stored
            {"query": "What is Python programming language?", "category": "exact_cache", "subcategory": "first_occurrence", 
             "expected": ["cache_miss", "store_in_cache"]},
            # Exact repeat - should be cache hit
            {"query": "What is Python programming language?", "category": "exact_cache", "subcategory": "exact_repeat", 
             "expected": ["exact_cache_hit", "zero_tokens"]},
            # Another first occurrence
            {"query": "Explain what machine learning is", "category": "exact_cache", "subcategory": "first_occurrence", 
             "expected": ["cache_miss", "store_in_cache"]},
            # Exact repeat
            {"query": "Explain what machine learning is", "category": "exact_cache", "subcategory": "exact_repeat", 
             "expected": ["exact_cache_hit"]},
            # Third unique query
            {"query": "What are the benefits of cloud computing?", "category": "exact_cache", "subcategory": "first_occurrence", 
             "expected": ["cache_miss"]},
            # Exact repeat
            {"query": "What are the benefits of cloud computing?", "category": "exact_cache", "subcategory": "exact_repeat", 
             "expected": ["exact_cache_hit"]},
        ]
        dataset.extend(exact_cache_queries)
        
        # ==========================================================
        # CATEGORY 2: SEMANTIC CACHE TESTING (8 queries)
        # Tests: semantic similarity matching, context injection
        # ==========================================================
        semantic_cache_queries = [
            # Base query to seed semantic cache
            {"query": "What is artificial intelligence and how does it work?", "category": "semantic_cache", 
             "subcategory": "seed_query", "expected": ["cache_miss", "semantic_store"]},
            # High similarity - should trigger semantic direct hit (>0.85)
            {"query": "Explain artificial intelligence and its workings", "category": "semantic_cache", 
             "subcategory": "high_similarity", "expected": ["semantic_match"]},
            # Medium similarity - should trigger context injection (0.75-0.85)
            {"query": "Tell me about AI technology", "category": "semantic_cache", 
             "subcategory": "medium_similarity", "expected": ["context_injection", "llmlingua_compress"]},
            # Another seed
            {"query": "What is deep learning and neural networks?", "category": "semantic_cache", 
             "subcategory": "seed_query", "expected": ["cache_miss"]},
            # Similar variant
            {"query": "Explain deep learning with neural network examples", "category": "semantic_cache", 
             "subcategory": "high_similarity", "expected": ["semantic_match"]},
            # Medium similarity variant
            {"query": "How do neural nets learn?", "category": "semantic_cache", 
             "subcategory": "medium_similarity", "expected": ["context_injection"]},
            # Low similarity - should be cache miss
            {"query": "What is quantum computing?", "category": "semantic_cache", 
             "subcategory": "low_similarity", "expected": ["cache_miss"]},
            # Another context injection test
            {"query": "Describe how AI systems are built", "category": "semantic_cache", 
             "subcategory": "context_test", "expected": ["possible_context"]},
        ]
        dataset.extend(semantic_cache_queries)
        
        # ==========================================================
        # CATEGORY 3: LLMLINGUA COMPRESSION (8 queries)
        # Tests: long query compression, context compression
        # ==========================================================
        # Long text for compression tests
        long_text_1 = " ".join(["The quick brown fox jumps over the lazy dog."] * 60)
        long_text_2 = " ".join(["Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."] * 25)
        long_text_3 = " ".join(["Data science combines statistics, programming, and domain expertise to extract insights from data."] * 35)
        long_text_4 = " ".join(["Cloud computing provides on-demand access to computing resources over the internet."] * 40)
        
        compression_queries = [
            # Long query that should trigger query compression (>200 tokens or >800 chars)
            {"query": f"Please summarize this text and extract key points: {long_text_1}", 
             "category": "compression", "subcategory": "long_query", 
             "expected": ["query_compression", "llmlingua_active"]},
            # Another long query
            {"query": f"Analyze this document and provide insights: {long_text_2}", 
             "category": "compression", "subcategory": "long_query", 
             "expected": ["query_compression"]},
            # Third long query
            {"query": f"Extract the main ideas from: {long_text_3}", 
             "category": "compression", "subcategory": "long_query", 
             "expected": ["query_compression"]},
            # Fourth long query
            {"query": f"Summarize and explain: {long_text_4}", 
             "category": "compression", "subcategory": "long_query", 
             "expected": ["query_compression"]},
            # Medium-length query (should NOT compress)
            {"query": "Explain the concept of containerization in software development and how Docker works", 
             "category": "compression", "subcategory": "medium_query", 
             "expected": ["no_compression"]},
            # Short query (should NOT compress)
            {"query": "What is Kubernetes?", 
             "category": "compression", "subcategory": "short_query", 
             "expected": ["no_compression"]},
            # Query that might trigger context compression (after semantic cache is seeded)
            {"query": "Compare artificial intelligence approaches", 
             "category": "compression", "subcategory": "context_compression", 
             "expected": ["possible_context_compression"]},
            # Another context compression candidate
            {"query": "How is deep learning different from traditional ML?", 
             "category": "compression", "subcategory": "context_compression", 
             "expected": ["possible_context_compression"]},
        ]
        dataset.extend(compression_queries)
        
        # ==========================================================
        # CATEGORY 4: ORCHESTRATOR TOKEN ALLOCATION (8 queries)
        # Tests: complexity analysis, token budgeting
        # ==========================================================
        orchestrator_queries = [
            # Simple queries (low complexity, minimal tokens)
            {"query": "Hi", "category": "orchestrator", "subcategory": "simple", 
             "expected": ["low_complexity", "small_budget"]},
            {"query": "2+2?", "category": "orchestrator", "subcategory": "simple", 
             "expected": ["low_complexity"]},
            {"query": "What is the capital of Japan?", "category": "orchestrator", "subcategory": "simple", 
             "expected": ["low_complexity", "factual"]},
            # Medium complexity queries
            {"query": "Compare Python and JavaScript for web development", "category": "orchestrator", 
             "subcategory": "medium", "expected": ["medium_complexity"]},
            {"query": "What are the pros and cons of microservices architecture?", "category": "orchestrator", 
             "subcategory": "medium", "expected": ["medium_complexity"]},
            # Complex queries (high complexity, larger budget)
            {"query": "Write a comprehensive analysis of distributed systems design patterns, including CAP theorem implications, consistency models, and partition tolerance strategies", 
             "category": "orchestrator", "subcategory": "complex", 
             "expected": ["high_complexity", "large_budget"]},
            {"query": "Explain the mathematical foundations of gradient descent optimization in neural networks, including backpropagation, learning rate scheduling, and convergence properties", 
             "category": "orchestrator", "subcategory": "complex", 
             "expected": ["high_complexity"]},
            # Budget constraint test
            {"query": "Give me a brief one-sentence answer: what is recursion?", 
             "category": "orchestrator", "subcategory": "constrained", 
             "expected": ["token_efficiency"]},
        ]
        dataset.extend(orchestrator_queries)
        
        # ==========================================================
        # CATEGORY 5: BANDIT + ROUTERBENCH (8 queries)
        # Tests: strategy selection, cost-quality routing
        # ==========================================================
        bandit_queries = [
            # Cheap strategy triggers (simple factual questions)
            {"query": "What color is the sky?", "category": "bandit", "subcategory": "cheap_trigger", 
             "expected": ["cheap_strategy", "low_cost"]},
            {"query": "How many days in a week?", "category": "bandit", "subcategory": "cheap_trigger", 
             "expected": ["cheap_strategy"]},
            {"query": "What is 10 * 5?", "category": "bandit", "subcategory": "cheap_trigger", 
             "expected": ["cheap_strategy"]},
            # Balanced strategy triggers
            {"query": "Explain the difference between HTTP and HTTPS", "category": "bandit", 
             "subcategory": "balanced_trigger", "expected": ["balanced_or_cheap"]},
            {"query": "What are REST API best practices?", "category": "bandit", 
             "subcategory": "balanced_trigger", "expected": ["balanced_strategy"]},
            # Premium strategy triggers (complex analysis)
            {"query": "Provide a detailed technical analysis of blockchain consensus mechanisms including proof-of-work, proof-of-stake, and their security implications", 
             "category": "bandit", "subcategory": "premium_trigger", 
             "expected": ["premium_strategy", "high_quality"]},
            {"query": "Write an in-depth comparison of SQL vs NoSQL databases for enterprise applications, covering scalability, consistency, and use cases", 
             "category": "bandit", "subcategory": "premium_trigger", 
             "expected": ["premium_strategy"]},
            # Exploration test
            {"query": "What is the best programming paradigm?", "category": "bandit", 
             "subcategory": "exploration", "expected": ["any_strategy", "reward_tracking"]},
        ]
        dataset.extend(bandit_queries)
        
        # ==========================================================
        # CATEGORY 6: PREFERENCE LEARNING (6 queries)
        # Tests: tone detection, format learning
        # ==========================================================
        preference_queries = [
            # Tone detection
            {"query": "Please explain APIs in a formal, technical manner suitable for developers", 
             "category": "preferences", "subcategory": "formal_tone", 
             "expected": ["tone_detection", "technical"]},
            {"query": "Hey, can you give me a simple explanation of databases? Keep it casual!", 
             "category": "preferences", "subcategory": "casual_tone", 
             "expected": ["tone_detection", "simple"]},
            # Format detection
            {"query": "List the top 5 programming languages with their use cases", 
             "category": "preferences", "subcategory": "list_format", 
             "expected": ["format_detection", "list"]},
            {"query": "Show me how to write a Python function with code examples", 
             "category": "preferences", "subcategory": "code_format", 
             "expected": ["format_detection", "code"]},
            # Combined
            {"query": "Give me a detailed technical overview of GraphQL in a structured format", 
             "category": "preferences", "subcategory": "combined", 
             "expected": ["tone_and_format"]},
            {"query": "Briefly explain what Docker containers are", 
             "category": "preferences", "subcategory": "concise", 
             "expected": ["concise_preference"]},
        ]
        dataset.extend(preference_queries)
        
        # ==========================================================
        # CATEGORY 7: EDGE CASES (4 queries)
        # Tests: error handling, graceful degradation
        # ==========================================================
        edge_case_queries = [
            # Empty query
            {"query": "", "category": "edge_case", "subcategory": "empty", 
             "expected": ["handle_empty", "graceful_error"]},
            # Minimal query
            {"query": "x", "category": "edge_case", "subcategory": "minimal", 
             "expected": ["handle_minimal"]},
            # Special characters
            {"query": "What is @#$%^&*() in programming?", "category": "edge_case", 
             "subcategory": "special_chars", "expected": ["handle_special"]},
            # Very repetitive
            {"query": "very " * 100 + "important question?", "category": "edge_case", 
             "subcategory": "repetitive", "expected": ["handle_repetitive", "possible_compression"]},
        ]
        dataset.extend(edge_case_queries)
        
        # ==========================================================
        # CATEGORY 8: QUALITY EVALUATION (4 queries)
        # Tests: quality judge, baseline comparison
        # ==========================================================
        quality_queries = [
            # Creative writing
            {"query": "Write a creative haiku about software engineering", 
             "category": "quality", "subcategory": "creative", 
             "expected": ["creative_response", "quality_comparison"]},
            # Explanation quality
            {"query": "Explain recursion to a 5-year-old", 
             "category": "quality", "subcategory": "simplified", 
             "expected": ["simplified_explanation", "quality_assessment"]},
            # Technical accuracy
            {"query": "What is the time complexity of binary search?", 
             "category": "quality", "subcategory": "technical", 
             "expected": ["accurate_answer", "technical_quality"]},
            # Comprehensive answer
            {"query": "What are the SOLID principles in software design?", 
             "category": "quality", "subcategory": "comprehensive", 
             "expected": ["comprehensive_response"]},
        ]
        dataset.extend(quality_queries)
        
        print(f"Created {len(dataset)} test queries across {len(set(q['category'] for q in dataset))} categories:")
        categories = {}
        for q in dataset:
            cat = q['category']
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count} queries")
        
        return dataset
    
    def run_single_query(self, query_data: Dict[str, Any], query_id: int) -> QueryResult:
        """Run a single query and collect ALL metrics."""
        result = QueryResult(
            query_id=query_id,
            query=query_data["query"][:500],  # Truncate for storage
            category=query_data["category"],
            subcategory=query_data.get("subcategory", ""),
            expected_behaviors=query_data.get("expected", []),
        )
        
        # Handle empty query
        if not query_data["query"].strip():
            result.error = "Empty query"
            result.success = False
            return result
        
        try:
            # Run optimized query through platform
            start_time = time.time()
            response = self.platform.query(
                query=query_data["query"],
                use_cache=True,
                use_bandit=True,
                use_compression=True,
                use_cost_aware_routing=True,
            )
            end_time = time.time()
            
            result.latency_ms = (end_time - start_time) * 1000
            result.success = True
            
            # Store raw response for debugging
            if isinstance(response, dict):
                # Create a sanitized copy for storage
                result.raw_response = {k: v for k, v in response.items() 
                                      if k not in ['plan', 'response'] and not callable(v)}
            
            # Extract all metrics
            if isinstance(response, dict):
                result.response = str(response.get("response", ""))[:500]
                
                # Token metrics
                result.input_tokens = response.get("input_tokens", 0)
                result.output_tokens = response.get("output_tokens", 0)
                result.total_tokens = response.get("tokens_used", result.input_tokens + result.output_tokens)
                
                # Cache metrics
                result.cache_hit = response.get("cache_hit", False)
                result.cache_type = response.get("cache_type") or "miss"
                result.similarity = response.get("similarity")
                
                # Memory metrics from memory_metrics dict
                memory_metrics = response.get("memory_metrics", {})
                result.context_injected = memory_metrics.get("context_injected", False)
                result.context_tokens_added = memory_metrics.get("context_tokens_added", 0)
                result.preference_used = memory_metrics.get("preferences_used", False)
                result.preference_tone = memory_metrics.get("preference_tone")
                result.preference_format = memory_metrics.get("preference_format")
                
                # Compression metrics (CRITICAL - from compression_metrics dict)
                compression_metrics = response.get("compression_metrics", {})
                result.context_compressed = compression_metrics.get("context_compressed", False)
                result.context_original_tokens = compression_metrics.get("context_original_tokens", 0)
                result.context_compressed_tokens = compression_metrics.get("context_compressed_tokens", 0)
                result.context_compression_ratio = compression_metrics.get("context_compression_ratio", 1.0)
                result.query_compressed = compression_metrics.get("query_compressed", False)
                result.query_original_tokens = compression_metrics.get("query_original_tokens", 0)
                result.query_compressed_tokens = compression_metrics.get("query_compressed_tokens", 0)
                result.query_compression_ratio = compression_metrics.get("query_compression_ratio", 1.0)
                result.total_compression_savings = compression_metrics.get("total_compression_savings", 0)
                
                # Orchestrator metrics
                orchestrator_metrics = response.get("orchestrator_metrics", {})
                result.complexity = orchestrator_metrics.get("complexity", response.get("query_type", ""))
                result.token_budget = orchestrator_metrics.get("token_budget", 0)
                result.max_response_tokens = orchestrator_metrics.get("max_response_tokens", 0)
                result.token_efficiency = orchestrator_metrics.get("token_efficiency", 0.0)
                result.model_used = orchestrator_metrics.get("model", response.get("model_used", ""))
                
                # Bandit metrics
                bandit_metrics = response.get("bandit_metrics", {})
                result.strategy_selected = bandit_metrics.get("strategy", response.get("strategy", ""))
                result.strategy_model = bandit_metrics.get("model", "")
                result.strategy_reward = bandit_metrics.get("reward") or response.get("reward")
                
                # RouterBench metrics
                routerbench = bandit_metrics.get("routerbench", {})
                result.cost_per_query = routerbench.get("avg_cost_per_query", 0.0)
                result.routerbench_efficiency = routerbench.get("efficiency_score", 0.0)
                
                # Component savings
                component_savings = response.get("component_savings", {})
                result.memory_savings = component_savings.get("memory_layer", 0)
                result.orchestrator_savings = component_savings.get("orchestrator", 0)
                result.bandit_savings = component_savings.get("bandit", 0)
                result.tokens_saved = component_savings.get("total_savings", 0)
                
                # Quality metrics (if quality judge ran)
                if "quality_result" in response:
                    qr = response["quality_result"]
                    result.quality_winner = qr.get("winner", "")
                    result.quality_confidence = qr.get("confidence", 0.0)
                    result.quality_explanation = qr.get("explanation", "")[:200]
                
                # Calculate savings percentage
                if result.baseline_total_tokens > 0:
                    result.savings_percentage = (result.tokens_saved / result.baseline_total_tokens) * 100
                elif result.total_tokens > 0 and result.tokens_saved > 0:
                    baseline_estimate = result.total_tokens + result.tokens_saved
                    result.savings_percentage = (result.tokens_saved / baseline_estimate) * 100
            else:
                result.response = str(response)[:500]
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            traceback.print_exc()
            
        return result
    
    def run_all_tests(self) -> bool:
        """Run all tests in the dataset."""
        dataset = self.create_test_dataset()
        
        print("\n" + "-" * 50)
        print("RUNNING TESTS")
        print("-" * 50)
        
        total = len(dataset)
        for i, query_data in enumerate(dataset):
            # Progress indicator
            progress = f"[{i+1}/{total}]"
            category = query_data['category'][:15]
            subcategory = query_data.get('subcategory', '')[:10]
            print(f"\r{progress} {category:<15} {subcategory:<10}", end="", flush=True)
            
            result = self.run_single_query(query_data, i + 1)
            self.results.append(result)
            
            # Brief pause to avoid rate limiting
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)
        
        print(f"\n\nCompleted {len(self.results)} tests")
        return True
    
    def aggregate_results(self):
        """Aggregate all results into comprehensive summary statistics."""
        print("\n" + "-" * 50)
        print("AGGREGATING RESULTS")
        print("-" * 50)
        
        self.summary.test_end = datetime.now()
        if self.summary.test_start:
            self.summary.duration_seconds = (self.summary.test_end - self.summary.test_start).total_seconds()
        
        self.summary.total_queries = len(self.results)
        self.summary.successful_queries = sum(1 for r in self.results if r.success)
        self.summary.failed_queries = sum(1 for r in self.results if not r.success)
        self.summary.success_rate = (self.summary.successful_queries / self.summary.total_queries * 100) if self.summary.total_queries > 0 else 0
        
        # Cache performance
        for r in self.results:
            if r.cache_type == "exact":
                self.summary.exact_cache_hits += 1
            elif r.cache_type == "semantic_direct":
                self.summary.semantic_direct_hits += 1
            elif r.cache_type == "context":
                self.summary.semantic_context_hits += 1
            else:
                self.summary.cache_misses += 1
        
        total_cache_attempts = self.summary.total_queries - sum(1 for r in self.results if not r.success)
        if total_cache_attempts > 0:
            self.summary.cache_hit_rate = ((self.summary.exact_cache_hits + self.summary.semantic_direct_hits) / total_cache_attempts) * 100
        
        # Token metrics
        self.summary.total_optimized_tokens = sum(r.total_tokens for r in self.results if r.success)
        self.summary.total_tokens_saved = sum(r.tokens_saved for r in self.results if r.success)
        
        # Estimate baseline tokens (optimized + saved)
        self.summary.total_baseline_tokens = self.summary.total_optimized_tokens + self.summary.total_tokens_saved
        
        if self.summary.total_baseline_tokens > 0:
            self.summary.average_savings_percentage = (self.summary.total_tokens_saved / self.summary.total_baseline_tokens) * 100
        
        # Latency metrics
        latencies = [r.latency_ms for r in self.results if r.success and r.latency_ms > 0]
        if latencies:
            self.summary.average_latency_ms = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            self.summary.p50_latency_ms = sorted_latencies[n // 2]
            self.summary.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0]
            self.summary.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0]
        
        # LLMLingua Compression metrics (CRITICAL FIX)
        self.summary.context_compressions = sum(1 for r in self.results if r.context_compressed)
        self.summary.query_compressions = sum(1 for r in self.results if r.query_compressed)
        self.summary.total_compressions = self.summary.context_compressions + self.summary.query_compressions
        
        context_ratios = [r.context_compression_ratio for r in self.results if r.context_compressed and r.context_compression_ratio < 1.0]
        if context_ratios:
            self.summary.average_context_compression_ratio = sum(context_ratios) / len(context_ratios)
        
        query_ratios = [r.query_compression_ratio for r in self.results if r.query_compressed and r.query_compression_ratio < 1.0]
        if query_ratios:
            self.summary.average_query_compression_ratio = sum(query_ratios) / len(query_ratios)
        
        self.summary.compression_tokens_saved = sum(r.total_compression_savings for r in self.results)
        
        # Orchestrator metrics
        for r in self.results:
            if r.complexity == "simple":
                self.summary.simple_queries += 1
            elif r.complexity == "medium":
                self.summary.medium_queries += 1
            elif r.complexity == "complex":
                self.summary.complex_queries += 1
        
        budgets = [r.token_budget for r in self.results if r.token_budget > 0]
        if budgets:
            self.summary.average_token_budget = sum(budgets) / len(budgets)
        
        efficiencies = [r.token_efficiency for r in self.results if r.token_efficiency > 0]
        if efficiencies:
            self.summary.average_token_efficiency = sum(efficiencies) / len(efficiencies)
        
        # Bandit + RouterBench metrics
        for r in self.results:
            if r.strategy_selected == "cheap":
                self.summary.cheap_strategy_uses += 1
            elif r.strategy_selected == "balanced":
                self.summary.balanced_strategy_uses += 1
            elif r.strategy_selected == "premium":
                self.summary.premium_strategy_uses += 1
        
        rewards = [r.strategy_reward for r in self.results if r.strategy_reward is not None]
        if rewards:
            self.summary.average_reward = sum(rewards) / len(rewards)
        
        costs = [r.cost_per_query for r in self.results if r.cost_per_query > 0]
        if costs:
            self.summary.total_cost = sum(costs)
            self.summary.average_cost_per_query = sum(costs) / len(costs)
        
        # Quality metrics
        for r in self.results:
            if r.quality_winner == "optimized":
                self.summary.optimized_wins += 1
            elif r.quality_winner == "baseline":
                self.summary.baseline_wins += 1
            elif r.quality_winner == "equivalent":
                self.summary.equivalent_results += 1
        
        confidences = [r.quality_confidence for r in self.results if r.quality_confidence > 0]
        if confidences:
            self.summary.average_quality_confidence = sum(confidences) / len(confidences)
        
        # Preference learning
        self.summary.preferences_detected = sum(1 for r in self.results if r.preference_used)
        self.summary.unique_tones = list(set(r.preference_tone for r in self.results if r.preference_tone))
        self.summary.unique_formats = list(set(r.preference_format for r in self.results if r.preference_format))
        
        # Category breakdown
        categories = set(r.category for r in self.results)
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            self.summary.category_results[cat] = {
                "total": len(cat_results),
                "successful": sum(1 for r in cat_results if r.success),
                "cache_hits": sum(1 for r in cat_results if r.cache_hit),
                "compressions": sum(1 for r in cat_results if r.context_compressed or r.query_compressed),
                "tokens_saved": sum(r.tokens_saved for r in cat_results),
                "compression_savings": sum(r.total_compression_savings for r in cat_results),
                "avg_latency_ms": sum(r.latency_ms for r in cat_results) / len(cat_results) if cat_results else 0,
            }
        
        print(f"Aggregation complete: {self.summary.successful_queries}/{self.summary.total_queries} successful")
        print(f"  - Cache hits: {self.summary.exact_cache_hits + self.summary.semantic_direct_hits}")
        print(f"  - LLMLingua compressions: {self.summary.total_compressions}")
        print(f"  - Total tokens saved: {self.summary.total_tokens_saved:,}")
    
    def display_results(self):
        """Display comprehensive results to console."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 70)
        
        # Overall Status
        print(f"\n{'='*30} OVERALL {'='*30}")
        print(f"Success Rate: {self.summary.success_rate:.1f}% ({self.summary.successful_queries}/{self.summary.total_queries})")
        print(f"Duration: {self.summary.duration_seconds:.1f} seconds")
        
        # Cache Performance
        print(f"\n{'='*25} CACHE PERFORMANCE {'='*25}")
        print(f"Exact Cache Hits:      {self.summary.exact_cache_hits}")
        print(f"Semantic Direct Hits:  {self.summary.semantic_direct_hits}")
        print(f"Context Injections:    {self.summary.semantic_context_hits}")
        print(f"Cache Misses:          {self.summary.cache_misses}")
        print(f"Cache Hit Rate:        {self.summary.cache_hit_rate:.1f}%")
        
        # Token Savings
        print(f"\n{'='*26} TOKEN SAVINGS {'='*26}")
        print(f"Total Optimized Tokens:  {self.summary.total_optimized_tokens:,}")
        print(f"Estimated Baseline:      {self.summary.total_baseline_tokens:,}")
        print(f"Tokens Saved:            {self.summary.total_tokens_saved:,}")
        print(f"Average Savings:         {self.summary.average_savings_percentage:.1f}%")
        
        # LLMLingua Compression
        print(f"\n{'='*23} LLMLINGUA COMPRESSION {'='*23}")
        print(f"Available:               {'Yes' if self.summary.llmlingua_available else 'No'}")
        if self.summary.llmlingua_available:
            print(f"Model:                   {self.summary.llmlingua_model}")
        print(f"Context Compressions:    {self.summary.context_compressions}")
        print(f"Query Compressions:      {self.summary.query_compressions}")
        print(f"Total Compressions:      {self.summary.total_compressions}")
        if self.summary.context_compressions > 0:
            print(f"Avg Context Ratio:       {self.summary.average_context_compression_ratio:.2%}")
        if self.summary.query_compressions > 0:
            print(f"Avg Query Ratio:         {self.summary.average_query_compression_ratio:.2%}")
        print(f"Compression Savings:     {self.summary.compression_tokens_saved:,} tokens")
        
        # Orchestrator
        print(f"\n{'='*28} ORCHESTRATOR {'='*28}")
        print(f"Simple Queries:   {self.summary.simple_queries}")
        print(f"Medium Queries:   {self.summary.medium_queries}")
        print(f"Complex Queries:  {self.summary.complex_queries}")
        print(f"Avg Token Budget: {self.summary.average_token_budget:.0f}")
        print(f"Avg Efficiency:   {self.summary.average_token_efficiency:.2f}")
        
        # Bandit + RouterBench
        print(f"\n{'='*24} BANDIT + ROUTERBENCH {'='*24}")
        print(f"Cheap Strategy:     {self.summary.cheap_strategy_uses}")
        print(f"Balanced Strategy:  {self.summary.balanced_strategy_uses}")
        print(f"Premium Strategy:   {self.summary.premium_strategy_uses}")
        print(f"Average Reward:     {self.summary.average_reward:.4f}")
        print(f"Total Cost:         ${self.summary.total_cost:.6f}")
        print(f"Avg Cost/Query:     ${self.summary.average_cost_per_query:.6f}")
        
        # Quality
        if self.summary.optimized_wins + self.summary.baseline_wins + self.summary.equivalent_results > 0:
            print(f"\n{'='*30} QUALITY {'='*30}")
            print(f"Optimized Wins:   {self.summary.optimized_wins}")
            print(f"Baseline Wins:    {self.summary.baseline_wins}")
            print(f"Equivalent:       {self.summary.equivalent_results}")
            print(f"Avg Confidence:   {self.summary.average_quality_confidence:.2f}")
        
        # Latency
        print(f"\n{'='*30} LATENCY {'='*30}")
        print(f"Average:  {self.summary.average_latency_ms:.0f}ms")
        print(f"P50:      {self.summary.p50_latency_ms:.0f}ms")
        print(f"P95:      {self.summary.p95_latency_ms:.0f}ms")
        print(f"P99:      {self.summary.p99_latency_ms:.0f}ms")
        
        # Category Breakdown
        print(f"\n{'='*25} RESULTS BY CATEGORY {'='*25}")
        for cat, stats in sorted(self.summary.category_results.items()):
            compressions = stats.get('compressions', 0)
            comp_str = f" [C:{compressions}]" if compressions > 0 else ""
            print(f"  {cat:<20}: {stats['successful']}/{stats['total']} ok, {stats['tokens_saved']:,} tokens saved{comp_str}")
    
    def save_reports(self):
        """Save comprehensive reports in multiple formats."""
        print("\n" + "-" * 50)
        print("GENERATING REPORTS")
        print("-" * 50)
        
        # Create output directory
        output_dir = "showcase_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON Report (complete data)
        json_path = os.path.join(output_dir, f"diagnostic_results_{timestamp}.json")
        report_data = {
            "test_info": {
                "name": "Comprehensive Diagnostic Test Suite",
                "start_time": self.summary.test_start.isoformat() if self.summary.test_start else None,
                "end_time": self.summary.test_end.isoformat() if self.summary.test_end else None,
                "duration_seconds": self.summary.duration_seconds,
                "total_queries": self.summary.total_queries,
            },
            "summary": asdict(self.summary),
            "component_health": [asdict(h) for h in self.component_health],
            "results": [asdict(r) for r in self.results],
        }
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, cls=CustomJSONEncoder, default=str)
        print(f"[OK] JSON report: {json_path}")
        
        # 2. CSV Report (query-by-query)
        csv_path = os.path.join(output_dir, f"diagnostic_results_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ID", "Category", "Subcategory", "Query", "Success", "Latency_ms",
                "Cache_Type", "Similarity", "Context_Compressed", "Query_Compressed",
                "Compression_Ratio", "Compression_Savings", "Complexity", "Strategy",
                "Tokens_Used", "Tokens_Saved", "Savings_%", "Reward", "Error"
            ])
            for r in self.results:
                compression_ratio = min(r.context_compression_ratio, r.query_compression_ratio) if (r.context_compressed or r.query_compressed) else 1.0
                writer.writerow([
                    r.query_id, r.category, r.subcategory, r.query[:80], r.success, f"{r.latency_ms:.0f}",
                    r.cache_type, f"{r.similarity:.3f}" if r.similarity else "",
                    r.context_compressed, r.query_compressed,
                    f"{compression_ratio:.2f}", r.total_compression_savings, r.complexity, r.strategy_selected,
                    r.total_tokens, r.tokens_saved, f"{r.savings_percentage:.1f}",
                    f"{r.strategy_reward:.4f}" if r.strategy_reward else "", r.error or ""
                ])
        print(f"[OK] CSV report: {csv_path}")
        
        # 3. Markdown Report (documentation)
        md_path = os.path.join(output_dir, f"diagnostic_results_{timestamp}.md")
        self._generate_markdown_report(md_path)
        print(f"[OK] Markdown report: {md_path}")
        
        # 4. HTML Report (interactive dashboard)
        html_path = os.path.join(output_dir, f"diagnostic_results_{timestamp}.html")
        self._generate_html_report(html_path)
        print(f"[OK] HTML report: {html_path}")
    
    def _generate_markdown_report(self, path: str):
        """Generate comprehensive Markdown documentation."""
        with open(path, "w") as f:
            f.write("# Tokenomics Platform - Comprehensive Diagnostic Test Results\n\n")
            f.write(f"**Test Date:** {self.summary.test_start.strftime('%Y-%m-%d %H:%M:%S') if self.summary.test_start else 'N/A'}\n")
            f.write(f"**Duration:** {self.summary.duration_seconds:.1f} seconds\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This diagnostic test validates all components of the Tokenomics platform using {self.summary.total_queries} carefully designed queries.\n\n")
            
            f.write("### Key Metrics\n\n")
            f.write("| Metric | Value | Status |\n")
            f.write("|--------|-------|--------|\n")
            f.write(f"| Success Rate | {self.summary.success_rate:.1f}% | {'✅' if self.summary.success_rate > 90 else '⚠️'} |\n")
            f.write(f"| Cache Hit Rate | {self.summary.cache_hit_rate:.1f}% | {'✅' if self.summary.cache_hit_rate > 0 else '⚠️'} |\n")
            f.write(f"| LLMLingua Compressions | {self.summary.total_compressions} | {'✅' if self.summary.total_compressions > 0 else '⚠️'} |\n")
            f.write(f"| Tokens Saved | {self.summary.total_tokens_saved:,} | {'✅' if self.summary.total_tokens_saved > 0 else '⚠️'} |\n")
            f.write(f"| Average Reward | {self.summary.average_reward:.4f} | {'✅' if self.summary.average_reward > 0.9 else '⚠️'} |\n\n")
            
            # Component Health
            f.write("## Component Health\n\n")
            f.write("| Component | Status | Details |\n")
            f.write("|-----------|--------|--------|\n")
            for h in self.component_health:
                status_icon = "✅" if h.available else "❌"
                details = ", ".join(f"{k}: {v}" for k, v in h.details.items()) if h.details else "-"
                f.write(f"| {h.name} | {status_icon} {h.status} | {details[:50]} |\n")
            f.write("\n")
            
            # Cache Performance
            f.write("## Cache Performance\n\n")
            f.write("The memory layer implements a tiered caching system:\n\n")
            f.write("| Cache Type | Count | Description |\n")
            f.write("|------------|-------|-------------|\n")
            f.write(f"| Exact Match | {self.summary.exact_cache_hits} | Identical query found in cache |\n")
            f.write(f"| Semantic Direct | {self.summary.semantic_direct_hits} | High similarity (>0.85) - direct return |\n")
            f.write(f"| Context Injection | {self.summary.semantic_context_hits} | Medium similarity (0.75-0.85) - context added |\n")
            f.write(f"| Cache Miss | {self.summary.cache_misses} | No match found - full LLM call |\n\n")
            
            # LLMLingua
            f.write("## LLMLingua Compression\n\n")
            if self.summary.llmlingua_available:
                f.write(f"**Status:** ✅ Active\n")
                f.write(f"**Model:** {self.summary.llmlingua_model}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Context Compressions | {self.summary.context_compressions} |\n")
                f.write(f"| Query Compressions | {self.summary.query_compressions} |\n")
                f.write(f"| Total Compressions | {self.summary.total_compressions} |\n")
                f.write(f"| Avg Context Ratio | {self.summary.average_context_compression_ratio:.2%} |\n")
                f.write(f"| Avg Query Ratio | {self.summary.average_query_compression_ratio:.2%} |\n")
                f.write(f"| Tokens Saved | {self.summary.compression_tokens_saved:,} |\n\n")
            else:
                f.write("**Status:** ❌ Not available (using fallback compression)\n\n")
            
            # Bandit + RouterBench
            f.write("## Bandit Optimizer + RouterBench\n\n")
            f.write("Strategy selection based on query complexity and cost-quality routing:\n\n")
            f.write("| Strategy | Uses | Description |\n")
            f.write("|----------|------|-------------|\n")
            f.write(f"| Cheap | {self.summary.cheap_strategy_uses} | gpt-4o-mini, low cost, fast |\n")
            f.write(f"| Balanced | {self.summary.balanced_strategy_uses} | gpt-4o-mini, balanced settings |\n")
            f.write(f"| Premium | {self.summary.premium_strategy_uses} | gpt-4o, high quality |\n\n")
            f.write(f"**Average Reward:** {self.summary.average_reward:.4f}\n")
            f.write(f"**Total Cost:** ${self.summary.total_cost:.6f}\n\n")
            
            # Results by Category
            f.write("## Results by Category\n\n")
            f.write("| Category | Success | Cache Hits | Compressions | Tokens Saved |\n")
            f.write("|----------|---------|------------|--------------|-------------|\n")
            for cat, stats in sorted(self.summary.category_results.items()):
                f.write(f"| {cat} | {stats['successful']}/{stats['total']} | {stats['cache_hits']} | {stats['compressions']} | {stats['tokens_saved']:,} |\n")
            f.write("\n")
            
            # Detailed Query Results
            f.write("## Detailed Query Results\n\n")
            f.write("<details>\n<summary>Click to expand all queries</summary>\n\n")
            for r in self.results:
                status = "✅" if r.success else "❌"
                f.write(f"### Query {r.query_id}: {r.category}/{r.subcategory}\n")
                f.write(f"- **Query:** `{r.query[:100]}...`\n")
                f.write(f"- **Status:** {status}\n")
                f.write(f"- **Cache:** {r.cache_type}")
                if r.similarity:
                    f.write(f" (similarity: {r.similarity:.3f})")
                f.write("\n")
                if r.context_compressed or r.query_compressed:
                    f.write(f"- **Compression:** Context={r.context_compressed}, Query={r.query_compressed}, Savings={r.total_compression_savings}\n")
                reward_str = f"{r.strategy_reward:.4f}" if r.strategy_reward is not None else "N/A"
                f.write(f"- **Strategy:** {r.strategy_selected}, Reward={reward_str}\n")
                f.write(f"- **Tokens:** {r.total_tokens:,} (saved: {r.tokens_saved:,})\n\n")
            f.write("</details>\n")
    
    def _generate_html_report(self, path: str):
        """Generate interactive HTML dashboard with charts."""
        
        # Prepare chart data
        cache_data = [
            self.summary.exact_cache_hits,
            self.summary.semantic_direct_hits,
            self.summary.semantic_context_hits,
            self.summary.cache_misses
        ]
        
        strategy_data = [
            self.summary.cheap_strategy_uses,
            self.summary.balanced_strategy_uses,
            self.summary.premium_strategy_uses
        ]
        
        complexity_data = [
            self.summary.simple_queries,
            self.summary.medium_queries,
            self.summary.complex_queries
        ]
        
        with open(path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tokenomics Diagnostic Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --accent-cyan: #00d4ff;
            --accent-green: #00ff88;
            --accent-purple: #a855f7;
            --accent-orange: #ff6b35;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --border: #2a2a4a;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid var(--accent-cyan);
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle { color: var(--text-secondary); font-size: 1.1rem; }
        
        .grid { display: grid; gap: 20px; margin-bottom: 30px; }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .grid-2 { grid-template-columns: repeat(2, 1fr); }
        
        @media (max-width: 1200px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; } }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        
        .stat-card {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--accent-cyan);
            margin-bottom: 8px;
        }
        
        .stat-value.green { color: var(--accent-green); }
        .stat-value.purple { color: var(--accent-purple); }
        .stat-value.orange { color: var(--accent-orange); }
        
        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        h2 {
            color: var(--accent-cyan);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: var(--bg-secondary);
            color: var(--accent-cyan);
            font-weight: 600;
        }
        
        tr:hover { background: var(--bg-secondary); }
        
        .status-ok { color: var(--accent-green); }
        .status-warn { color: var(--accent-orange); }
        .status-fail { color: #ff4757; }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .badge-green { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); }
        .badge-cyan { background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); }
        .badge-purple { background: rgba(168, 85, 247, 0.2); color: var(--accent-purple); }
        .badge-orange { background: rgba(255, 107, 53, 0.2); color: var(--accent-orange); }
        
        .component-list { list-style: none; }
        .component-list li {
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
        }
        
        .llmlingua-highlight {
            background: linear-gradient(135deg, var(--bg-card), rgba(0, 212, 255, 0.1));
            border: 2px solid var(--accent-cyan);
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Tokenomics Diagnostic Results</h1>
            <p class="subtitle">Comprehensive Platform Test Suite</p>
""")
            f.write(f'            <p class="subtitle">Test Date: {self.summary.test_start.strftime("%Y-%m-%d %H:%M:%S") if self.summary.test_start else "N/A"} | Duration: {self.summary.duration_seconds:.1f}s</p>\n')
            f.write("""        </header>
        
        <!-- Summary Stats -->
        <div class="grid grid-4">
            <div class="card stat-card">
""")
            f.write(f'                <div class="stat-value">{self.summary.success_rate:.1f}%</div>\n')
            f.write(f'                <div class="stat-label">Success Rate ({self.summary.successful_queries}/{self.summary.total_queries})</div>\n')
            f.write("""            </div>
            <div class="card stat-card">
""")
            f.write(f'                <div class="stat-value green">{self.summary.total_tokens_saved:,}</div>\n')
            f.write("""                <div class="stat-label">Tokens Saved</div>
            </div>
            <div class="card stat-card">
""")
            f.write(f'                <div class="stat-value purple">{self.summary.total_compressions}</div>\n')
            f.write("""                <div class="stat-label">LLMLingua Compressions</div>
            </div>
            <div class="card stat-card">
""")
            f.write(f'                <div class="stat-value orange">{self.summary.average_reward:.4f}</div>\n')
            f.write("""                <div class="stat-label">Avg Bandit Reward</div>
            </div>
        </div>
        
        <!-- LLMLingua Section -->
        <div class="card llmlingua-highlight">
            <h2>LLMLingua Compression</h2>
            <div class="grid grid-4">
                <div class="stat-card">
""")
            f.write(f'                    <div class="stat-value">{"✅" if self.summary.llmlingua_available else "❌"}</div>\n')
            f.write("""                    <div class="stat-label">Status</div>
                </div>
                <div class="stat-card">
""")
            f.write(f'                    <div class="stat-value">{self.summary.context_compressions}</div>\n')
            f.write("""                    <div class="stat-label">Context Compressions</div>
                </div>
                <div class="stat-card">
""")
            f.write(f'                    <div class="stat-value">{self.summary.query_compressions}</div>\n')
            f.write("""                    <div class="stat-label">Query Compressions</div>
                </div>
                <div class="stat-card">
""")
            f.write(f'                    <div class="stat-value">{self.summary.compression_tokens_saved:,}</div>\n')
            f.write("""                    <div class="stat-label">Tokens Saved</div>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="grid grid-3" style="margin-top: 30px;">
            <div class="card">
                <h2>Cache Performance</h2>
                <div class="chart-container">
                    <canvas id="cacheChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Strategy Distribution</h2>
                <div class="chart-container">
                    <canvas id="strategyChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Query Complexity</h2>
                <div class="chart-container">
                    <canvas id="complexityChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Component Health -->
        <div class="card" style="margin-top: 30px;">
            <h2>Component Health</h2>
            <ul class="component-list">
""")
            for h in self.component_health:
                status_class = "status-ok" if h.available else "status-fail"
                details = ", ".join(f"{k}: {v}" for k, v in list(h.details.items())[:2]) if h.details else ""
                f.write(f'                <li><span>{h.name}</span><span class="{status_class}">{h.status}</span></li>\n')
            
            f.write("""            </ul>
        </div>
        
        <!-- Results Table -->
        <div class="card" style="margin-top: 30px;">
            <h2>Query Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Category</th>
                        <th>Cache</th>
                        <th>Compressed</th>
                        <th>Strategy</th>
                        <th>Tokens Saved</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
""")
            for r in self.results:
                status = "status-ok" if r.success else "status-fail"
                status_text = "OK" if r.success else "FAIL"
                compressed = "Yes" if (r.context_compressed or r.query_compressed) else "No"
                comp_class = "badge-cyan" if compressed == "Yes" else ""
                f.write(f'                    <tr>\n')
                f.write(f'                        <td>{r.query_id}</td>\n')
                f.write(f'                        <td>{r.category}</td>\n')
                f.write(f'                        <td><span class="badge badge-purple">{r.cache_type}</span></td>\n')
                f.write(f'                        <td><span class="badge {comp_class}">{compressed}</span></td>\n')
                f.write(f'                        <td>{r.strategy_selected or "-"}</td>\n')
                f.write(f'                        <td>{r.tokens_saved:,}</td>\n')
                f.write(f'                        <td class="{status}">{status_text}</td>\n')
                f.write(f'                    </tr>\n')
            
            f.write("""                </tbody>
            </table>
        </div>
        
        <footer>
            <p>Generated by Tokenomics Platform Diagnostic Suite</p>
        </footer>
    </div>
    
    <script>
        // Cache Chart
        new Chart(document.getElementById('cacheChart'), {
            type: 'doughnut',
            data: {
                labels: ['Exact Hit', 'Semantic Direct', 'Context Injection', 'Miss'],
""")
            f.write(f'                datasets: [{{ data: {cache_data}, backgroundColor: ["#00ff88", "#00d4ff", "#a855f7", "#ff4757"] }}]\n')
            f.write("""            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } }
            }
        });
        
        // Strategy Chart
        new Chart(document.getElementById('strategyChart'), {
            type: 'bar',
            data: {
                labels: ['Cheap', 'Balanced', 'Premium'],
""")
            f.write(f'                datasets: [{{ label: "Uses", data: {strategy_data}, backgroundColor: ["#00ff88", "#00d4ff", "#a855f7"] }}]\n')
            f.write("""            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { ticks: { color: '#fff' } }, x: { ticks: { color: '#fff' } } }
            }
        });
        
        // Complexity Chart
        new Chart(document.getElementById('complexityChart'), {
            type: 'pie',
            data: {
                labels: ['Simple', 'Medium', 'Complex'],
""")
            f.write(f'                datasets: [{{ data: {complexity_data}, backgroundColor: ["#00ff88", "#00d4ff", "#a855f7"] }}]\n')
            f.write("""            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } }
            }
        });
    </script>
</body>
</html>
""")
    
    def run(self) -> bool:
        """Run the complete diagnostic test suite."""
        try:
            # Initialize
            if not self.initialize_platform():
                return False
            
            # Health check
            self.check_component_health()
            
            # Run tests
            self.run_all_tests()
            
            # Aggregate and display
            self.aggregate_results()
            self.display_results()
            
            # Save reports
            self.save_reports()
            
            print("\n" + "=" * 70)
            print("DIAGNOSTIC TEST COMPLETE")
            print("=" * 70)
            
            return self.summary.successful_queries > 0
            
        except Exception as e:
            print(f"\n[FAIL] Test suite error: {e}")
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    test = ComprehensiveShowcaseTest()
    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
