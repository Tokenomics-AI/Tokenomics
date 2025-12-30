"""
Extensive Diagnostic Test for Tokenomics Platform - First Version Presentation

This test runs a carefully designed phased query dataset that triggers EVERY
platform component and documents genuine results including:

- Memory Layer: Exact cache, semantic cache (direct + context injection)
- LLMLingua Compression: Query and context compression
- Token Orchestrator: Complexity analysis, token allocation
- Bandit Optimizer: Strategy selection, RouterBench routing
- Quality Judge: Baseline vs optimized comparison

Each phase builds on the previous to demonstrate platform capabilities.
Results are documented honestly - issues are flagged for fixing.
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import structlog

# Suppress warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

logger = structlog.get_logger()


@dataclass
class QueryTestCase:
    """A single test query with expected behavior."""
    id: str
    phase: str
    query: str
    description: str
    expected_cache_type: Optional[str] = None  # exact, semantic_direct, context, None
    expected_complexity: Optional[str] = None  # simple, medium, complex
    expected_strategy: Optional[str] = None  # cheap, balanced, premium
    expected_tone: Optional[str] = None  # formal, casual, technical, simple
    expected_format: Optional[str] = None  # list, paragraph, code, concise
    min_similarity: Optional[float] = None
    max_similarity: Optional[float] = None
    expect_compression: bool = False
    token_budget: Optional[int] = None


@dataclass
class QueryResult:
    """Result from running a single query."""
    test_id: str
    phase: str
    query: str
    description: str
    
    # Actual results
    cache_hit: bool = False
    cache_type: Optional[str] = None
    similarity: Optional[float] = None
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    strategy: Optional[str] = None
    model: Optional[str] = None
    complexity: Optional[str] = None
    response_length: int = 0
    
    # Memory metrics
    memory_exact_hits: int = 0
    memory_semantic_direct_hits: int = 0
    memory_semantic_context_hits: int = 0
    memory_tokens_saved: int = 0
    
    # Compression metrics
    query_compressed: bool = False
    context_compressed: bool = False
    compression_savings: int = 0
    
    # Preference metrics
    detected_tone: Optional[str] = None
    detected_format: Optional[str] = None
    preference_confidence: float = 0.0
    
    # Orchestrator metrics
    token_budget: int = 0
    max_response_tokens: int = 0
    
    # Bandit metrics
    reward: Optional[float] = None
    
    # Quality judge metrics
    quality_winner: Optional[str] = None
    quality_confidence: Optional[float] = None
    baseline_tokens: int = 0
    
    # Component savings
    memory_savings: int = 0
    orchestrator_savings: int = 0
    bandit_savings: int = 0
    total_savings: int = 0
    
    # Validation
    validations: Dict = field(default_factory=dict)
    passed: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class ComponentSummary:
    """Summary statistics for a component."""
    name: str
    queries_processed: int = 0
    total_tokens_saved: int = 0
    success_rate: float = 0.0
    metrics: Dict = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class ExtensiveDiagnosticTest:
    """Extensive diagnostic test suite for Tokenomics Platform."""
    
    def __init__(self):
        """Initialize the test suite."""
        # Load config
        self.config = TokenomicsConfig.from_env()
        self.config.llm.provider = "openai"
        self.config.llm.model = "gpt-4o-mini"
        self.config.llm.api_key = os.getenv("OPENAI_API_KEY")
        
        # Enable all features
        self.config.memory.use_exact_cache = True
        self.config.memory.use_semantic_cache = True
        self.config.memory.enable_llmlingua = True
        self.config.memory.similarity_threshold = 0.70  # For context injection
        self.config.memory.direct_return_threshold = 0.85  # For semantic direct hits
        
        # Initialize platform
        print("Initializing Tokenomics Platform...")
        self.platform = TokenomicsPlatform(config=self.config)
        
        # Test results
        self.results: List[QueryResult] = []
        self.component_summaries: Dict[str, ComponentSummary] = {}
        self.issues_found: List[Dict] = []
        
        # Create test dataset
        self.test_cases = self._create_test_dataset()
        
        print(f"Initialized with {len(self.test_cases)} test cases across {len(set(tc.phase for tc in self.test_cases))} phases")
    
    def _create_test_dataset(self) -> List[QueryTestCase]:
        """Create the phased test dataset."""
        
        # Long query for compression testing (>800 chars, >200 tokens)
        long_ml_query = """I need a comprehensive and detailed explanation of how modern machine learning 
        algorithms work in production systems. Please cover the following topics in depth: 
        1) Neural network architectures including CNNs, RNNs, LSTMs, and Transformers
        2) The backpropagation algorithm and how gradients flow through the network
        3) Optimization techniques like SGD, Adam, and learning rate scheduling
        4) Regularization methods including dropout, batch normalization, and L2 regularization
        5) How these models are deployed in production with considerations for latency and throughput
        6) Best practices for model monitoring and retraining in production environments
        Please provide specific examples and code snippets where appropriate."""
        
        complex_architecture_query = """Design a complete, production-ready microservices architecture 
        for an enterprise e-commerce platform that needs to handle millions of daily transactions. 
        The system must include: user authentication and authorization with OAuth2 and JWT, 
        product catalog management with search and filtering, shopping cart with session persistence, 
        payment processing with PCI compliance, order management and fulfillment tracking, 
        inventory management with real-time updates, recommendation engine using collaborative filtering,
        analytics and reporting dashboard, notification system for email and push notifications,
        and a customer service chat system. Include considerations for scalability, fault tolerance,
        caching strategies, database sharding, and monitoring. Provide architecture diagrams and 
        technology stack recommendations."""
        
        return [
            # =====================================================================
            # PHASE 1: CACHE SEEDING (Populate cache for later phases)
            # =====================================================================
            QueryTestCase(
                id="seed_python",
                phase="1_cache_seeding",
                query="What is Python?",
                description="Seed query for Python - will be used for exact and semantic cache tests",
                expected_cache_type=None,  # First call - no cache
                expected_complexity="simple",
            ),
            QueryTestCase(
                id="seed_ml",
                phase="1_cache_seeding",
                query="Explain machine learning in simple terms",
                description="Seed query for ML - will be used for semantic matching",
                expected_cache_type=None,
                expected_complexity="simple",
            ),
            QueryTestCase(
                id="seed_rest",
                phase="1_cache_seeding",
                query="How do REST APIs work?",
                description="Seed query for REST APIs - will be used for context injection",
                expected_cache_type=None,
                expected_complexity="medium",
            ),
            QueryTestCase(
                id="seed_database",
                phase="1_cache_seeding",
                query="What is a relational database?",
                description="Seed query for databases",
                expected_cache_type=None,
                expected_complexity="simple",
            ),
            
            # =====================================================================
            # PHASE 2: EXACT CACHE HITS
            # =====================================================================
            QueryTestCase(
                id="exact_python_1",
                phase="2_exact_cache",
                query="What is Python?",
                description="EXACT same query as seed - must hit exact cache with 0 tokens",
                expected_cache_type="exact",
                expected_complexity="simple",
            ),
            QueryTestCase(
                id="exact_python_2",
                phase="2_exact_cache",
                query="What is Python?",
                description="Third identical query - validates cache consistency",
                expected_cache_type="exact",
            ),
            QueryTestCase(
                id="exact_rest",
                phase="2_exact_cache",
                query="How do REST APIs work?",
                description="Exact match for REST query",
                expected_cache_type="exact",
            ),
            
            # =====================================================================
            # PHASE 3: SEMANTIC DIRECT HITS (similarity > 0.85)
            # =====================================================================
            QueryTestCase(
                id="semantic_direct_python_1",
                phase="3_semantic_direct",
                query="What is the Python programming language?",
                description="Very similar to 'What is Python?' - should return cached response directly",
                expected_cache_type="semantic_direct",
                min_similarity=0.85,
            ),
            QueryTestCase(
                id="semantic_direct_python_2",
                phase="3_semantic_direct",
                query="Tell me about Python",
                description="Another high similarity Python query",
                expected_cache_type="semantic_direct",
                min_similarity=0.85,
            ),
            QueryTestCase(
                id="semantic_direct_ml",
                phase="3_semantic_direct",
                query="Explain what machine learning is",
                description="High similarity to ML seed query",
                expected_cache_type="semantic_direct",
                min_similarity=0.85,
            ),
            
            # =====================================================================
            # PHASE 4: CONTEXT INJECTION (similarity 0.70-0.85)
            # =====================================================================
            QueryTestCase(
                id="context_python_packages",
                phase="4_context_injection",
                query="How do I install Python packages using pip?",
                description="Related to Python but different enough - should inject context",
                expected_cache_type="context",
                min_similarity=0.70,
                max_similarity=0.85,
            ),
            QueryTestCase(
                id="context_rest_benefits",
                phase="4_context_injection",
                query="What are the main benefits and use cases of REST APIs?",
                description="Related to REST but asks different question - context injection",
                expected_cache_type="context",
                min_similarity=0.70,
                max_similarity=0.85,
            ),
            QueryTestCase(
                id="context_ml_vs_dl",
                phase="4_context_injection",
                query="What is the difference between machine learning and deep learning?",
                description="Related to ML seed - should use context",
                expected_cache_type="context",
                min_similarity=0.70,
                max_similarity=0.85,
            ),
            
            # =====================================================================
            # PHASE 5: LLMLINGUA COMPRESSION
            # =====================================================================
            QueryTestCase(
                id="compress_long_query",
                phase="5_compression",
                query=long_ml_query,
                description="Long query (>800 chars) - should trigger LLMLingua query compression",
                expect_compression=True,
                expected_complexity="complex",
            ),
            QueryTestCase(
                id="compress_follow_up",
                phase="5_compression",
                query="Summarize the key machine learning concepts",
                description="Follow-up that may use compressed context from cache",
                expected_cache_type="context",
            ),
            
            # =====================================================================
            # PHASE 6: USER PREFERENCE LEARNING
            # =====================================================================
            QueryTestCase(
                id="pref_formal",
                phase="6_preferences",
                query="Could you please kindly explain what Docker containers are?",
                description="Formal tone indicators: 'please', 'kindly'",
                expected_tone="formal",
            ),
            QueryTestCase(
                id="pref_casual",
                phase="6_preferences",
                query="Hey, what's the deal with Docker?",
                description="Casual tone indicators: 'hey', 'what's'",
                expected_tone="casual",
            ),
            QueryTestCase(
                id="pref_technical",
                phase="6_preferences",
                query="Explain the Docker container runtime architecture and implementation",
                description="Technical tone indicators: 'architecture', 'implementation'",
                expected_tone="technical",
            ),
            QueryTestCase(
                id="pref_list",
                phase="6_preferences",
                query="List the steps to deploy a Docker container",
                description="List format indicator: 'list the steps'",
                expected_format="list",
            ),
            QueryTestCase(
                id="pref_code",
                phase="6_preferences",
                query="Show me code example for Docker compose",
                description="Code format indicator: 'code', 'example'",
                expected_format="code",
            ),
            
            # =====================================================================
            # PHASE 7: COMPLEXITY & STRATEGY SELECTION
            # =====================================================================
            QueryTestCase(
                id="complexity_simple_1",
                phase="7_complexity",
                query="What is JSON?",
                description="Simple query - should use 'cheap' strategy",
                expected_complexity="simple",
                expected_strategy="cheap",
            ),
            QueryTestCase(
                id="complexity_simple_2",
                phase="7_complexity",
                query="Define API",
                description="Another simple query",
                expected_complexity="simple",
                expected_strategy="cheap",
            ),
            QueryTestCase(
                id="complexity_medium_1",
                phase="7_complexity",
                query="How does JSON parsing work in Python with error handling?",
                description="Medium complexity - should use 'balanced' strategy",
                expected_complexity="medium",
                expected_strategy="balanced",
            ),
            QueryTestCase(
                id="complexity_medium_2",
                phase="7_complexity",
                query="Explain the difference between GET and POST HTTP methods",
                description="Another medium complexity query",
                expected_complexity="medium",
            ),
            QueryTestCase(
                id="complexity_complex_1",
                phase="7_complexity",
                query=complex_architecture_query,
                description="Complex query - should use 'premium' strategy with gpt-4o",
                expected_complexity="complex",
                expected_strategy="premium",
            ),
            
            # =====================================================================
            # PHASE 8: QUALITY JUDGE VALIDATION
            # =====================================================================
            QueryTestCase(
                id="quality_test_1",
                phase="8_quality_judge",
                query="Explain the concept of recursion in programming",
                description="Test quality judge - compare baseline vs optimized",
            ),
            QueryTestCase(
                id="quality_test_2",
                phase="8_quality_judge",
                query="What are design patterns in software engineering?",
                description="Another quality comparison test",
            ),
            QueryTestCase(
                id="quality_test_3",
                phase="8_quality_judge",
                query="How does garbage collection work?",
                description="Third quality test for statistics",
            ),
            
            # =====================================================================
            # PHASE 9: TOKEN BUDGET SCENARIOS
            # =====================================================================
            QueryTestCase(
                id="budget_low",
                phase="9_token_budget",
                query="What is SQL injection?",
                description="Test with low token budget (1000)",
                token_budget=1000,
            ),
            QueryTestCase(
                id="budget_high",
                phase="9_token_budget",
                query="Explain SQL injection prevention techniques in detail",
                description="Test with high token budget (4000)",
                token_budget=4000,
            ),
            
            # =====================================================================
            # PHASE 10: FINAL CACHE UTILIZATION
            # =====================================================================
            QueryTestCase(
                id="final_exact_1",
                phase="10_final_validation",
                query="What is Python?",
                description="Final exact cache validation",
                expected_cache_type="exact",
            ),
            QueryTestCase(
                id="final_semantic_1",
                phase="10_final_validation",
                query="Explain Python to me",
                description="Final semantic validation",
                expected_cache_type="semantic_direct",
                min_similarity=0.85,
            ),
        ]
    
    def run_query(self, test_case: QueryTestCase) -> QueryResult:
        """Run a single query and collect all metrics."""
        result = QueryResult(
            test_id=test_case.id,
            phase=test_case.phase,
            query=test_case.query[:200],  # Truncate for storage
            description=test_case.description,
        )
        
        try:
            # Run the query through the platform
            start_time = time.time()
            response = self.platform.query(
                query=test_case.query,
                token_budget=test_case.token_budget,
                use_cache=True,
                use_bandit=True,
                use_compression=True,
                use_cost_aware_routing=True,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Extract basic metrics
            result.cache_hit = response.get("cache_hit", False)
            result.cache_type = response.get("cache_type")
            result.similarity = response.get("similarity")
            result.tokens_used = response.get("tokens_used", 0)
            result.input_tokens = response.get("input_tokens", 0)
            result.output_tokens = response.get("output_tokens", 0)
            result.latency_ms = response.get("latency_ms", elapsed_ms)
            result.strategy = response.get("strategy")
            result.model = response.get("model")
            result.complexity = response.get("query_type")
            result.response_length = len(response.get("response", ""))
            
            # Memory metrics
            memory_metrics = response.get("memory_metrics", {})
            result.memory_exact_hits = memory_metrics.get("exact_cache_hits", 0)
            result.memory_semantic_direct_hits = memory_metrics.get("semantic_direct_hits", 0)
            result.memory_semantic_context_hits = memory_metrics.get("semantic_context_hits", 0)
            
            # Compression metrics
            compression = response.get("compression_metrics", {})
            result.query_compressed = compression.get("query_compressed", False)
            result.context_compressed = compression.get("context_compressed", False)
            result.compression_savings = compression.get("total_compression_savings", 0)
            
            # Preference metrics
            pref_context = response.get("preference_context", {})
            result.detected_tone = pref_context.get("tone")
            result.detected_format = pref_context.get("format")
            result.preference_confidence = memory_metrics.get("preference_confidence", 0)
            
            # Orchestrator metrics
            orchestrator = response.get("orchestrator_metrics", {})
            result.token_budget = orchestrator.get("token_budget", 0)
            result.max_response_tokens = response.get("max_response_tokens", 0)
            
            # Bandit metrics
            result.reward = response.get("reward")
            
            # Quality judge
            quality = response.get("quality_judge", {})
            if quality:
                result.quality_winner = quality.get("winner")
                result.quality_confidence = quality.get("confidence")
            
            # Baseline comparison
            baseline = response.get("baseline_comparison_result", {})
            if baseline:
                result.baseline_tokens = baseline.get("tokens_used", 0)
            
            # Component savings
            savings = response.get("component_savings", {})
            result.memory_savings = savings.get("memory_layer", 0)
            result.orchestrator_savings = savings.get("orchestrator", 0)
            result.bandit_savings = savings.get("bandit", 0)
            result.total_savings = savings.get("total_savings", 0)
            
            # Validate expectations
            result.validations, result.issues = self._validate_result(test_case, result)
            result.passed = len(result.issues) == 0
            
        except Exception as e:
            result.issues.append(f"Query failed with error: {str(e)}")
            result.passed = False
            logger.error("Query failed", test_id=test_case.id, error=str(e))
        
        return result
    
    def _validate_result(self, test_case: QueryTestCase, result: QueryResult) -> Tuple[Dict, List[str]]:
        """Validate result against expectations."""
        validations = {}
        issues = []
        
        # Validate cache type
        if test_case.expected_cache_type is not None:
            expected = test_case.expected_cache_type
            actual = result.cache_type
            passed = (expected == actual) or (expected == "exact" and result.cache_hit and actual == "exact")
            
            # For semantic_direct, check if it's a semantic hit
            if expected == "semantic_direct" and actual == "semantic_direct":
                passed = True
            elif expected == "context" and actual == "context":
                passed = True
            elif expected is None and not result.cache_hit:
                passed = True
            
            validations["cache_type"] = {
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
            
            if not passed:
                issues.append(f"Cache type mismatch: expected '{expected}', got '{actual}'")
        
        # Validate complexity
        if test_case.expected_complexity is not None:
            expected = test_case.expected_complexity
            actual = result.complexity
            passed = expected == actual
            validations["complexity"] = {
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Complexity mismatch: expected '{expected}', got '{actual}'")
        
        # Validate strategy
        if test_case.expected_strategy is not None:
            expected = test_case.expected_strategy
            actual = result.strategy
            passed = expected == actual
            validations["strategy"] = {
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Strategy mismatch: expected '{expected}', got '{actual}'")
        
        # Validate similarity range
        if test_case.min_similarity is not None and result.similarity is not None:
            passed = result.similarity >= test_case.min_similarity
            validations["min_similarity"] = {
                "expected": f">= {test_case.min_similarity}",
                "actual": result.similarity,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Similarity too low: expected >= {test_case.min_similarity}, got {result.similarity:.3f}")
        
        if test_case.max_similarity is not None and result.similarity is not None:
            passed = result.similarity <= test_case.max_similarity
            validations["max_similarity"] = {
                "expected": f"<= {test_case.max_similarity}",
                "actual": result.similarity,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Similarity too high: expected <= {test_case.max_similarity}, got {result.similarity:.3f}")
        
        # Validate compression
        if test_case.expect_compression:
            passed = result.query_compressed or result.context_compressed
            validations["compression"] = {
                "expected": "query or context compressed",
                "actual": f"query={result.query_compressed}, context={result.context_compressed}",
                "passed": passed,
            }
            if not passed:
                issues.append("Expected compression but none occurred")
        
        # Validate tone detection
        if test_case.expected_tone is not None:
            passed = result.detected_tone == test_case.expected_tone
            validations["tone"] = {
                "expected": test_case.expected_tone,
                "actual": result.detected_tone,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Tone mismatch: expected '{test_case.expected_tone}', got '{result.detected_tone}'")
        
        # Validate format detection
        if test_case.expected_format is not None:
            passed = result.detected_format == test_case.expected_format
            validations["format"] = {
                "expected": test_case.expected_format,
                "actual": result.detected_format,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Format mismatch: expected '{test_case.expected_format}', got '{result.detected_format}'")
        
        # Validate exact cache hits use 0 tokens
        if test_case.expected_cache_type == "exact" and result.cache_type == "exact":
            passed = result.tokens_used == 0
            validations["zero_tokens"] = {
                "expected": 0,
                "actual": result.tokens_used,
                "passed": passed,
            }
            if not passed:
                issues.append(f"Exact cache hit should use 0 tokens, got {result.tokens_used}")
        
        return validations, issues
    
    def run_all_tests(self) -> Dict:
        """Run all test phases and collect results."""
        print("\n" + "=" * 80)
        print("EXTENSIVE DIAGNOSTIC TEST - TOKENOMICS PLATFORM")
        print("=" * 80)
        
        start_time = time.time()
        
        # Group tests by phase
        phases = {}
        for tc in self.test_cases:
            if tc.phase not in phases:
                phases[tc.phase] = []
            phases[tc.phase].append(tc)
        
        # Run each phase in order
        for phase_name in sorted(phases.keys()):
            phase_tests = phases[phase_name]
            print(f"\n--- Phase: {phase_name} ({len(phase_tests)} queries) ---")
            
            for i, test_case in enumerate(phase_tests, 1):
                print(f"  [{i}/{len(phase_tests)}] {test_case.id}: {test_case.description[:50]}...")
                result = self.run_query(test_case)
                self.results.append(result)
                
                # Print immediate feedback
                status = "PASS" if result.passed else "FAIL"
                cache_info = f"cache={result.cache_type}" if result.cache_hit else "no_cache"
                tokens_info = f"tokens={result.tokens_used}"
                print(f"         [{status}] {cache_info}, {tokens_info}, strategy={result.strategy}")
                
                if result.issues:
                    for issue in result.issues:
                        print(f"         [ISSUE] {issue}")
                        self.issues_found.append({
                            "test_id": test_case.id,
                            "phase": phase_name,
                            "issue": issue,
                        })
        
        elapsed_time = time.time() - start_time
        
        # Calculate component summaries
        self._calculate_summaries()
        
        print(f"\n{'=' * 80}")
        print(f"Test completed in {elapsed_time:.1f} seconds")
        print(f"{'=' * 80}")
        
        return self._build_report(elapsed_time)
    
    def _calculate_summaries(self):
        """Calculate component-level summaries."""
        
        # Memory Layer Summary
        memory_summary = ComponentSummary(name="Memory Layer")
        exact_hits = sum(1 for r in self.results if r.cache_type == "exact")
        semantic_direct_hits = sum(1 for r in self.results if r.cache_type == "semantic_direct")
        context_hits = sum(1 for r in self.results if r.cache_type == "context")
        total_queries = len(self.results)
        cache_hits = exact_hits + semantic_direct_hits + context_hits
        
        memory_summary.queries_processed = total_queries
        memory_summary.total_tokens_saved = sum(r.memory_savings for r in self.results)
        memory_summary.success_rate = cache_hits / total_queries if total_queries > 0 else 0
        memory_summary.metrics = {
            "exact_cache_hits": exact_hits,
            "semantic_direct_hits": semantic_direct_hits,
            "semantic_context_hits": context_hits,
            "total_cache_hits": cache_hits,
            "cache_hit_rate": f"{cache_hits / total_queries * 100:.1f}%" if total_queries > 0 else "0%",
            "tokens_saved_from_cache": sum(r.memory_savings for r in self.results if r.cache_hit),
        }
        
        # Find issues with memory layer
        memory_issues = [r for r in self.results if r.validations.get("cache_type", {}).get("passed") == False]
        if memory_issues:
            memory_summary.issues = [f"Cache type validation failed for {len(memory_issues)} queries"]
        
        self.component_summaries["memory_layer"] = memory_summary
        
        # LLMLingua Compression Summary
        compression_summary = ComponentSummary(name="LLMLingua Compression")
        queries_compressed = sum(1 for r in self.results if r.query_compressed)
        contexts_compressed = sum(1 for r in self.results if r.context_compressed)
        
        compression_summary.queries_processed = total_queries
        compression_summary.total_tokens_saved = sum(r.compression_savings for r in self.results)
        compression_summary.metrics = {
            "queries_compressed": queries_compressed,
            "contexts_compressed": contexts_compressed,
            "total_compression_events": queries_compressed + contexts_compressed,
            "total_tokens_saved": sum(r.compression_savings for r in self.results),
        }
        
        # Check if compression worked when expected
        compression_issues = [r for r in self.results if r.validations.get("compression", {}).get("passed") == False]
        if compression_issues:
            compression_summary.issues = [f"Compression expected but not triggered for {len(compression_issues)} queries"]
        
        self.component_summaries["compression"] = compression_summary
        
        # Token Orchestrator Summary
        orchestrator_summary = ComponentSummary(name="Token Orchestrator")
        simple = sum(1 for r in self.results if r.complexity == "simple")
        medium = sum(1 for r in self.results if r.complexity == "medium")
        complex_q = sum(1 for r in self.results if r.complexity == "complex")
        
        orchestrator_summary.queries_processed = total_queries
        orchestrator_summary.total_tokens_saved = sum(r.orchestrator_savings for r in self.results)
        orchestrator_summary.metrics = {
            "simple_queries": simple,
            "medium_queries": medium,
            "complex_queries": complex_q,
            "avg_token_budget": sum(r.token_budget for r in self.results) / total_queries if total_queries > 0 else 0,
            "total_output_tokens": sum(r.output_tokens for r in self.results),
            "tokens_saved": sum(r.orchestrator_savings for r in self.results),
        }
        
        complexity_issues = [r for r in self.results if r.validations.get("complexity", {}).get("passed") == False]
        if complexity_issues:
            orchestrator_summary.issues = [f"Complexity mismatch for {len(complexity_issues)} queries"]
        
        self.component_summaries["orchestrator"] = orchestrator_summary
        
        # Bandit Optimizer Summary
        bandit_summary = ComponentSummary(name="Bandit Optimizer")
        cheap = sum(1 for r in self.results if r.strategy == "cheap")
        balanced = sum(1 for r in self.results if r.strategy == "balanced")
        premium = sum(1 for r in self.results if r.strategy == "premium")
        
        rewards = [r.reward for r in self.results if r.reward is not None]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        
        bandit_summary.queries_processed = total_queries
        bandit_summary.total_tokens_saved = sum(r.bandit_savings for r in self.results)
        bandit_summary.metrics = {
            "cheap_selections": cheap,
            "balanced_selections": balanced,
            "premium_selections": premium,
            "avg_reward": f"{avg_reward:.4f}",
            "tokens_saved": sum(r.bandit_savings for r in self.results),
        }
        
        # Get routing stats from platform
        routing_stats = self.platform.bandit.get_routing_stats()
        bandit_summary.metrics["routing_stats"] = routing_stats
        
        strategy_issues = [r for r in self.results if r.validations.get("strategy", {}).get("passed") == False]
        if strategy_issues:
            bandit_summary.issues = [f"Strategy selection mismatch for {len(strategy_issues)} queries"]
        
        self.component_summaries["bandit"] = bandit_summary
        
        # Quality Judge Summary
        judge_summary = ComponentSummary(name="Quality Judge")
        judged = [r for r in self.results if r.quality_winner is not None]
        optimized_wins = sum(1 for r in judged if r.quality_winner == "optimized")
        equivalent = sum(1 for r in judged if r.quality_winner == "equivalent")
        baseline_wins = sum(1 for r in judged if r.quality_winner == "baseline")
        
        confidences = [r.quality_confidence for r in judged if r.quality_confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        judge_summary.queries_processed = len(judged)
        judge_summary.metrics = {
            "comparisons_run": len(judged),
            "optimized_wins": optimized_wins,
            "equivalent_results": equivalent,
            "baseline_wins": baseline_wins,
            "quality_maintained_rate": f"{(optimized_wins + equivalent) / len(judged) * 100:.1f}%" if judged else "N/A",
            "avg_confidence": f"{avg_confidence:.2f}",
        }
        
        if judged and baseline_wins / len(judged) > 0.2:
            judge_summary.issues = [f"Baseline wins {baseline_wins / len(judged) * 100:.1f}% - quality may be compromised"]
        
        self.component_summaries["quality_judge"] = judge_summary
        
        # Preference Learning Summary
        pref_summary = ComponentSummary(name="Preference Learning")
        tone_detected = sum(1 for r in self.results if r.detected_tone and r.detected_tone != "neutral")
        format_detected = sum(1 for r in self.results if r.detected_format and r.detected_format != "paragraph")
        
        pref_summary.queries_processed = total_queries
        pref_summary.metrics = {
            "tone_detections": tone_detected,
            "format_detections": format_detected,
            "preferences_learned": tone_detected + format_detected,
        }
        
        tone_issues = [r for r in self.results if r.validations.get("tone", {}).get("passed") == False]
        format_issues = [r for r in self.results if r.validations.get("format", {}).get("passed") == False]
        if tone_issues:
            pref_summary.issues.append(f"Tone detection failed for {len(tone_issues)} queries")
        if format_issues:
            pref_summary.issues.append(f"Format detection failed for {len(format_issues)} queries")
        
        self.component_summaries["preferences"] = pref_summary
    
    def _build_report(self, elapsed_time: float) -> Dict:
        """Build the final report."""
        
        total_queries = len(self.results)
        passed_queries = sum(1 for r in self.results if r.passed)
        failed_queries = total_queries - passed_queries
        
        # Calculate total savings
        total_tokens_used = sum(r.tokens_used for r in self.results)
        total_baseline_tokens = sum(r.baseline_tokens for r in self.results if r.baseline_tokens > 0)
        total_savings = sum(r.total_savings for r in self.results)
        
        # Calculate savings by component
        memory_savings = sum(r.memory_savings for r in self.results)
        orchestrator_savings = sum(r.orchestrator_savings for r in self.results)
        bandit_savings = sum(r.bandit_savings for r in self.results)
        compression_savings = sum(r.compression_savings for r in self.results)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "platform_version": "1.0",
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "total_test_cases": total_queries,
                "elapsed_seconds": elapsed_time,
            },
            "executive_summary": {
                "total_queries": total_queries,
                "passed": passed_queries,
                "failed": failed_queries,
                "pass_rate": f"{passed_queries / total_queries * 100:.1f}%",
                "total_tokens_used": total_tokens_used,
                "total_baseline_tokens": total_baseline_tokens,
                "total_savings": total_savings,
                "savings_percentage": f"{total_savings / (total_tokens_used + total_savings) * 100:.1f}%" if (total_tokens_used + total_savings) > 0 else "0%",
            },
            "savings_breakdown": {
                "memory_layer": {
                    "tokens_saved": memory_savings,
                    "percentage": f"{memory_savings / (total_savings) * 100:.1f}%" if total_savings > 0 else "0%",
                    "description": "Tokens saved from exact and semantic cache hits",
                },
                "orchestrator": {
                    "tokens_saved": orchestrator_savings,
                    "percentage": f"{orchestrator_savings / (total_savings) * 100:.1f}%" if total_savings > 0 else "0%",
                    "description": "Tokens saved from optimized token allocation",
                },
                "bandit": {
                    "tokens_saved": bandit_savings,
                    "percentage": f"{bandit_savings / (total_savings) * 100:.1f}%" if total_savings > 0 else "0%",
                    "description": "Tokens saved from strategy selection",
                },
                "compression": {
                    "tokens_saved": compression_savings,
                    "percentage": f"{compression_savings / (total_savings) * 100:.1f}%" if total_savings > 0 else "0%",
                    "description": "Tokens saved from LLMLingua compression",
                },
            },
            "component_analysis": {
                name: {
                    "queries_processed": summary.queries_processed,
                    "tokens_saved": summary.total_tokens_saved,
                    "metrics": summary.metrics,
                    "issues": summary.issues,
                    "status": "OK" if not summary.issues else "ISSUES_FOUND",
                }
                for name, summary in self.component_summaries.items()
            },
            "issues_found": self.issues_found,
            "test_results": [
                {
                    "test_id": r.test_id,
                    "phase": r.phase,
                    "query": r.query,
                    "description": r.description,
                    "passed": r.passed,
                    "cache_hit": r.cache_hit,
                    "cache_type": r.cache_type,
                    "similarity": r.similarity,
                    "tokens_used": r.tokens_used,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "latency_ms": r.latency_ms,
                    "strategy": r.strategy,
                    "model": r.model,
                    "complexity": r.complexity,
                    "memory_savings": r.memory_savings,
                    "orchestrator_savings": r.orchestrator_savings,
                    "bandit_savings": r.bandit_savings,
                    "total_savings": r.total_savings,
                    "quality_winner": r.quality_winner,
                    "quality_confidence": r.quality_confidence,
                    "validations": r.validations,
                    "issues": r.issues,
                }
                for r in self.results
            ],
        }
        
        return report
    
    def save_results(self, output_dir: str = "diagnostic_results") -> Tuple[str, str, str]:
        """Save results to JSON, Markdown, and CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"extensive_diagnostic_{timestamp}"
        
        # Build report
        report = self._build_report(0)  # elapsed_time already in report
        
        # Save JSON
        json_file = output_path / f"{base_name}.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save Markdown
        md_file = output_path / f"{base_name}.md"
        with open(md_file, "w") as f:
            f.write(self._generate_markdown_report(report))
        
        # Save CSV
        csv_file = output_path / f"{base_name}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Test ID", "Phase", "Description", "Passed", "Cache Type",
                "Similarity", "Tokens Used", "Strategy", "Complexity",
                "Memory Savings", "Orchestrator Savings", "Total Savings", "Issues"
            ])
            for r in self.results:
                writer.writerow([
                    r.test_id, r.phase, r.description[:50], r.passed, r.cache_type,
                    f"{r.similarity:.3f}" if r.similarity else "",
                    r.tokens_used, r.strategy, r.complexity,
                    r.memory_savings, r.orchestrator_savings, r.total_savings,
                    "; ".join(r.issues) if r.issues else ""
                ])
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")
        print(f"  CSV: {csv_file}")
        
        return str(json_file), str(md_file), str(csv_file)
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate a Markdown report for presentation."""
        md = []
        
        md.append("# Tokenomics Platform - Extensive Diagnostic Test Report")
        md.append(f"\n**Generated:** {report['metadata']['timestamp']}")
        md.append(f"\n**Platform Version:** {report['metadata']['platform_version']}")
        md.append(f"\n**LLM Provider:** {report['metadata']['llm_provider']} ({report['metadata']['llm_model']})")
        md.append(f"\n**Test Duration:** {report['metadata']['elapsed_seconds']:.1f} seconds")
        
        # Executive Summary
        md.append("\n\n## Executive Summary")
        summary = report["executive_summary"]
        md.append(f"\n| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Total Queries | {summary['total_queries']} |")
        md.append(f"| Passed | {summary['passed']} ({summary['pass_rate']}) |")
        md.append(f"| Failed | {summary['failed']} |")
        md.append(f"| Total Tokens Used | {summary['total_tokens_used']:,} |")
        md.append(f"| Total Savings | {summary['total_savings']:,} tokens |")
        md.append(f"| Savings Rate | {summary['savings_percentage']} |")
        
        # Savings Breakdown
        md.append("\n\n## Token Savings Breakdown")
        md.append("\n| Component | Tokens Saved | Percentage | Description |")
        md.append("|-----------|--------------|------------|-------------|")
        for name, data in report["savings_breakdown"].items():
            md.append(f"| {name.replace('_', ' ').title()} | {data['tokens_saved']:,} | {data['percentage']} | {data['description']} |")
        
        # Component Analysis
        md.append("\n\n## Component Analysis")
        
        for name, data in report["component_analysis"].items():
            status_emoji = "OK" if data["status"] == "OK" else "ISSUES"
            md.append(f"\n### {name.replace('_', ' ').title()} [{status_emoji}]")
            
            md.append(f"\n- **Queries Processed:** {data['queries_processed']}")
            md.append(f"- **Tokens Saved:** {data['tokens_saved']:,}")
            
            md.append("\n\n**Metrics:**")
            for metric, value in data["metrics"].items():
                if metric != "routing_stats":  # Skip nested dict
                    md.append(f"- {metric}: {value}")
            
            if data["issues"]:
                md.append("\n\n**Issues Found:**")
                for issue in data["issues"]:
                    md.append(f"- {issue}")
        
        # Issues Summary
        if report["issues_found"]:
            md.append("\n\n## Issues Found")
            md.append("\n| Test ID | Phase | Issue |")
            md.append("|---------|-------|-------|")
            for issue in report["issues_found"]:
                md.append(f"| {issue['test_id']} | {issue['phase']} | {issue['issue']} |")
        else:
            md.append("\n\n## Issues Found")
            md.append("\n**No critical issues found.**")
        
        # Phase Results
        md.append("\n\n## Phase-by-Phase Results")
        
        phases = {}
        for result in report["test_results"]:
            phase = result["phase"]
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(result)
        
        for phase_name in sorted(phases.keys()):
            phase_results = phases[phase_name]
            passed = sum(1 for r in phase_results if r["passed"])
            total = len(phase_results)
            
            md.append(f"\n### {phase_name} ({passed}/{total} passed)")
            md.append("\n| Test ID | Cache | Tokens | Strategy | Savings | Passed |")
            md.append("|---------|-------|--------|----------|---------|--------|")
            
            for r in phase_results:
                cache = r["cache_type"] or "none"
                passed_str = "PASS" if r["passed"] else "FAIL"
                md.append(f"| {r['test_id']} | {cache} | {r['tokens_used']} | {r['strategy'] or 'N/A'} | {r['total_savings']} | {passed_str} |")
        
        # Conclusion
        md.append("\n\n## Conclusion")
        
        pass_rate = float(summary["pass_rate"].replace("%", ""))
        if pass_rate >= 90:
            md.append("\nThe Tokenomics Platform is performing excellently with a high pass rate.")
        elif pass_rate >= 70:
            md.append("\nThe Tokenomics Platform is performing well with some areas for improvement.")
        else:
            md.append("\nThe Tokenomics Platform requires attention to several issues.")
        
        savings_pct = summary["savings_percentage"]
        md.append(f"\nToken savings of **{savings_pct}** demonstrates the platform's value proposition.")
        
        return "\n".join(md)


def main():
    """Run the extensive diagnostic test."""
    print("=" * 80)
    print("TOKENOMICS PLATFORM - EXTENSIVE DIAGNOSTIC TEST")
    print("First Version Presentation")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = ExtensiveDiagnosticTest()
    
    # Run all tests
    report = test_suite.run_all_tests()
    
    # Save results
    json_file, md_file, csv_file = test_suite.save_results()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    summary = report["executive_summary"]
    print(f"\nTest Results: {summary['passed']}/{summary['total_queries']} passed ({summary['pass_rate']})")
    print(f"Total Tokens Used: {summary['total_tokens_used']:,}")
    print(f"Total Savings: {summary['total_savings']:,} tokens ({summary['savings_percentage']})")
    
    print("\nSavings by Component:")
    for name, data in report["savings_breakdown"].items():
        print(f"  {name.replace('_', ' ').title()}: {data['tokens_saved']:,} tokens ({data['percentage']})")
    
    print("\nComponent Status:")
    for name, data in report["component_analysis"].items():
        status = "OK" if data["status"] == "OK" else "ISSUES"
        print(f"  {name.replace('_', ' ').title()}: [{status}]")
    
    if report["issues_found"]:
        print(f"\nTotal Issues Found: {len(report['issues_found'])}")
        print("See detailed report for issue descriptions.")
    else:
        print("\nNo critical issues found!")
    
    print("\n" + "=" * 80)
    print(f"Full report: {md_file}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    main()


