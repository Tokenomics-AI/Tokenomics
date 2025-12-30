#!/usr/bin/env python3
"""
Comprehensive A/B Test: Baseline vs Tokenomics Optimized

Runs 50 diverse queries through both baseline (naive) and optimized (Tokenomics) paths,
measuring efficiency, savings, and quality to demonstrate platform value.
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.judge.quality_judge import QualityJudge

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available, using basic aggregation")


@dataclass
class QueryMetadata:
    """Metadata for a test query."""
    query: str
    category: str  # simple, medium, complex, duplicate, paraphrase, context_heavy, long
    expected_complexity: Optional[str] = None  # simple, medium, complex
    expected_cache_type: Optional[str] = None  # exact, semantic, context, none
    is_duplicate_of: Optional[int] = None  # Index of original query for duplicates
    paraphrase_of: Optional[int] = None  # Index of original query for paraphrases
    context_document: Optional[str] = None  # Document to inject for context-heavy queries


@dataclass
class BaselineResult:
    """Result from baseline (naive) query."""
    tokens_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    response_text: str
    cost: float
    model: str


@dataclass
class OptimizedResult:
    """Result from optimized (Tokenomics) query."""
    tokens_used: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    response_text: str
    cost: float
    model: str
    
    # Memory Layer Metrics
    cache_hit: bool
    cache_type: Optional[str]  # exact, semantic_direct, context, none
    similarity: Optional[float]
    context_injected: bool
    context_tokens_added: int
    preferences_used: bool
    
    # Orchestrator Metrics
    complexity: Optional[str]  # simple, medium, complex
    context_quality_score: Optional[float]
    context_compression_ratio: Optional[float]
    query_compression_ratio: Optional[float]
    token_allocations: Optional[Dict[str, int]]
    
    # Bandit Metrics
    strategy: Optional[str]  # cheap, balanced, premium
    context_aware_routing: bool
    
    # Compression Metrics
    query_compressed: bool
    query_original_tokens: Optional[int]
    query_compressed_tokens: Optional[int]
    context_compressed: bool
    context_original_tokens: Optional[int]
    context_compressed_tokens: Optional[int]


@dataclass
class QualityComparison:
    """Quality comparison result."""
    winner: str  # baseline, optimized, equivalent
    confidence: float
    explanation: str
    baseline_score: Optional[float] = None
    optimized_score: Optional[float] = None


@dataclass
class ComparisonMetrics:
    """Metrics comparing baseline vs optimized."""
    token_savings: int
    token_savings_percent: float
    cost_savings: float
    cost_savings_percent: float
    latency_improvement_ms: float
    latency_improvement_percent: float
    quality_winner: str
    quality_confidence: float
    quality_explanation: str


@dataclass
class QueryResult:
    """Complete result for a single query."""
    query_index: int
    query: str
    category: str
    baseline: BaselineResult
    optimized: OptimizedResult
    quality_comparison: Optional[QualityComparison]
    comparison_metrics: ComparisonMetrics
    success: bool
    error: Optional[str] = None


class ABComparisonTest:
    """A/B comparison test suite."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize test suite."""
        self.output_dir = output_dir or (project_root / "tests" / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[QueryResult] = []
        self.platform: Optional[TokenomicsPlatform] = None
        self.baseline_platform: Optional[TokenomicsPlatform] = None
        self.quality_judge: Optional[QualityJudge] = None
        
        # Model pricing (per 1M tokens) - approximate
        self.model_pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
        }
    
    def create_query_dataset(self) -> List[QueryMetadata]:
        """Create diverse query dataset (50 queries)."""
        queries = []
        
        # Simple queries (10)
        simple_queries = [
            "What is the capital of France?",
            "How many days are in a week?",
            "What is 2 + 2?",
            "What is the largest planet in our solar system?",
            "Who wrote Romeo and Juliet?",
            "What is the speed of light?",
            "What is the chemical symbol for gold?",
            "How many continents are there?",
            "What is the smallest prime number?",
            "What is the boiling point of water in Celsius?",
        ]
        for q in simple_queries:
            queries.append(QueryMetadata(
                query=q,
                category="simple",
                expected_complexity="simple",
                expected_cache_type="none"
            ))
        
        # Medium queries (10)
        medium_queries = [
            "Explain how photosynthesis works in plants.",
            "What are the main differences between Python and JavaScript?",
            "Describe the water cycle in nature.",
            "How does a refrigerator work?",
            "What is the difference between HTTP and HTTPS?",
            "Explain the concept of supply and demand in economics.",
            "How do vaccines work in the human body?",
            "What is the difference between RAM and ROM?",
            "Explain the greenhouse effect.",
            "How does a computer process information?",
        ]
        for q in medium_queries:
            queries.append(QueryMetadata(
                query=q,
                category="medium",
                expected_complexity="medium",
                expected_cache_type="none"
            ))
        
        # Complex queries (10)
        complex_queries = [
            "Design a microservices architecture for an e-commerce platform with payment processing, inventory management, and recommendation engine. Include API gateway, service mesh, and database per service pattern.",
            "Explain the mathematical foundations of machine learning, including gradient descent, backpropagation, and regularization techniques. Provide examples of how these concepts apply to neural networks.",
            "Create a comprehensive security strategy for a cloud-based application handling sensitive financial data, including encryption, access control, monitoring, and incident response procedures.",
            "Design a distributed system for real-time analytics on streaming data from IoT devices, including data ingestion, processing pipeline, storage, and visualization components.",
            "Explain the principles of quantum computing, including qubits, superposition, entanglement, and quantum algorithms like Shor's algorithm and Grover's algorithm.",
            "Design a scalable database architecture for a social media platform with 100 million users, including schema design, indexing strategies, caching layers, and replication.",
            "Create a comprehensive DevOps pipeline for a microservices application, including CI/CD, containerization, orchestration, monitoring, and automated testing strategies.",
            "Explain the architecture of a modern web browser, including rendering engine, JavaScript engine, networking stack, and security mechanisms.",
            "Design a machine learning pipeline for fraud detection in financial transactions, including feature engineering, model selection, training, validation, and deployment strategies.",
            "Explain the principles of distributed consensus algorithms, including Paxos, Raft, and Byzantine fault tolerance, with examples of their use in distributed systems.",
        ]
        for q in complex_queries:
            queries.append(QueryMetadata(
                query=q,
                category="complex",
                expected_complexity="complex",
                expected_cache_type="none"
            ))
        
        # Duplicate queries (5) - exact repeats of first 5 simple queries
        for i in range(5):
            queries.append(QueryMetadata(
                query=simple_queries[i],
                category="duplicate",
                expected_complexity="simple",
                expected_cache_type="exact",
                is_duplicate_of=i
            ))
        
        # Semantic paraphrases (5) - similar meaning, different wording
        # First add originals, then paraphrases
        paraphrase_originals = [
            "How do I bake a cake?",
            "Explain the theory of evolution.",
            "What causes climate change?",
            "How does the internet work?",
            "What is artificial intelligence?",
        ]
        paraphrase_variations = [
            "What is the recipe for baking a cake?",
            "Can you describe how evolution works?",
            "Why does the climate change?",
            "Explain how the internet functions.",
            "Can you define what AI is?",
        ]
        
        # Add originals first
        original_indices = []
        for original in paraphrase_originals:
            idx = len(queries)
            queries.append(QueryMetadata(
                query=original,
                category="paraphrase_original",
                expected_cache_type="none"
            ))
            original_indices.append(idx)
        
        # Add paraphrases
        for i, paraphrase in enumerate(paraphrase_variations):
            queries.append(QueryMetadata(
                query=paraphrase,
                category="paraphrase",
                expected_cache_type="semantic",
                paraphrase_of=original_indices[i]
            ))
        
        # Context-heavy queries (5) - will have pre-populated memory
        context_queries = [
            "What was the main goal of the Apollo program?",
            "How many Apollo missions landed on the moon?",
            "Who was the first person to walk on the moon?",
            "What was the name of the Apollo 11 command module?",
            "What year did the Apollo program end?",
        ]
        # Context document about Apollo program
        apollo_document = """
        The Apollo program was a United States human spaceflight program carried out by NASA from 1961 to 1972. 
        Its primary goal was to land humans on the Moon and bring them safely back to Earth. 
        The program achieved this goal with Apollo 11 on July 20, 1969, when astronauts Neil Armstrong and Buzz Aldrin 
        became the first humans to land on the Moon. Armstrong was the first person to step onto the lunar surface, 
        followed by Aldrin 19 minutes later. The Apollo program consisted of 11 crewed missions, with 6 successfully 
        landing on the Moon (Apollo 11, 12, 14, 15, 16, and 17). The command module for Apollo 11 was named Columbia, 
        and the lunar module was named Eagle. The program ended in 1972 with Apollo 17, the last crewed mission to the Moon.
        """
        for q in context_queries:
            queries.append(QueryMetadata(
                query=q,
                category="context_heavy",
                expected_cache_type="context",
                context_document=apollo_document
            ))
        
        # Long queries (5) - >500 chars to trigger LLMLingua compression
        long_queries = [
            "I need a comprehensive explanation of how modern web applications handle authentication and authorization. Please cover OAuth 2.0, JWT tokens, session management, password hashing, multi-factor authentication, role-based access control, and best practices for securing APIs. Also explain common vulnerabilities like CSRF, XSS, and SQL injection, and how to prevent them. Include examples of implementation in popular frameworks.",
            "Explain in detail the architecture of distributed systems, including concepts like consistency models (strong consistency, eventual consistency), CAP theorem, distributed transactions, consensus algorithms (Paxos, Raft), load balancing strategies, service discovery, circuit breakers, and monitoring. Provide examples of real-world distributed systems and how they handle challenges like network partitions and node failures.",
            "I want a thorough guide on machine learning model deployment in production. Cover topics like model versioning, A/B testing, feature stores, online vs batch inference, model serving infrastructure, monitoring model performance and drift, handling data quality issues, scaling inference systems, and maintaining models in production. Include best practices and common pitfalls.",
            "Provide a detailed explanation of cloud computing architectures, including Infrastructure as a Service (IaaS), Platform as a Service (PaaS), Software as a Service (SaaS), serverless computing, containerization with Docker and Kubernetes, microservices architecture, API gateways, service meshes, and cloud-native design patterns. Explain the benefits and trade-offs of each approach.",
            "I need comprehensive information about database design and optimization. Cover relational database design principles, normalization, indexing strategies, query optimization, transaction management, ACID properties, NoSQL database types (document, key-value, column-family, graph), database scaling strategies (vertical and horizontal), replication, sharding, and choosing the right database for different use cases.",
        ]
        for q in long_queries:
            queries.append(QueryMetadata(
                query=q,
                category="long",
                expected_complexity="complex",
                expected_cache_type="none"
            ))
        
        return queries
    
    def initialize_platforms(self):
        """Initialize Tokenomics platform and baseline platform."""
        config = TokenomicsConfig.from_env()
        
        # Optimized platform (all features enabled)
        self.platform = TokenomicsPlatform(config=config)
        
        # Baseline platform (minimal config, will use _run_baseline_query)
        # Note: _run_baseline_query doesn't use cache, bandit, or compression anyway
        # So we can use the same platform instance, just call _run_baseline_query
        self.baseline_platform = TokenomicsPlatform(config=config)
        
        # Initialize quality judge
        if config.judge.enabled:
            self.quality_judge = QualityJudge(config=config.judge)
    
    def pre_populate_memory(self, context_document: str):
        """Pre-populate memory with context document."""
        if not self.platform:
            return
        
        # Store the document in memory by making a query that will store it
        # We'll use a simple query that references the document
        # Split document into chunks if too long
        chunk_size = 1000  # characters per chunk
        chunks = [context_document[i:i+chunk_size] for i in range(0, len(context_document), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            self.platform.memory.store(
                query=f"Apollo program information part {i+1}",
                response=chunk,
                tokens_used=0,
                metadata={"source": "test_context", "chunk_index": i}
            )
    
    def run_baseline_query(self, query: str) -> BaselineResult:
        """Run baseline (naive) query."""
        start_time = time.time()
        
        result = self.baseline_platform._run_baseline_query(query)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost
        model = result.get("model", "gpt-4o-mini")
        pricing = self.model_pricing.get(model, self.model_pricing["gpt-4o-mini"])
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
        
        return BaselineResult(
            tokens_used=result.get("tokens_used", 0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            response_text=result.get("response", ""),
            cost=cost,
            model=model
        )
    
    def run_optimized_query(self, query: str, pre_populate: bool = False, context_doc: Optional[str] = None) -> OptimizedResult:
        """Run optimized (Tokenomics) query."""
        if pre_populate and context_doc:
            self.pre_populate_memory(context_doc)
        
        start_time = time.time()
        
        result = self.platform.query(
            query=query,
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract memory metrics
        memory_metrics = result.get("memory_metrics", {})
        cache_type = result.get("cache_type")
        if not cache_type:
            if memory_metrics.get("exact_cache_hits", 0) > 0:
                cache_type = "exact"
            elif memory_metrics.get("semantic_direct_hits", 0) > 0:
                cache_type = "semantic_direct"
            elif memory_metrics.get("semantic_context_hits", 0) > 0:
                cache_type = "context"
            else:
                cache_type = "none"
        
        # Extract orchestrator metrics
        plan = result.get("plan")
        complexity = result.get("query_type")  # Also available as query_type
        context_quality_score = None
        context_compression_ratio = None
        query_compression_ratio = None
        token_allocations = None
        
        if plan:
            complexity = getattr(plan, "complexity", None) or complexity
            context_quality_score = getattr(plan, "context_quality_score", None)
            context_compression_ratio = getattr(plan, "context_compression_ratio", None)
            # Extract token allocations if available
            if hasattr(plan, "components"):
                token_allocations = {}
                for comp_name, comp_data in plan.components.items():
                    if isinstance(comp_data, dict):
                        token_allocations[comp_name] = comp_data.get("allocated", comp_data.get("cost", 0))
        
        # Extract compression metrics
        compression_metrics = result.get("compression_metrics", {})
        query_compression_ratio = compression_metrics.get("query_compression_ratio")
        if not context_compression_ratio:
            context_compression_ratio = compression_metrics.get("context_compression_ratio")
        
        # Extract bandit metrics
        strategy = result.get("strategy")
        strategy_name = None
        if strategy:
            if isinstance(strategy, dict):
                strategy_name = strategy.get("arm_id", "").replace("strategy_", "")
            elif isinstance(strategy, str):
                strategy_name = strategy.replace("strategy_", "")
        
        # Determine context-aware routing
        context_aware_routing = False
        if context_quality_score is not None and context_quality_score < 0.7:
            # Check if premium strategy was selected (indicating context-aware routing)
            if strategy_name == "premium":
                context_aware_routing = True
        
        # Extract compression metrics from compression_metrics dict
        query_compressed = compression_metrics.get("query_compressed", False)
        query_original_tokens = compression_metrics.get("query_original_tokens")
        query_compressed_tokens = compression_metrics.get("query_compressed_tokens")
        context_compressed = compression_metrics.get("context_compressed", False) or memory_metrics.get("context_compressed", False)
        context_original_tokens = compression_metrics.get("context_original_tokens") or memory_metrics.get("context_original_tokens", 0)
        context_compressed_tokens = compression_metrics.get("context_compressed_tokens") or memory_metrics.get("context_compressed_tokens", 0)
        
        # Calculate cost
        model = result.get("model", "gpt-4o-mini")
        pricing = self.model_pricing.get(model, self.model_pricing["gpt-4o-mini"])
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
        
        return OptimizedResult(
            tokens_used=result.get("tokens_used", 0),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            response_text=result.get("response", ""),
            cost=cost,
            model=model,
            cache_hit=cache_type != "none",
            cache_type=cache_type,
            similarity=memory_metrics.get("top_similarity"),
            context_injected=memory_metrics.get("context_injected", False),
            context_tokens_added=memory_metrics.get("context_tokens_added", 0),
            preferences_used=memory_metrics.get("preferences_used", False),
            complexity=complexity,
            context_quality_score=context_quality_score,
            context_compression_ratio=context_compression_ratio,
            query_compression_ratio=query_compression_ratio,
            token_allocations=token_allocations,
            strategy=strategy_name,
            context_aware_routing=context_aware_routing,
            query_compressed=query_compressed,
            query_original_tokens=query_original_tokens,
            query_compressed_tokens=query_compressed_tokens,
            context_compressed=context_compressed,
            context_original_tokens=context_original_tokens if context_original_tokens else None,
            context_compressed_tokens=context_compressed_tokens if context_compressed_tokens else None,
        )
    
    def compare_quality(self, query: str, baseline_response: str, optimized_response: str) -> Optional[QualityComparison]:
        """Compare quality of baseline vs optimized responses."""
        if not self.quality_judge:
            return None
        
        try:
            judge_result = self.quality_judge.judge(query, baseline_response, optimized_response)
            if not judge_result:
                return None
            
            return QualityComparison(
                winner=judge_result.winner,
                confidence=judge_result.confidence,
                explanation=judge_result.explanation or "",
                baseline_score=None,  # Judge doesn't provide individual scores
                optimized_score=None
            )
        except Exception as e:
            print(f"⚠️  Quality comparison failed: {e}")
            return None
    
    def calculate_metrics(self, baseline: BaselineResult, optimized: OptimizedResult, quality: Optional[QualityComparison]) -> ComparisonMetrics:
        """Calculate comparison metrics."""
        token_savings = baseline.tokens_used - optimized.tokens_used
        token_savings_percent = (token_savings / baseline.tokens_used * 100) if baseline.tokens_used > 0 else 0
        
        cost_savings = baseline.cost - optimized.cost
        cost_savings_percent = (cost_savings / baseline.cost * 100) if baseline.cost > 0 else 0
        
        latency_improvement_ms = baseline.latency_ms - optimized.latency_ms
        latency_improvement_percent = (latency_improvement_ms / baseline.latency_ms * 100) if baseline.latency_ms > 0 else 0
        
        quality_winner = quality.winner if quality else "unknown"
        quality_confidence = quality.confidence if quality else 0.0
        quality_explanation = quality.explanation if quality else ""
        
        return ComparisonMetrics(
            token_savings=token_savings,
            token_savings_percent=token_savings_percent,
            cost_savings=cost_savings,
            cost_savings_percent=cost_savings_percent,
            latency_improvement_ms=latency_improvement_ms,
            latency_improvement_percent=latency_improvement_percent,
            quality_winner=quality_winner,
            quality_confidence=quality_confidence,
            quality_explanation=quality_explanation
        )
    
    def run_test(self):
        """Run the complete A/B test."""
        print("🚀 Starting A/B Comparison Test")
        print("=" * 80)
        
        # Initialize platforms
        print("\n📦 Initializing platforms...")
        self.initialize_platforms()
        print("✅ Platforms initialized")
        
        # Create query dataset
        print("\n📝 Creating query dataset...")
        queries = self.create_query_dataset()
        print(f"✅ Created {len(queries)} queries")
        
        # Run tests
        print("\n🧪 Running tests...")
        print("=" * 80)
        
        for i, query_meta in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Testing: {query_meta.category} - {query_meta.query[:60]}...")
            
            try:
                # Pre-populate memory if needed
                pre_populate = query_meta.category == "context_heavy"
                context_doc = query_meta.context_document if pre_populate else None
                
                # Run baseline
                print("  → Running baseline...")
                baseline_result = self.run_baseline_query(query_meta.query)
                
                # Run optimized
                print("  → Running optimized...")
                optimized_result = self.run_optimized_query(
                    query_meta.query,
                    pre_populate=pre_populate,
                    context_doc=context_doc
                )
                
                # Compare quality
                print("  → Comparing quality...")
                quality_comparison = self.compare_quality(
                    query_meta.query,
                    baseline_result.response_text,
                    optimized_result.response_text
                )
                
                # Calculate metrics
                comparison_metrics = self.calculate_metrics(
                    baseline_result,
                    optimized_result,
                    quality_comparison
                )
                
                # Store result
                result = QueryResult(
                    query_index=i,
                    query=query_meta.query,
                    category=query_meta.category,
                    baseline=baseline_result,
                    optimized=optimized_result,
                    quality_comparison=quality_comparison,
                    comparison_metrics=comparison_metrics,
                    success=True
                )
                self.results.append(result)
                
                # Print summary
                print(f"  ✅ Baseline: {baseline_result.tokens_used} tokens, ${baseline_result.cost:.4f}, {baseline_result.latency_ms:.0f}ms")
                print(f"  ✅ Optimized: {optimized_result.tokens_used} tokens, ${optimized_result.cost:.4f}, {optimized_result.latency_ms:.0f}ms")
                print(f"  💰 Savings: {comparison_metrics.token_savings} tokens ({comparison_metrics.token_savings_percent:.1f}%), ${comparison_metrics.cost_savings:.4f}")
                if quality_comparison:
                    print(f"  🏆 Quality: {quality_comparison.winner} (confidence: {quality_comparison.confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                result = QueryResult(
                    query_index=i,
                    query=query_meta.query,
                    category=query_meta.category,
                    baseline=BaselineResult(0, 0, 0, 0, "", 0, ""),
                    optimized=OptimizedResult(0, 0, 0, 0, "", 0, "", False, None, None, False, 0, False, None, None, None, None, None, None, False, None, None, False, None, None),
                    quality_comparison=None,
                    comparison_metrics=ComparisonMetrics(0, 0, 0, 0, 0, 0, "unknown", 0, ""),
                    success=False,
                    error=str(e)
                )
                self.results.append(result)
        
        print("\n" + "=" * 80)
        print("✅ Test completed!")
        
        # Print summary statistics
        self.print_summary_statistics()
        
        # Generate reports
        print("\n📊 Generating reports...")
        self.generate_reports()
        print("✅ Reports generated")
    
    def print_summary_statistics(self):
        """Print summary statistics."""
        aggregated = self.aggregate_results()
        
        print("\n" + "=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        
        print(f"\n✅ Successful Queries: {aggregated.get('successful_queries', 0)}/{aggregated.get('total_queries', 0)}")
        print(f"❌ Failed Queries: {aggregated.get('failed_queries', 0)}")
        
        print(f"\n💰 TOKEN SAVINGS:")
        print(f"  Total: {aggregated.get('total_token_savings', 0):,} tokens ({aggregated.get('total_token_savings_percent', 0):.1f}%)")
        print(f"  Baseline: {aggregated.get('total_baseline_tokens', 0):,} tokens")
        print(f"  Optimized: {aggregated.get('total_optimized_tokens', 0):,} tokens")
        
        print(f"\n💵 COST SAVINGS:")
        print(f"  Total: ${aggregated.get('total_cost_savings', 0):.4f} ({aggregated.get('total_cost_savings_percent', 0):.1f}%)")
        print(f"  Baseline: ${aggregated.get('total_baseline_cost', 0):.4f}")
        print(f"  Optimized: ${aggregated.get('total_optimized_cost', 0):.4f}")
        
        print(f"\n⚡ LATENCY IMPROVEMENT:")
        print(f"  Average: {aggregated.get('avg_latency_improvement_ms', 0):.0f}ms ({aggregated.get('avg_latency_improvement_percent', 0):.1f}%)")
        print(f"  Baseline: {aggregated.get('avg_baseline_latency_ms', 0):.0f}ms")
        print(f"  Optimized: {aggregated.get('avg_optimized_latency_ms', 0):.0f}ms")
        
        cache_metrics = aggregated.get('cache_metrics', {})
        print(f"\n💾 CACHE EFFICIENCY:")
        print(f"  Exact Hits: {cache_metrics.get('exact_cache_hits', 0)}")
        print(f"  Semantic Hits: {cache_metrics.get('semantic_cache_hits', 0)}")
        print(f"  Misses: {cache_metrics.get('cache_misses', 0)}")
        print(f"  Hit Rate: {cache_metrics.get('cache_hit_rate', 0):.1f}%")
        
        orchestrator_metrics = aggregated.get('orchestrator_metrics', {})
        print(f"\n🎯 ORCHESTRATOR:")
        complexity_dist = orchestrator_metrics.get('complexity_distribution', {})
        print(f"  Complexity: Simple={complexity_dist.get('simple', 0)}, Medium={complexity_dist.get('medium', 0)}, Complex={complexity_dist.get('complex', 0)}")
        print(f"  Query Compression: {orchestrator_metrics.get('query_compressed_count', 0)} queries")
        print(f"  Context Compression: {orchestrator_metrics.get('context_compressed_count', 0)} contexts")
        
        bandit_metrics = aggregated.get('bandit_metrics', {})
        strategy_dist = bandit_metrics.get('strategy_distribution', {})
        print(f"\n🎲 BANDIT OPTIMIZER:")
        print(f"  Strategy: Cheap={strategy_dist.get('cheap', 0)}, Balanced={strategy_dist.get('balanced', 0)}, Premium={strategy_dist.get('premium', 0)}")
        print(f"  Context-Aware Routing: {bandit_metrics.get('context_aware_routing_count', 0)} queries")
        
        quality_metrics = aggregated.get('quality_metrics', {})
        print(f"\n🏆 QUALITY:")
        print(f"  Preservation Rate: {quality_metrics.get('quality_preservation_rate', 0):.1f}%")
        print(f"  Average Confidence: {quality_metrics.get('avg_quality_confidence', 0):.2f}")
        winners = quality_metrics.get('quality_winners', {})
        print(f"  Winners: Optimized={winners.get('optimized', 0)}, Baseline={winners.get('baseline', 0)}, Equivalent={winners.get('equivalent', 0)}")
        
        print("\n" + "=" * 80)
    
    def generate_reports(self):
        """Generate markdown, JSON, and CSV reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate markdown report
        md_path = self.output_dir / f"AB_COMPARISON_REPORT_{timestamp}.md"
        self.generate_markdown_report(md_path)
        
        # Generate JSON report
        json_path = self.output_dir / f"ab_comparison_results_{timestamp}.json"
        self.generate_json_report(json_path)
        
        # Generate CSV report
        csv_path = self.output_dir / f"ab_comparison_results_{timestamp}.csv"
        self.generate_csv_report(csv_path)
        
        print(f"\n📄 Reports saved:")
        print(f"  - Markdown: {md_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV: {csv_path}")
    
    def aggregate_results(self) -> Dict:
        """Aggregate results for reporting."""
        successful = [r for r in self.results if r.success]
        
        if not successful:
            return {}
        
        # Basic metrics
        total_baseline_tokens = sum(r.baseline.tokens_used for r in successful)
        total_optimized_tokens = sum(r.optimized.tokens_used for r in successful)
        total_baseline_cost = sum(r.baseline.cost for r in successful)
        total_optimized_cost = sum(r.optimized.cost for r in successful)
        avg_baseline_latency = sum(r.baseline.latency_ms for r in successful) / len(successful)
        avg_optimized_latency = sum(r.optimized.latency_ms for r in successful) / len(successful)
        
        # Component-specific metrics
        exact_cache_hits = sum(1 for r in successful if r.optimized.cache_type == "exact")
        semantic_cache_hits = sum(1 for r in successful if r.optimized.cache_type in ["semantic_direct", "context"])
        cache_misses = sum(1 for r in successful if r.optimized.cache_type == "none")
        cache_hit_rate = (exact_cache_hits + semantic_cache_hits) / len(successful) * 100
        
        context_injection_count = sum(1 for r in successful if r.optimized.context_injected)
        preferences_used_count = sum(1 for r in successful if r.optimized.preferences_used)
        
        # Complexity distribution
        complexity_dist = {}
        for r in successful:
            if r.optimized.complexity:
                complexity_dist[r.optimized.complexity] = complexity_dist.get(r.optimized.complexity, 0) + 1
        
        # Strategy distribution
        strategy_dist = {}
        for r in successful:
            if r.optimized.strategy:
                strategy_dist[r.optimized.strategy] = strategy_dist.get(r.optimized.strategy, 0) + 1
        
        # Compression metrics
        query_compressed_count = sum(1 for r in successful if r.optimized.query_compressed)
        context_compressed_count = sum(1 for r in successful if r.optimized.context_compressed)
        
        compression_ratios = [r.optimized.query_compression_ratio for r in successful if r.optimized.query_compression_ratio]
        if compression_ratios:
            avg_query_compression = sum(compression_ratios) / len(compression_ratios)
        else:
            avg_query_compression = None
        
        context_compression_ratios = [r.optimized.context_compression_ratio for r in successful if r.optimized.context_compression_ratio]
        if context_compression_ratios:
            avg_context_compression = sum(context_compression_ratios) / len(context_compression_ratios)
        else:
            avg_context_compression = None
        
        # Context-aware routing
        context_aware_routing_count = sum(1 for r in successful if r.optimized.context_aware_routing)
        
        # Quality metrics
        quality_winners = {}
        quality_confidences = []
        for r in successful:
            if r.quality_comparison:
                winner = r.quality_comparison.winner
                quality_winners[winner] = quality_winners.get(winner, 0) + 1
                quality_confidences.append(r.quality_comparison.confidence)
        
        avg_quality_confidence = sum(quality_confidences) / len(quality_confidences) if quality_confidences else 0
        quality_preservation_rate = (quality_winners.get("optimized", 0) + quality_winners.get("equivalent", 0)) / len([r for r in successful if r.quality_comparison]) * 100 if quality_confidences else 0
        
        # Savings by category
        savings_by_category = {}
        for r in successful:
            cat = r.category
            if cat not in savings_by_category:
                savings_by_category[cat] = {"tokens": 0, "cost": 0, "count": 0}
            savings_by_category[cat]["tokens"] += r.comparison_metrics.token_savings
            savings_by_category[cat]["cost"] += r.comparison_metrics.cost_savings
            savings_by_category[cat]["count"] += 1
        
        # Savings by cache type
        savings_by_cache = {}
        for r in successful:
            cache_type = r.optimized.cache_type or "none"
            if cache_type not in savings_by_cache:
                savings_by_cache[cache_type] = {"tokens": 0, "cost": 0, "count": 0}
            savings_by_cache[cache_type]["tokens"] += r.comparison_metrics.token_savings
            savings_by_cache[cache_type]["cost"] += r.comparison_metrics.cost_savings
            savings_by_cache[cache_type]["count"] += 1
        
        # Savings by strategy
        savings_by_strategy = {}
        for r in successful:
            strategy = r.optimized.strategy or "unknown"
            if strategy not in savings_by_strategy:
                savings_by_strategy[strategy] = {"tokens": 0, "cost": 0, "count": 0}
            savings_by_strategy[strategy]["tokens"] += r.comparison_metrics.token_savings
            savings_by_strategy[strategy]["cost"] += r.comparison_metrics.cost_savings
            savings_by_strategy[strategy]["count"] += 1
        
        return {
            "total_queries": len(self.results),
            "successful_queries": len(successful),
            "failed_queries": len(self.results) - len(successful),
            "total_baseline_tokens": total_baseline_tokens,
            "total_optimized_tokens": total_optimized_tokens,
            "total_token_savings": total_baseline_tokens - total_optimized_tokens,
            "total_token_savings_percent": ((total_baseline_tokens - total_optimized_tokens) / total_baseline_tokens * 100) if total_baseline_tokens > 0 else 0,
            "total_baseline_cost": total_baseline_cost,
            "total_optimized_cost": total_optimized_cost,
            "total_cost_savings": total_baseline_cost - total_optimized_cost,
            "total_cost_savings_percent": ((total_baseline_cost - total_optimized_cost) / total_baseline_cost * 100) if total_baseline_cost > 0 else 0,
            "avg_baseline_latency_ms": avg_baseline_latency,
            "avg_optimized_latency_ms": avg_optimized_latency,
            "avg_latency_improvement_ms": avg_baseline_latency - avg_optimized_latency,
            "avg_latency_improvement_percent": ((avg_baseline_latency - avg_optimized_latency) / avg_baseline_latency * 100) if avg_baseline_latency > 0 else 0,
            "cache_metrics": {
                "exact_cache_hits": exact_cache_hits,
                "semantic_cache_hits": semantic_cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate,
            },
            "memory_metrics": {
                "context_injection_count": context_injection_count,
                "preferences_used_count": preferences_used_count,
            },
            "orchestrator_metrics": {
                "complexity_distribution": complexity_dist,
                "avg_query_compression_ratio": avg_query_compression,
                "avg_context_compression_ratio": avg_context_compression,
                "query_compressed_count": query_compressed_count,
                "context_compressed_count": context_compressed_count,
            },
            "bandit_metrics": {
                "strategy_distribution": strategy_dist,
                "context_aware_routing_count": context_aware_routing_count,
            },
            "quality_metrics": {
                "quality_winners": quality_winners,
                "avg_quality_confidence": avg_quality_confidence,
                "quality_preservation_rate": quality_preservation_rate,
            },
            "savings_by_category": savings_by_category,
            "savings_by_cache_type": savings_by_cache,
            "savings_by_strategy": savings_by_strategy,
        }
    
    def generate_markdown_report(self, filepath: Path):
        """Generate markdown report."""
        aggregated = self.aggregate_results()
        
        with open(filepath, 'w') as f:
            f.write("# A/B Comparison Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Queries:** {aggregated.get('total_queries', 0)}\n")
            f.write(f"- **Successful Queries:** {aggregated.get('successful_queries', 0)}\n")
            f.write(f"- **Failed Queries:** {aggregated.get('failed_queries', 0)}\n\n")
            
            f.write("### Overall Savings\n\n")
            f.write(f"- **Token Savings:** {aggregated.get('total_token_savings', 0):,} tokens ({aggregated.get('total_token_savings_percent', 0):.1f}%)\n")
            f.write(f"- **Cost Savings:** ${aggregated.get('total_cost_savings', 0):.4f} ({aggregated.get('total_cost_savings_percent', 0):.1f}%)\n")
            f.write(f"- **Latency Improvement:** {aggregated.get('avg_latency_improvement_ms', 0):.0f}ms ({aggregated.get('avg_latency_improvement_percent', 0):.1f}%)\n\n")
            
            f.write("### Quality Preservation\n\n")
            quality_metrics = aggregated.get('quality_metrics', {})
            f.write(f"- **Quality Preservation Rate:** {quality_metrics.get('quality_preservation_rate', 0):.1f}%\n")
            f.write(f"- **Average Quality Confidence:** {quality_metrics.get('avg_quality_confidence', 0):.2f}\n")
            winners = quality_metrics.get('quality_winners', {})
            f.write(f"- **Quality Winners:** Optimized: {winners.get('optimized', 0)}, Baseline: {winners.get('baseline', 0)}, Equivalent: {winners.get('equivalent', 0)}\n\n")
            
            # Component Analysis
            f.write("## Component Analysis\n\n")
            
            # Memory Layer
            f.write("### Memory Layer\n\n")
            cache_metrics = aggregated.get('cache_metrics', {})
            f.write(f"- **Exact Cache Hits:** {cache_metrics.get('exact_cache_hits', 0)}\n")
            f.write(f"- **Semantic Cache Hits:** {cache_metrics.get('semantic_cache_hits', 0)}\n")
            f.write(f"- **Cache Misses:** {cache_metrics.get('cache_misses', 0)}\n")
            f.write(f"- **Cache Hit Rate:** {cache_metrics.get('cache_hit_rate', 0):.1f}%\n\n")
            
            memory_metrics = aggregated.get('memory_metrics', {})
            f.write(f"- **Context Injection Usage:** {memory_metrics.get('context_injection_count', 0)}\n")
            f.write(f"- **Preference Learning Usage:** {memory_metrics.get('preferences_used_count', 0)}\n\n")
            
            # Orchestrator
            f.write("### Token Orchestrator\n\n")
            orchestrator_metrics = aggregated.get('orchestrator_metrics', {})
            complexity_dist = orchestrator_metrics.get('complexity_distribution', {})
            f.write(f"- **Complexity Distribution:** Simple: {complexity_dist.get('simple', 0)}, Medium: {complexity_dist.get('medium', 0)}, Complex: {complexity_dist.get('complex', 0)}\n")
            f.write(f"- **Query Compression:** {orchestrator_metrics.get('query_compressed_count', 0)} queries compressed\n")
            if orchestrator_metrics.get('avg_query_compression_ratio'):
                f.write(f"- **Average Query Compression Ratio:** {orchestrator_metrics.get('avg_query_compression_ratio', 0):.3f}\n")
            f.write(f"- **Context Compression:** {orchestrator_metrics.get('context_compressed_count', 0)} contexts compressed\n")
            if orchestrator_metrics.get('avg_context_compression_ratio'):
                f.write(f"- **Average Context Compression Ratio:** {orchestrator_metrics.get('avg_context_compression_ratio', 0):.3f}\n\n")
            
            # Bandit
            f.write("### Bandit Optimizer\n\n")
            bandit_metrics = aggregated.get('bandit_metrics', {})
            strategy_dist = bandit_metrics.get('strategy_distribution', {})
            f.write(f"- **Strategy Distribution:** Cheap: {strategy_dist.get('cheap', 0)}, Balanced: {strategy_dist.get('balanced', 0)}, Premium: {strategy_dist.get('premium', 0)}\n")
            f.write(f"- **Context-Aware Routing:** {bandit_metrics.get('context_aware_routing_count', 0)} queries\n\n")
            
            # Savings Breakdown
            f.write("## Savings Breakdown\n\n")
            
            f.write("### By Category\n\n")
            savings_by_category = aggregated.get('savings_by_category', {})
            for category, savings in savings_by_category.items():
                count = savings.get('count', 0)
                tokens = savings.get('tokens', 0)
                cost = savings.get('cost', 0)
                f.write(f"- **{category}:** {tokens:,} tokens (${cost:.4f}) across {count} queries\n")
            f.write("\n")
            
            f.write("### By Cache Type\n\n")
            savings_by_cache = aggregated.get('savings_by_cache_type', {})
            for cache_type, savings in savings_by_cache.items():
                count = savings.get('count', 0)
                tokens = savings.get('tokens', 0)
                cost = savings.get('cost', 0)
                f.write(f"- **{cache_type}:** {tokens:,} tokens (${cost:.4f}) across {count} queries\n")
            f.write("\n")
            
            f.write("### By Strategy\n\n")
            savings_by_strategy = aggregated.get('savings_by_strategy', {})
            for strategy, savings in savings_by_strategy.items():
                count = savings.get('count', 0)
                tokens = savings.get('tokens', 0)
                cost = savings.get('cost', 0)
                f.write(f"- **{strategy}:** {tokens:,} tokens (${cost:.4f}) across {count} queries\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Query | Category | Baseline Tokens | Optimized Tokens | Savings | Baseline Cost | Optimized Cost | Cost Savings | Quality Winner |\n")
            f.write("|-------|----------|-----------------|------------------|---------|---------------|----------------|--------------|----------------|\n")
            
            for r in self.results:
                if r.success:
                    quality_winner = r.quality_comparison.winner if r.quality_comparison else "N/A"
                    f.write(f"| {r.query[:50]}... | {r.category} | {r.baseline.tokens_used} | {r.optimized.tokens_used} | {r.comparison_metrics.token_savings} ({r.comparison_metrics.token_savings_percent:.1f}%) | ${r.baseline.cost:.4f} | ${r.optimized.cost:.4f} | ${r.comparison_metrics.cost_savings:.4f} | {quality_winner} |\n")
    
    def generate_json_report(self, filepath: Path):
        """Generate JSON report."""
        aggregated = self.aggregate_results()
        
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_queries": len(self.results),
                "successful_queries": aggregated.get('successful_queries', 0),
                "failed_queries": aggregated.get('failed_queries', 0),
            },
            "summary": {
                "total_token_savings": aggregated.get('total_token_savings', 0),
                "total_token_savings_percent": aggregated.get('total_token_savings_percent', 0),
                "total_cost_savings": aggregated.get('total_cost_savings', 0),
                "total_cost_savings_percent": aggregated.get('total_cost_savings_percent', 0),
                "avg_latency_improvement_ms": aggregated.get('avg_latency_improvement_ms', 0),
                "avg_latency_improvement_percent": aggregated.get('avg_latency_improvement_percent', 0),
            },
            "aggregated_metrics": aggregated,
            "detailed_results": []
        }
        
        for r in self.results:
            result_dict = {
                "query_index": r.query_index,
                "query": r.query,
                "category": r.category,
                "success": r.success,
                "error": r.error,
                "baseline": {
                    "tokens_used": r.baseline.tokens_used,
                    "input_tokens": r.baseline.input_tokens,
                    "output_tokens": r.baseline.output_tokens,
                    "latency_ms": r.baseline.latency_ms,
                    "cost": r.baseline.cost,
                    "model": r.baseline.model,
                },
                "optimized": {
                    "tokens_used": r.optimized.tokens_used,
                    "input_tokens": r.optimized.input_tokens,
                    "output_tokens": r.optimized.output_tokens,
                    "latency_ms": r.optimized.latency_ms,
                    "cost": r.optimized.cost,
                    "model": r.optimized.model,
                    "cache_type": r.optimized.cache_type,
                    "similarity": r.optimized.similarity,
                    "context_injected": r.optimized.context_injected,
                    "context_tokens_added": r.optimized.context_tokens_added,
                    "preferences_used": r.optimized.preferences_used,
                    "complexity": r.optimized.complexity,
                    "context_quality_score": r.optimized.context_quality_score,
                    "context_compression_ratio": r.optimized.context_compression_ratio,
                    "query_compression_ratio": r.optimized.query_compression_ratio,
                    "token_allocations": r.optimized.token_allocations,
                    "strategy": r.optimized.strategy,
                    "context_aware_routing": r.optimized.context_aware_routing,
                    "query_compressed": r.optimized.query_compressed,
                    "query_original_tokens": r.optimized.query_original_tokens,
                    "query_compressed_tokens": r.optimized.query_compressed_tokens,
                    "context_compressed": r.optimized.context_compressed,
                    "context_original_tokens": r.optimized.context_original_tokens,
                    "context_compressed_tokens": r.optimized.context_compressed_tokens,
                },
                "comparison_metrics": {
                    "token_savings": r.comparison_metrics.token_savings,
                    "token_savings_percent": r.comparison_metrics.token_savings_percent,
                    "cost_savings": r.comparison_metrics.cost_savings,
                    "cost_savings_percent": r.comparison_metrics.cost_savings_percent,
                    "latency_improvement_ms": r.comparison_metrics.latency_improvement_ms,
                    "latency_improvement_percent": r.comparison_metrics.latency_improvement_percent,
                    "quality_winner": r.comparison_metrics.quality_winner,
                    "quality_confidence": r.comparison_metrics.quality_confidence,
                    "quality_explanation": r.comparison_metrics.quality_explanation,
                },
            }
            
            if r.quality_comparison:
                result_dict["quality_comparison"] = {
                    "winner": r.quality_comparison.winner,
                    "confidence": r.quality_comparison.confidence,
                    "explanation": r.quality_comparison.explanation,
                }
            
            report_data["detailed_results"].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def generate_csv_report(self, filepath: Path):
        """Generate CSV report."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "query_index", "category", "query",
                "baseline_tokens", "baseline_input_tokens", "baseline_output_tokens", "baseline_latency_ms", "baseline_cost", "baseline_model",
                "optimized_tokens", "optimized_input_tokens", "optimized_output_tokens", "optimized_latency_ms", "optimized_cost", "optimized_model",
                "token_savings", "token_savings_percent", "cost_savings", "cost_savings_percent", "latency_improvement_ms", "latency_improvement_percent",
                "cache_type", "similarity", "context_injected", "context_tokens_added", "preferences_used",
                "complexity", "context_quality_score", "context_compression_ratio", "query_compression_ratio",
                "strategy", "context_aware_routing", "query_compressed", "context_compressed",
                "quality_winner", "quality_confidence", "quality_explanation",
                "success", "error"
            ])
            
            # Data rows
            for r in self.results:
                writer.writerow([
                    r.query_index, r.category, r.query,
                    r.baseline.tokens_used, r.baseline.input_tokens, r.baseline.output_tokens, r.baseline.latency_ms, r.baseline.cost, r.baseline.model,
                    r.optimized.tokens_used, r.optimized.input_tokens, r.optimized.output_tokens, r.optimized.latency_ms, r.optimized.cost, r.optimized.model,
                    r.comparison_metrics.token_savings, r.comparison_metrics.token_savings_percent, r.comparison_metrics.cost_savings, r.comparison_metrics.cost_savings_percent,
                    r.comparison_metrics.latency_improvement_ms, r.comparison_metrics.latency_improvement_percent,
                    r.optimized.cache_type, r.optimized.similarity, r.optimized.context_injected, r.optimized.context_tokens_added, r.optimized.preferences_used,
                    r.optimized.complexity, r.optimized.context_quality_score, r.optimized.context_compression_ratio, r.optimized.query_compression_ratio,
                    r.optimized.strategy, r.optimized.context_aware_routing, r.optimized.query_compressed, r.optimized.context_compressed,
                    r.comparison_metrics.quality_winner, r.comparison_metrics.quality_confidence, r.comparison_metrics.quality_explanation,
                    r.success, r.error or ""
                ])


if __name__ == "__main__":
    test = ABComparisonTest()
    test.run_test()







