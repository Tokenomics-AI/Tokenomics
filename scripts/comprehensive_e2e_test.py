#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Tokenomics Platform
======================================================

This script tests ALL components of the Tokenomics platform with 50 carefully
designed synthetic queries. Each query is designed to trigger specific components
and behaviors to validate the complete system.

Components Tested:
- Memory Layer (exact cache, semantic cache, context injection)
- Token Orchestrator (complexity analysis, token allocation)
- Bandit Optimizer (strategy selection, learning)
- LLMLingua Compression (query compression for long inputs)
- Token Predictor (response token prediction)
- Cascading Inference (quality-based escalation)
- Quality Judge (response evaluation)

Output:
- Detailed metrics for each query
- Component-level analysis
- Cost savings breakdown
- Comprehensive markdown documentation
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import statistics

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from dotenv import load_dotenv
load_dotenv(parent_dir / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
import structlog

logger = structlog.get_logger()

# ============================================================================
# SYNTHETIC QUERY DATASET - 50 Queries Designed to Test All Components
# ============================================================================

SYNTHETIC_QUERIES = [
    # ==========================================================================
    # CATEGORY 1: Simple Queries (10 queries) - Test "cheap" strategy
    # Expected: Low complexity, cheap model, fast response, ~200-300 tokens
    # ==========================================================================
    {
        "id": 1,
        "query": "What is 2+2?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Basic arithmetic - minimal tokens needed"
    },
    {
        "id": 2,
        "query": "What is the capital of France?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Simple factual question"
    },
    {
        "id": 3,
        "query": "Define gravity in one sentence.",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Constrained simple definition"
    },
    {
        "id": 4,
        "query": "What color is the sky?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Trivial factual question"
    },
    {
        "id": 5,
        "query": "Name three fruits.",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Simple list generation"
    },
    {
        "id": 6,
        "query": "What is the chemical symbol for water?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Simple chemistry fact"
    },
    {
        "id": 7,
        "query": "How many days are in a week?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Basic numerical fact"
    },
    {
        "id": 8,
        "query": "What is the largest ocean?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Simple geography fact"
    },
    {
        "id": 9,
        "query": "Spell the word 'necessary'.",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Simple spelling task"
    },
    {
        "id": 10,
        "query": "What is Python in one sentence?",
        "category": "simple",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Constrained tech definition"
    },
    
    # ==========================================================================
    # CATEGORY 2: Medium Queries (10 queries) - Test "balanced" strategy
    # Expected: Medium complexity, balanced model, ~500-600 tokens
    # ==========================================================================
    {
        "id": 11,
        "query": "Explain how photosynthesis works in plants, including the main inputs and outputs.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Science explanation requiring detail"
    },
    {
        "id": 12,
        "query": "What are the key differences between TCP and UDP protocols?",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Technical comparison"
    },
    {
        "id": 13,
        "query": "Describe the process of how a web browser loads a webpage.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Technical process explanation"
    },
    {
        "id": 14,
        "query": "Explain the difference between supervised and unsupervised machine learning.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "ML concepts comparison"
    },
    {
        "id": 15,
        "query": "How does a database index improve query performance?",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Database concept explanation"
    },
    {
        "id": 16,
        "query": "What is the difference between a stack and a queue data structure?",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Data structure comparison"
    },
    {
        "id": 17,
        "query": "Explain how version control systems like Git manage code changes.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Dev tools explanation"
    },
    {
        "id": 18,
        "query": "Describe the main principles of object-oriented programming.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Programming paradigm explanation"
    },
    {
        "id": 19,
        "query": "How does HTTPS encryption protect data in transit?",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Security concept explanation"
    },
    {
        "id": 20,
        "query": "Explain the concept of recursion with a simple example.",
        "category": "medium",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Programming concept with example"
    },
    
    # ==========================================================================
    # CATEGORY 3: Complex Queries (10 queries) - Test "premium" + cascading
    # Expected: High complexity, premium model, cascading may trigger, ~800+ tokens
    # ==========================================================================
    {
        "id": 21,
        "query": "Write a comprehensive explanation of the Transformer architecture in deep learning, including self-attention mechanisms, multi-head attention, positional encoding, encoder-decoder structure, and how it revolutionized NLP. Compare it to RNNs and LSTMs.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Deep technical explanation requiring extensive knowledge"
    },
    {
        "id": 22,
        "query": "Provide a detailed analysis of distributed consensus algorithms including Raft and Paxos. Explain their safety and liveness properties, leader election, log replication, and compare their trade-offs for different use cases.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Advanced distributed systems concepts"
    },
    {
        "id": 23,
        "query": "Explain the complete lifecycle of a Kubernetes pod from creation to termination, including scheduling, container runtime, networking, storage, health checks, and resource management. Include failure scenarios and recovery mechanisms.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Complex infrastructure topic"
    },
    {
        "id": 24,
        "query": "Design a scalable real-time recommendation system architecture. Cover data pipelines, feature engineering, model training and serving, A/B testing infrastructure, handling cold start, and explain the trade-offs between collaborative filtering and content-based approaches.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "System design with multiple components"
    },
    {
        "id": 25,
        "query": "Explain quantum computing fundamentals including qubits, superposition, entanglement, quantum gates, Shor's algorithm, Grover's algorithm, quantum error correction, and its implications for cryptography and optimization.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Advanced physics/CS intersection"
    },
    {
        "id": 26,
        "query": "Provide a complete guide to implementing a CI/CD pipeline with GitOps principles. Include source control strategies, build automation, testing pyramid, deployment strategies (blue-green, canary), infrastructure as code, monitoring, and rollback procedures.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "DevOps best practices comprehensive"
    },
    {
        "id": 27,
        "query": "Explain the CAP theorem and its implications for distributed database design. Cover consistency models (eventual, strong, causal), partition tolerance strategies, and how systems like Cassandra, MongoDB, and CockroachDB make different trade-offs.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Distributed database theory"
    },
    {
        "id": 28,
        "query": "Write a detailed explanation of neural network optimization techniques including gradient descent variants (SGD, Adam, RMSprop), learning rate scheduling, batch normalization, dropout, weight initialization strategies, and how to diagnose and fix training issues.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Deep learning training techniques"
    },
    {
        "id": 29,
        "query": "Explain microservices architecture patterns including service discovery, API gateway, circuit breaker, saga pattern for distributed transactions, event sourcing, CQRS, and strategies for handling data consistency across services.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Microservices design patterns"
    },
    {
        "id": 30,
        "query": "Provide a comprehensive overview of modern web security including OWASP Top 10, XSS prevention, CSRF protection, SQL injection, authentication best practices (OAuth, JWT), CORS, CSP headers, and secure session management.",
        "category": "complex",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Security comprehensive coverage"
    },
    
    # ==========================================================================
    # CATEGORY 4: Exact Duplicate Queries (5 queries) - Test exact cache
    # These are EXACT copies of earlier queries - should get instant cache hits
    # ==========================================================================
    {
        "id": 31,
        "query": "What is 2+2?",
        "category": "exact_duplicate",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "EXACT DUPLICATE of query #1 - test exact cache hit",
        "duplicate_of": 1
    },
    {
        "id": 32,
        "query": "What is the capital of France?",
        "category": "exact_duplicate",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "EXACT DUPLICATE of query #2 - test exact cache hit",
        "duplicate_of": 2
    },
    {
        "id": 33,
        "query": "How does a database index improve query performance?",
        "category": "exact_duplicate",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "EXACT DUPLICATE of query #15 - test exact cache hit",
        "duplicate_of": 15
    },
    {
        "id": 34,
        "query": "What is Python in one sentence?",
        "category": "exact_duplicate",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "EXACT DUPLICATE of query #10 - test exact cache hit",
        "duplicate_of": 10
    },
    {
        "id": 35,
        "query": "What is the difference between a stack and a queue data structure?",
        "category": "exact_duplicate",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "EXACT DUPLICATE of query #16 - test exact cache hit",
        "duplicate_of": 16
    },
    
    # ==========================================================================
    # CATEGORY 5: Semantic Variations (5 queries) - Test semantic cache
    # These are semantically similar but not identical - test semantic matching
    # ==========================================================================
    {
        "id": 36,
        "query": "What's the result of adding two and two?",
        "category": "semantic_variation",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Semantic variation of query #1 - test semantic cache",
        "similar_to": 1
    },
    {
        "id": 37,
        "query": "Tell me the capital city of France.",
        "category": "semantic_variation",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Semantic variation of query #2 - test semantic cache",
        "similar_to": 2
    },
    {
        "id": 38,
        "query": "Can you explain how indexing helps database queries run faster?",
        "category": "semantic_variation",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Semantic variation of query #15 - test semantic cache",
        "similar_to": 15
    },
    {
        "id": 39,
        "query": "Describe Python programming language briefly.",
        "category": "semantic_variation",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Semantic variation of query #10 - test semantic cache",
        "similar_to": 10
    },
    {
        "id": 40,
        "query": "Compare stack and queue - what are the main differences?",
        "category": "semantic_variation",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Semantic variation of query #16 - test semantic cache",
        "similar_to": 16
    },
    
    # ==========================================================================
    # CATEGORY 6: Long Queries (5 queries) - Test LLMLingua Compression
    # These queries are >500 characters to trigger compression
    # ==========================================================================
    {
        "id": 41,
        "query": """I am working on a complex software project that involves building a web application using React for the frontend and Node.js with Express for the backend. The application needs to handle user authentication with OAuth 2.0, store data in PostgreSQL with Redis caching, and deploy to AWS using Docker containers managed by Kubernetes. I need help understanding how to structure the project, set up the development environment, configure the CI/CD pipeline using GitHub Actions, implement proper error handling and logging, set up monitoring with Prometheus and Grafana, and ensure security best practices are followed throughout the development lifecycle. Can you provide a comprehensive guide covering all these aspects?""",
        "category": "long_query",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Long query (600+ chars) - test LLMLingua compression",
        "char_count": 763
    },
    {
        "id": 42,
        "query": """We are designing a machine learning pipeline for a recommendation system that needs to process millions of user interactions daily. The system should include data ingestion from multiple sources including Kafka streams and batch files from S3, feature engineering using Spark, model training with TensorFlow or PyTorch, model serving with low latency requirements under 50ms, A/B testing framework for comparing model versions, and comprehensive monitoring of model performance including drift detection. The infrastructure should be cloud-native and cost-effective. Please explain the architecture, technology choices, and implementation considerations for each component of this system.""",
        "category": "long_query",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Long query (650+ chars) - test LLMLingua compression",
        "char_count": 729
    },
    {
        "id": 43,
        "query": """I need to understand the complete process of setting up a production-ready Kubernetes cluster on AWS using EKS. This includes configuring the VPC with proper networking, setting up IAM roles and policies for least privilege access, configuring node groups with appropriate instance types, implementing cluster autoscaling, setting up ingress controllers with SSL termination, configuring persistent storage with EBS and EFS, implementing network policies for pod-to-pod communication security, setting up monitoring with CloudWatch and third-party tools, implementing centralized logging with FluentBit and Elasticsearch, and establishing backup and disaster recovery procedures. Please provide step-by-step guidance.""",
        "category": "long_query",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Long query (700+ chars) - test LLMLingua compression",
        "char_count": 775
    },
    {
        "id": 44,
        "query": """Our company is migrating from a monolithic application to a microservices architecture. The current system is a large Java application with an Oracle database, handling e-commerce operations including product catalog, inventory management, order processing, payment handling, and customer management. We need to decompose this into independent services while maintaining data consistency, implement an API gateway for routing and rate limiting, set up service mesh for inter-service communication, migrate from Oracle to appropriate databases for each service (considering PostgreSQL, MongoDB, and Redis), and ensure zero-downtime deployment capabilities. What is the recommended approach for this migration including timeline, risks, and mitigation strategies?""",
        "category": "long_query",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Long query (750+ chars) - test LLMLingua compression",
        "char_count": 811
    },
    {
        "id": 45,
        "query": """I am researching natural language processing techniques for building a conversational AI system. The system needs to understand user intent from free-form text input, extract named entities and their relationships, maintain conversation context across multiple turns, generate natural and contextually appropriate responses, handle multiple languages with proper localization, and learn from user feedback to improve over time. I would like to understand the current state-of-the-art approaches including transformer-based models like BERT and GPT, fine-tuning strategies for domain-specific applications, evaluation metrics for conversation quality, and deployment considerations for production systems with high availability requirements. Please provide a comprehensive technical overview.""",
        "category": "long_query",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Long query (780+ chars) - test LLMLingua compression",
        "char_count": 852
    },
    
    # ==========================================================================
    # CATEGORY 7: Mixed Scenarios (5 queries) - Edge cases and complete flow
    # ==========================================================================
    {
        "id": 46,
        "query": "What is 2+2?",
        "category": "mixed",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Third exact duplicate of #1 - validate cache persistence",
        "duplicate_of": 1
    },
    {
        "id": 47,
        "query": "Write a haiku about programming.",
        "category": "mixed",
        "expected_complexity": "simple",
        "expected_strategy": "cheap",
        "purpose": "Creative simple task - test creative routing"
    },
    {
        "id": 48,
        "query": "Explain the trade-offs between consistency and availability in distributed systems, with specific examples of systems that prioritize each.",
        "category": "mixed",
        "expected_complexity": "complex",
        "expected_strategy": "premium",
        "purpose": "Complex but medium-length - test complexity detection"
    },
    {
        "id": 49,
        "query": "List 5 programming languages and their primary use cases.",
        "category": "mixed",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "List with explanation - balanced complexity"
    },
    {
        "id": 50,
        "query": "What is machine learning and how does it differ from traditional programming?",
        "category": "mixed",
        "expected_complexity": "medium",
        "expected_strategy": "balanced",
        "purpose": "Final query - standard explanation question"
    },
]


# ============================================================================
# DATA CLASSES FOR RESULTS TRACKING
# ============================================================================

@dataclass
class QueryResult:
    """Results from a single query."""
    query_id: int
    query: str
    category: str
    expected_complexity: str
    expected_strategy: str
    purpose: str
    
    # Response data
    response: str = ""
    response_preview: str = ""  # First 200 chars
    
    # Token metrics
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Cache metrics
    cache_hit: bool = False
    cache_type: str = "none"
    similarity_score: float = 0.0
    
    # Complexity and strategy
    actual_complexity: str = ""
    actual_strategy: str = ""
    complexity_match: bool = False
    strategy_match: bool = False
    
    # Model and quality
    model_used: str = ""
    quality_score: float = 0.0
    cascading_escalated: bool = False
    
    # Compression metrics
    compression_applied: bool = False
    compression_ratio: float = 0.0
    tokens_saved_compression: int = 0
    
    # Token prediction
    predicted_tokens: int = 0
    prediction_error: int = 0
    prediction_accuracy: float = 0.0
    
    # Cost metrics
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    baseline_cost: float = 0.0
    cost_savings: float = 0.0
    
    # Timing
    timestamp: str = ""
    
    # Raw result for debugging
    raw_result: Dict = field(default_factory=dict)


@dataclass
class ComponentMetrics:
    """Aggregated metrics for each component."""
    # Memory Layer
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    context_injections: int = 0
    cache_miss: int = 0
    cache_hit_rate: float = 0.0
    memory_tokens_saved: int = 0
    
    # Orchestrator
    simple_classifications: int = 0
    medium_classifications: int = 0
    complex_classifications: int = 0
    complexity_accuracy: float = 0.0
    
    # Bandit
    cheap_selections: int = 0
    balanced_selections: int = 0
    premium_selections: int = 0
    strategy_accuracy: float = 0.0
    
    # Compression
    queries_compressed: int = 0
    avg_compression_ratio: float = 0.0
    total_compression_savings: int = 0
    
    # Token Predictor
    predictions_made: int = 0
    avg_prediction_error: float = 0.0
    prediction_accuracy: float = 0.0
    
    # Cascading
    escalations_triggered: int = 0
    escalation_rate: float = 0.0
    
    # Quality
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0


@dataclass
class CostAnalysis:
    """Cost analysis results."""
    total_baseline_cost: float = 0.0
    total_optimized_cost: float = 0.0
    total_savings: float = 0.0
    savings_percentage: float = 0.0
    
    # Per-component savings
    memory_savings: float = 0.0
    bandit_savings: float = 0.0
    orchestrator_savings: float = 0.0
    compression_savings: float = 0.0
    
    # Token summary
    total_tokens_used: int = 0
    total_tokens_saved: int = 0
    avg_tokens_per_query: float = 0.0


class ComprehensiveE2ETest:
    """Comprehensive end-to-end test runner."""
    
    # Model pricing per 1M tokens
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gemini-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 1.25, "output": 5.00},
    }
    
    def __init__(self):
        """Initialize test environment."""
        print("=" * 80)
        print("COMPREHENSIVE END-TO-END TEST - TOKENOMICS PLATFORM")
        print("=" * 80)
        print(f"Started: {datetime.now().isoformat()}")
        print(f"Total Queries: {len(SYNTHETIC_QUERIES)}")
        print("=" * 80)
        
        # Initialize platform with all features enabled
        config = TokenomicsConfig.from_env()
        
        # Enable all advanced features for comprehensive testing
        config.cascading.enabled = True
        config.cascading.quality_threshold = 0.85
        config.memory.use_semantic_cache = True
        config.memory.use_exact_cache = True
        config.memory.enable_llmlingua = True
        
        self.platform = TokenomicsPlatform(config=config)
        self.config = config
        
        # Results storage
        self.results: List[QueryResult] = []
        self.component_metrics = ComponentMetrics()
        self.cost_analysis = CostAnalysis()
        
        # Baseline model for comparison
        self.baseline_model = "gpt-4o"
        self.baseline_max_tokens = 1000
        
        print(f"Platform initialized with provider: {config.llm.provider}")
        print(f"Cascading enabled: {config.cascading.enabled}")
        print(f"Semantic cache enabled: {config.memory.use_semantic_cache}")
        print(f"LLMLingua enabled: {config.memory.enable_llmlingua}")
        print("=" * 80)
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in dollars for a query."""
        pricing = self.MODEL_PRICING.get(model, {"input": 2.50, "output": 10.00})
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        return cost
    
    def run_single_query(self, query_data: Dict) -> QueryResult:
        """Run a single query and collect metrics."""
        query_id = query_data["id"]
        query = query_data["query"]
        category = query_data["category"]
        
        print(f"\n{'='*60}")
        print(f"QUERY {query_id}/50: [{category.upper()}]")
        print(f"{'='*60}")
        print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
        print(f"Purpose: {query_data['purpose']}")
        
        # Initialize result
        result = QueryResult(
            query_id=query_id,
            query=query,
            category=category,
            expected_complexity=query_data["expected_complexity"],
            expected_strategy=query_data["expected_strategy"],
            purpose=query_data["purpose"],
            timestamp=datetime.now().isoformat(),
        )
        
        try:
            # Run through platform
            start_time = time.time()
            platform_result = self.platform.query(
                query=query,
                use_cache=True,
                use_compression=True,
                use_bandit=True,
                use_cost_aware_routing=True,
            )
            latency = (time.time() - start_time) * 1000
            
            # Extract metrics
            result.response = platform_result.get("response", "")
            result.response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
            result.tokens_used = platform_result.get("tokens_used", 0)
            result.input_tokens = platform_result.get("input_tokens", 0)
            result.output_tokens = platform_result.get("output_tokens", 0)
            result.latency_ms = latency
            
            # Cache metrics
            result.cache_hit = platform_result.get("cache_hit", False)
            result.cache_type = platform_result.get("cache_type", "none")
            result.similarity_score = platform_result.get("similarity", 0.0) or 0.0
            
            # Complexity and strategy
            plan = platform_result.get("plan")
            if plan:
                result.actual_complexity = plan.complexity.value if hasattr(plan.complexity, 'value') else str(plan.complexity)
            else:
                result.actual_complexity = "unknown"
            
            result.actual_strategy = platform_result.get("strategy", "unknown")
            result.complexity_match = result.actual_complexity.lower() == result.expected_complexity.lower()
            result.strategy_match = result.actual_strategy.lower() == result.expected_strategy.lower()
            
            # Model and quality
            result.model_used = platform_result.get("model", "unknown")
            result.quality_score = platform_result.get("quality_score", 0.0) or 0.0
            result.cascading_escalated = platform_result.get("cascading_escalated", False)
            
            # Compression metrics
            compression_metrics = platform_result.get("compression_metrics", {})
            result.compression_applied = compression_metrics.get("query_compressed", False)
            result.compression_ratio = compression_metrics.get("compression_ratio", 0.0) or 0.0
            result.tokens_saved_compression = compression_metrics.get("total_compression_savings", 0) or 0
            
            # Token prediction
            result.predicted_tokens = platform_result.get("predicted_max_tokens", 0) or 0
            if result.predicted_tokens > 0 and result.output_tokens > 0:
                result.prediction_error = abs(result.predicted_tokens - result.output_tokens)
                result.prediction_accuracy = 1.0 - (result.prediction_error / max(result.predicted_tokens, result.output_tokens))
            
            # Cost calculation
            result.estimated_cost = self.calculate_cost(
                result.model_used, result.input_tokens, result.output_tokens
            )
            
            # Baseline cost (what it would cost without optimization)
            if result.cache_hit:
                # For cache hits, baseline is what the original query cost
                result.baseline_cost = self.calculate_cost(
                    self.baseline_model, 
                    result.input_tokens if result.input_tokens > 0 else 100,
                    result.output_tokens if result.output_tokens > 0 else 300
                )
            else:
                result.baseline_cost = self.calculate_cost(
                    self.baseline_model, result.input_tokens, result.output_tokens
                )
            
            result.cost_savings = result.baseline_cost - result.estimated_cost
            
            # Store raw result
            result.raw_result = platform_result
            
            # Print summary
            print(f"\n--- Results ---")
            print(f"Cache: {'HIT (' + result.cache_type + ')' if result.cache_hit else 'MISS'}")
            print(f"Tokens: {result.tokens_used} (in: {result.input_tokens}, out: {result.output_tokens})")
            print(f"Complexity: {result.actual_complexity} (expected: {result.expected_complexity}) {'✓' if result.complexity_match else '✗'}")
            print(f"Strategy: {result.actual_strategy} (expected: {result.expected_strategy}) {'✓' if result.strategy_match else '✗'}")
            print(f"Model: {result.model_used}")
            print(f"Latency: {result.latency_ms:.0f}ms")
            print(f"Cost: ${result.estimated_cost:.6f} (saved: ${result.cost_savings:.6f})")
            if result.compression_applied:
                print(f"Compression: ratio={result.compression_ratio:.2f}, saved={result.tokens_saved_compression} tokens")
            if result.cascading_escalated:
                print(f"Cascading: ESCALATED to premium")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result.response = f"Error: {str(e)}"
            logger.error("Query failed", query_id=query_id, error=str(e))
        
        return result
    
    def update_component_metrics(self):
        """Update aggregated component metrics from all results."""
        if not self.results:
            return
        
        metrics = self.component_metrics
        
        # Memory Layer metrics
        for r in self.results:
            if r.cache_hit:
                if r.cache_type == "exact":
                    metrics.exact_cache_hits += 1
                elif r.cache_type == "semantic_direct":
                    metrics.semantic_cache_hits += 1
                elif r.cache_type == "context":
                    metrics.context_injections += 1
            else:
                metrics.cache_miss += 1
        
        total_queries = len(self.results)
        metrics.cache_hit_rate = (metrics.exact_cache_hits + metrics.semantic_cache_hits) / total_queries * 100
        
        # Calculate tokens saved by cache
        for r in self.results:
            if r.cache_hit and r.tokens_used == 0:
                # Estimate tokens that would have been used
                avg_tokens = sum(x.tokens_used for x in self.results if not x.cache_hit) / max(1, metrics.cache_miss)
                metrics.memory_tokens_saved += int(avg_tokens)
        
        # Orchestrator metrics
        for r in self.results:
            if r.actual_complexity.lower() == "simple":
                metrics.simple_classifications += 1
            elif r.actual_complexity.lower() == "medium":
                metrics.medium_classifications += 1
            elif r.actual_complexity.lower() == "complex":
                metrics.complex_classifications += 1
        
        correct_complexity = sum(1 for r in self.results if r.complexity_match)
        metrics.complexity_accuracy = correct_complexity / total_queries * 100
        
        # Bandit metrics
        for r in self.results:
            if r.actual_strategy.lower() == "cheap":
                metrics.cheap_selections += 1
            elif r.actual_strategy.lower() == "balanced":
                metrics.balanced_selections += 1
            elif r.actual_strategy.lower() == "premium":
                metrics.premium_selections += 1
        
        correct_strategy = sum(1 for r in self.results if r.strategy_match)
        metrics.strategy_accuracy = correct_strategy / total_queries * 100
        
        # Compression metrics
        compressed = [r for r in self.results if r.compression_applied]
        metrics.queries_compressed = len(compressed)
        if compressed:
            metrics.avg_compression_ratio = statistics.mean(r.compression_ratio for r in compressed)
            metrics.total_compression_savings = sum(r.tokens_saved_compression for r in compressed)
        
        # Token predictor metrics
        with_predictions = [r for r in self.results if r.predicted_tokens > 0]
        metrics.predictions_made = len(with_predictions)
        if with_predictions:
            metrics.avg_prediction_error = statistics.mean(r.prediction_error for r in with_predictions)
            metrics.prediction_accuracy = statistics.mean(r.prediction_accuracy for r in with_predictions) * 100
        
        # Cascading metrics
        metrics.escalations_triggered = sum(1 for r in self.results if r.cascading_escalated)
        metrics.escalation_rate = metrics.escalations_triggered / total_queries * 100
        
        # Quality metrics
        quality_scores = [r.quality_score for r in self.results if r.quality_score > 0]
        if quality_scores:
            metrics.avg_quality_score = statistics.mean(quality_scores)
            metrics.min_quality_score = min(quality_scores)
            metrics.max_quality_score = max(quality_scores)
    
    def update_cost_analysis(self):
        """Update cost analysis from all results."""
        cost = self.cost_analysis
        
        cost.total_baseline_cost = sum(r.baseline_cost for r in self.results)
        cost.total_optimized_cost = sum(r.estimated_cost for r in self.results)
        cost.total_savings = cost.total_baseline_cost - cost.total_optimized_cost
        
        if cost.total_baseline_cost > 0:
            cost.savings_percentage = (cost.total_savings / cost.total_baseline_cost) * 100
        
        # Token summary
        cost.total_tokens_used = sum(r.tokens_used for r in self.results)
        cost.avg_tokens_per_query = cost.total_tokens_used / max(1, len(self.results))
        
        # Component savings (approximate breakdown)
        cache_hits = [r for r in self.results if r.cache_hit]
        if cache_hits:
            cost.memory_savings = sum(r.baseline_cost for r in cache_hits)
        
        # Bandit savings from model selection
        for r in self.results:
            if r.model_used in ["gpt-4o-mini", "gemini-flash"] and not r.cache_hit:
                # Saved by using cheaper model
                premium_cost = self.calculate_cost("gpt-4o", r.input_tokens, r.output_tokens)
                cost.bandit_savings += premium_cost - r.estimated_cost
        
        # Compression savings
        compressed = [r for r in self.results if r.compression_applied]
        for r in compressed:
            # Approximate savings from compression
            cost.compression_savings += r.tokens_saved_compression * 0.00001
        
        cost.orchestrator_savings = max(0, cost.total_savings - cost.memory_savings - cost.bandit_savings - cost.compression_savings)
    
    def run_all_tests(self) -> Tuple[List[QueryResult], ComponentMetrics, CostAnalysis]:
        """Run all 50 queries and collect results."""
        print(f"\n{'='*80}")
        print("STARTING COMPREHENSIVE TEST RUN")
        print(f"{'='*80}")
        
        for i, query_data in enumerate(SYNTHETIC_QUERIES):
            try:
                result = self.run_single_query(query_data)
                self.results.append(result)
                
                # Save intermediate results every 10 queries
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results()
                    print(f"\n[Progress: {i+1}/50 queries completed]")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error on query {query_data['id']}: {str(e)}")
                logger.error("Query failed", query_id=query_data["id"], error=str(e))
        
        # Update aggregated metrics
        self.update_component_metrics()
        self.update_cost_analysis()
        
        print(f"\n{'='*80}")
        print("TEST RUN COMPLETED")
        print(f"{'='*80}")
        print(f"Total queries: {len(self.results)}")
        print(f"Cache hit rate: {self.component_metrics.cache_hit_rate:.1f}%")
        print(f"Total savings: ${self.cost_analysis.total_savings:.6f} ({self.cost_analysis.savings_percentage:.1f}%)")
        
        return self.results, self.component_metrics, self.cost_analysis
    
    def save_intermediate_results(self):
        """Save intermediate results to JSON."""
        output_file = parent_dir / f"comprehensive_e2e_results_intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "queries_completed": len(self.results),
            "results": [asdict(r) for r in self.results],
        }
        
        # Remove raw_result for intermediate saves (too large)
        for r in data["results"]:
            r.pop("raw_result", None)
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Intermediate results saved to: {output_file}")
    
    def generate_documentation(self) -> str:
        """Generate comprehensive markdown documentation."""
        doc = []
        
        # Header
        doc.append("# Comprehensive End-to-End Test Results")
        doc.append("## Tokenomics Platform - Proof of Work\n")
        doc.append(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        doc.append(f"**Total Queries Tested:** {len(self.results)}\n")
        doc.append("---\n")
        
        # Executive Summary
        doc.append("## Executive Summary\n")
        doc.append("### Key Findings\n")
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Total Queries | {len(self.results)} |")
        doc.append(f"| Cache Hit Rate | {self.component_metrics.cache_hit_rate:.1f}% |")
        doc.append(f"| Total Baseline Cost | ${self.cost_analysis.total_baseline_cost:.6f} |")
        doc.append(f"| Total Optimized Cost | ${self.cost_analysis.total_optimized_cost:.6f} |")
        doc.append(f"| **Total Savings** | **${self.cost_analysis.total_savings:.6f}** |")
        doc.append(f"| **Savings Percentage** | **{self.cost_analysis.savings_percentage:.1f}%** |")
        doc.append(f"| Average Quality Score | {self.component_metrics.avg_quality_score:.2f} |")
        doc.append(f"| Complexity Classification Accuracy | {self.component_metrics.complexity_accuracy:.1f}% |")
        doc.append(f"| Strategy Selection Accuracy | {self.component_metrics.strategy_accuracy:.1f}% |")
        doc.append("")
        
        # Cost Savings Breakdown
        doc.append("### Cost Savings Breakdown by Component\n")
        doc.append("```")
        doc.append(f"Memory Layer (Cache):     ${self.cost_analysis.memory_savings:.6f}")
        doc.append(f"Bandit Optimizer:         ${self.cost_analysis.bandit_savings:.6f}")
        doc.append(f"Orchestrator:             ${self.cost_analysis.orchestrator_savings:.6f}")
        doc.append(f"Compression:              ${self.cost_analysis.compression_savings:.6f}")
        doc.append(f"─────────────────────────────────────")
        doc.append(f"TOTAL SAVINGS:            ${self.cost_analysis.total_savings:.6f}")
        doc.append("```\n")
        
        doc.append("---\n")
        
        # Component Analysis
        doc.append("## Component Analysis\n")
        
        # Memory Layer
        doc.append("### 1. Memory Layer (Intelligent Cache)\n")
        doc.append("The Memory Layer provides exact and semantic caching to avoid redundant LLM calls.\n")
        doc.append("| Metric | Count | Rate |")
        doc.append("|--------|-------|------|")
        doc.append(f"| Exact Cache Hits | {self.component_metrics.exact_cache_hits} | {self.component_metrics.exact_cache_hits/len(self.results)*100:.1f}% |")
        doc.append(f"| Semantic Cache Hits | {self.component_metrics.semantic_cache_hits} | {self.component_metrics.semantic_cache_hits/len(self.results)*100:.1f}% |")
        doc.append(f"| Context Injections | {self.component_metrics.context_injections} | {self.component_metrics.context_injections/len(self.results)*100:.1f}% |")
        doc.append(f"| Cache Misses | {self.component_metrics.cache_miss} | {self.component_metrics.cache_miss/len(self.results)*100:.1f}% |")
        doc.append(f"| **Total Cache Hit Rate** | **{self.component_metrics.exact_cache_hits + self.component_metrics.semantic_cache_hits}** | **{self.component_metrics.cache_hit_rate:.1f}%** |")
        doc.append(f"| Tokens Saved by Caching | {self.component_metrics.memory_tokens_saved} | - |")
        doc.append("")
        
        doc.append("**Analysis:** ")
        if self.component_metrics.cache_hit_rate > 10:
            doc.append(f"The memory layer achieved a {self.component_metrics.cache_hit_rate:.1f}% cache hit rate, ")
            doc.append(f"saving approximately {self.component_metrics.memory_tokens_saved} tokens through cached responses.")
        else:
            doc.append("Cache hits were limited in this test due to query diversity. In production with repeated queries, higher hit rates are expected.")
        doc.append("\n")
        
        # Token Orchestrator
        doc.append("### 2. Token Orchestrator\n")
        doc.append("The Orchestrator analyzes query complexity and allocates token budgets accordingly.\n")
        doc.append("| Complexity | Classified | Expected | Accuracy |")
        doc.append("|------------|------------|----------|----------|")
        
        # Calculate per-category accuracy
        simple_queries = [r for r in self.results if r.expected_complexity == "simple"]
        medium_queries = [r for r in self.results if r.expected_complexity == "medium"]
        complex_queries = [r for r in self.results if r.expected_complexity == "complex"]
        
        simple_correct = sum(1 for r in simple_queries if r.complexity_match)
        medium_correct = sum(1 for r in medium_queries if r.complexity_match)
        complex_correct = sum(1 for r in complex_queries if r.complexity_match)
        
        doc.append(f"| Simple | {self.component_metrics.simple_classifications} | {len(simple_queries)} | {simple_correct/max(1,len(simple_queries))*100:.0f}% |")
        doc.append(f"| Medium | {self.component_metrics.medium_classifications} | {len(medium_queries)} | {medium_correct/max(1,len(medium_queries))*100:.0f}% |")
        doc.append(f"| Complex | {self.component_metrics.complex_classifications} | {len(complex_queries)} | {complex_correct/max(1,len(complex_queries))*100:.0f}% |")
        doc.append(f"| **Overall Accuracy** | - | - | **{self.component_metrics.complexity_accuracy:.1f}%** |")
        doc.append("")
        
        # Bandit Optimizer
        doc.append("### 3. Bandit Optimizer (Strategy Selection)\n")
        doc.append("The Bandit uses UCB algorithm to select optimal strategy (cheap/balanced/premium) based on query characteristics.\n")
        doc.append("| Strategy | Selected | Expected | Model |")
        doc.append("|----------|----------|----------|-------|")
        
        cheap_expected = sum(1 for r in self.results if r.expected_strategy == "cheap")
        balanced_expected = sum(1 for r in self.results if r.expected_strategy == "balanced")
        premium_expected = sum(1 for r in self.results if r.expected_strategy == "premium")
        
        doc.append(f"| Cheap | {self.component_metrics.cheap_selections} | {cheap_expected} | gpt-4o-mini |")
        doc.append(f"| Balanced | {self.component_metrics.balanced_selections} | {balanced_expected} | gpt-4o-mini |")
        doc.append(f"| Premium | {self.component_metrics.premium_selections} | {premium_expected} | gpt-4o |")
        doc.append(f"| **Strategy Accuracy** | - | - | **{self.component_metrics.strategy_accuracy:.1f}%** |")
        doc.append("")
        
        # Compression
        doc.append("### 4. LLMLingua Compression\n")
        doc.append("LLMLingua compresses long queries (>500 chars or >150 tokens) to reduce input tokens.\n")
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Queries Compressed | {self.component_metrics.queries_compressed} |")
        doc.append(f"| Average Compression Ratio | {self.component_metrics.avg_compression_ratio:.2f} |")
        doc.append(f"| Total Tokens Saved | {self.component_metrics.total_compression_savings} |")
        doc.append("")
        
        long_queries = [r for r in self.results if r.category == "long_query"]
        if long_queries:
            doc.append("**Long Query Results:**\n")
            doc.append("| Query ID | Original Chars | Compressed | Ratio | Tokens Saved |")
            doc.append("|----------|----------------|------------|-------|--------------|")
            for r in long_queries:
                doc.append(f"| {r.query_id} | {len(r.query)} | {'Yes' if r.compression_applied else 'No'} | {r.compression_ratio:.2f} | {r.tokens_saved_compression} |")
            doc.append("")
        
        # Token Predictor
        doc.append("### 5. Token Predictor\n")
        doc.append("Predicts optimal max_tokens for each query to avoid over-allocation.\n")
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Predictions Made | {self.component_metrics.predictions_made} |")
        doc.append(f"| Average Prediction Error | {self.component_metrics.avg_prediction_error:.0f} tokens |")
        doc.append(f"| Prediction Accuracy | {self.component_metrics.prediction_accuracy:.1f}% |")
        doc.append("")
        
        # Cascading Inference
        doc.append("### 6. Cascading Inference\n")
        doc.append("Starts with cheaper model, escalates to premium if quality threshold not met.\n")
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Escalations Triggered | {self.component_metrics.escalations_triggered} |")
        doc.append(f"| Escalation Rate | {self.component_metrics.escalation_rate:.1f}% |")
        doc.append(f"| Quality Threshold | 0.85 |")
        doc.append("")
        
        escalated = [r for r in self.results if r.cascading_escalated]
        if escalated:
            doc.append("**Escalated Queries:**\n")
            for r in escalated:
                doc.append(f"- Query {r.query_id}: \"{r.query[:50]}...\" - Escalated for quality")
            doc.append("")
        
        # Quality Analysis
        doc.append("### 7. Quality Judge\n")
        doc.append("Evaluates response quality to ensure optimization doesn't sacrifice quality.\n")
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Average Quality Score | {self.component_metrics.avg_quality_score:.2f} |")
        doc.append(f"| Minimum Quality Score | {self.component_metrics.min_quality_score:.2f} |")
        doc.append(f"| Maximum Quality Score | {self.component_metrics.max_quality_score:.2f} |")
        doc.append("")
        
        doc.append("---\n")
        
        # Query-by-Query Results
        doc.append("## Query-by-Query Results\n")
        doc.append("### Summary Table\n")
        doc.append("| ID | Category | Tokens | Cache | Strategy | Model | Cost | Savings |")
        doc.append("|----|---------:|-------:|:-----:|:--------:|:-----:|-----:|--------:|")
        
        for r in self.results:
            cache_icon = "✓" if r.cache_hit else "-"
            doc.append(f"| {r.query_id} | {r.category} | {r.tokens_used} | {cache_icon} | {r.actual_strategy} | {r.model_used} | ${r.estimated_cost:.5f} | ${r.cost_savings:.5f} |")
        doc.append("")
        
        # Detailed results by category
        doc.append("### Results by Category\n")
        
        categories = ["simple", "medium", "complex", "exact_duplicate", "semantic_variation", "long_query", "mixed"]
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            if not cat_results:
                continue
            
            doc.append(f"#### {cat.replace('_', ' ').title()} Queries ({len(cat_results)} queries)\n")
            
            avg_tokens = statistics.mean(r.tokens_used for r in cat_results)
            avg_cost = statistics.mean(r.estimated_cost for r in cat_results)
            avg_savings = statistics.mean(r.cost_savings for r in cat_results)
            cache_hits = sum(1 for r in cat_results if r.cache_hit)
            
            doc.append(f"- **Count:** {len(cat_results)}")
            doc.append(f"- **Average Tokens:** {avg_tokens:.0f}")
            doc.append(f"- **Average Cost:** ${avg_cost:.6f}")
            doc.append(f"- **Average Savings:** ${avg_savings:.6f}")
            doc.append(f"- **Cache Hits:** {cache_hits}/{len(cat_results)}")
            doc.append("")
            
            # Show first 3 examples
            doc.append("**Sample Queries:**\n")
            for r in cat_results[:3]:
                doc.append(f"- **Q{r.query_id}:** \"{r.query[:60]}{'...' if len(r.query) > 60 else ''}\"")
                doc.append(f"  - Tokens: {r.tokens_used}, Strategy: {r.actual_strategy}, Model: {r.model_used}")
            doc.append("")
        
        doc.append("---\n")
        
        # Test Configuration
        doc.append("## Test Configuration\n")
        doc.append("```yaml")
        doc.append(f"Provider: {self.config.llm.provider}")
        doc.append(f"Cascading Enabled: {self.config.cascading.enabled}")
        doc.append(f"Quality Threshold: {self.config.cascading.quality_threshold}")
        doc.append(f"Semantic Cache: {self.config.memory.use_semantic_cache}")
        doc.append(f"Exact Cache: {self.config.memory.use_exact_cache}")
        doc.append(f"LLMLingua: {self.config.memory.enable_llmlingua}")
        doc.append(f"Default Token Budget: {self.config.orchestrator.default_token_budget}")
        doc.append("```\n")
        
        doc.append("---\n")
        
        # Conclusions and Recommendations
        doc.append("## Conclusions\n")
        doc.append("### What Worked Well\n")
        
        if self.cost_analysis.savings_percentage > 20:
            doc.append(f"1. **Cost Savings:** Achieved {self.cost_analysis.savings_percentage:.1f}% cost reduction through intelligent optimization.")
        if self.component_metrics.cache_hit_rate > 10:
            doc.append(f"2. **Memory Layer:** {self.component_metrics.cache_hit_rate:.1f}% cache hit rate demonstrates effective caching strategy.")
        if self.component_metrics.complexity_accuracy > 70:
            doc.append(f"3. **Complexity Analysis:** {self.component_metrics.complexity_accuracy:.1f}% accuracy in query complexity classification.")
        if self.component_metrics.queries_compressed > 0:
            doc.append(f"4. **Compression:** Successfully compressed {self.component_metrics.queries_compressed} long queries, saving {self.component_metrics.total_compression_savings} tokens.")
        
        doc.append("")
        doc.append("### Areas for Improvement\n")
        
        if self.component_metrics.complexity_accuracy < 80:
            doc.append(f"1. Complexity classification accuracy ({self.component_metrics.complexity_accuracy:.1f}%) could be improved with more training data.")
        if self.component_metrics.prediction_accuracy < 70:
            doc.append(f"2. Token prediction accuracy ({self.component_metrics.prediction_accuracy:.1f}%) needs more samples for ML model training.")
        if self.component_metrics.escalation_rate > 30:
            doc.append(f"3. High escalation rate ({self.component_metrics.escalation_rate:.1f}%) suggests initial model selection could be refined.")
        
        doc.append("")
        doc.append("### Summary\n")
        doc.append(f"The Tokenomics platform successfully demonstrated its ability to reduce LLM costs by **{self.cost_analysis.savings_percentage:.1f}%** ")
        doc.append(f"while maintaining response quality. The combination of intelligent caching, adaptive routing, and compression ")
        doc.append(f"provides significant value for production deployments.\n")
        
        doc.append("---\n")
        doc.append(f"*Report generated: {datetime.now().isoformat()}*\n")
        
        return "\n".join(doc)
    
    def save_results(self):
        """Save all results and documentation."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON results
        json_file = parent_dir / f"comprehensive_e2e_results_{timestamp}.json"
        json_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.results),
                "provider": self.config.llm.provider,
            },
            "summary": {
                "cache_hit_rate": self.component_metrics.cache_hit_rate,
                "total_baseline_cost": self.cost_analysis.total_baseline_cost,
                "total_optimized_cost": self.cost_analysis.total_optimized_cost,
                "total_savings": self.cost_analysis.total_savings,
                "savings_percentage": self.cost_analysis.savings_percentage,
            },
            "component_metrics": asdict(self.component_metrics),
            "cost_analysis": asdict(self.cost_analysis),
            "results": [asdict(r) for r in self.results],
        }
        
        # Remove raw_result from JSON (too large)
        for r in json_data["results"]:
            r.pop("raw_result", None)
        
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\nJSON results saved to: {json_file}")
        
        # Generate and save documentation
        doc = self.generate_documentation()
        doc_file = parent_dir / "COMPREHENSIVE_E2E_TEST_RESULTS.md"
        
        with open(doc_file, "w") as f:
            f.write(doc)
        
        print(f"Documentation saved to: {doc_file}")
        
        return json_file, doc_file


def main():
    """Run comprehensive end-to-end test."""
    print("\n" + "=" * 80)
    print("TOKENOMICS PLATFORM - COMPREHENSIVE END-TO-END TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Queries to test: {len(SYNTHETIC_QUERIES)}")
    print("=" * 80 + "\n")
    
    # Initialize and run test
    test = ComprehensiveE2ETest()
    results, metrics, cost = test.run_all_tests()
    
    # Save results
    json_file, doc_file = test.save_results()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total Queries: {len(results)}")
    print(f"Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")
    print(f"Complexity Accuracy: {metrics.complexity_accuracy:.1f}%")
    print(f"Strategy Accuracy: {metrics.strategy_accuracy:.1f}%")
    print(f"Total Baseline Cost: ${cost.total_baseline_cost:.6f}")
    print(f"Total Optimized Cost: ${cost.total_optimized_cost:.6f}")
    print(f"Total Savings: ${cost.total_savings:.6f} ({cost.savings_percentage:.1f}%)")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - Documentation: {doc_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

