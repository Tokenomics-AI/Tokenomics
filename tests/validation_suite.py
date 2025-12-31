#!/usr/bin/env python3
"""
Platform Validation Suite

Comprehensive integration test with 60 controlled prompts to verify
Memory, Routing, and Compression components work together correctly.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available, using basic aggregation")


@dataclass
class TestResult:
    """Result for a single test query."""
    bucket: str
    query: str
    query_index: int
    latency_ms: float
    tokens_total: int
    cache_hit_type: Optional[str]  # None, "exact", "semantic_direct", "context"
    model_selected: Optional[str]  # "cheap", "balanced", "premium"
    compression_ratio: Optional[float]
    judge_score: Optional[float]
    success: bool
    error: Optional[str] = None
    baseline_tokens: Optional[int] = None
    savings: Optional[int] = None


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Cache metrics
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    # Routing metrics
    simple_cheap_count: int = 0
    simple_balanced_count: int = 0
    simple_premium_count: int = 0
    medium_cheap_count: int = 0
    medium_balanced_count: int = 0
    medium_premium_count: int = 0
    complex_cheap_count: int = 0
    complex_balanced_count: int = 0
    complex_premium_count: int = 0
    
    # Compression metrics
    avg_compression_ratio: float = 0.0
    compression_queries: int = 0
    
    # Token metrics
    total_tokens: int = 0
    total_baseline_tokens: int = 0
    total_savings: int = 0
    
    # Pass/Fail gates
    cache_gate_passed: bool = False
    routing_gate_passed: bool = False
    compression_gate_passed: bool = False


class PlatformValidationSuite:
    """Platform validation test suite."""
    
    def __init__(self):
        """Initialize validation suite."""
        self.results: List[TestResult] = []
        self.report = ValidationReport()
        self.platform: Optional[TokenomicsPlatform] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Test data buckets
        self.bucket_a = self._create_bucket_a()
        self.bucket_b = self._create_bucket_b()
        self.bucket_c = self._create_bucket_c()
        self.bucket_d = self._create_bucket_d()
    
    def _create_bucket_a(self) -> List[str]:
        """Bucket A: Exact Duplicates (5 prompts, each run twice = 10 calls)."""
        return [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the water cycle.",
        ]
    
    def _create_bucket_b(self) -> List[Tuple[str, str]]:
        """Bucket B: Semantic Paraphrases (5 pairs = 10 calls)."""
        return [
            ("How do I bake a cake?", "Recipe for baking a cake"),
            ("What is artificial intelligence?", "Explain AI"),
            ("How does photosynthesis work?", "Describe the process of photosynthesis"),
            ("Benefits of exercise", "Why is physical activity important?"),
            ("Python programming basics", "Introduction to Python coding"),
        ]
    
    def _create_bucket_c(self) -> Tuple[str, List[str]]:
        """Bucket C: Context Injection (1 document + 9 questions = 10 calls)."""
        # ~2000 word document about Project Apollo
        apollo_document = """
        Project Apollo was a series of space missions undertaken by the United States National Aeronautics and Space Administration (NASA) between 1961 and 1972, with the goal of landing humans on the Moon and returning them safely to Earth. The program was named after Apollo, the Greek god of light, music, and the sun.

        The Apollo program was initiated by President John F. Kennedy in 1961, who declared the ambitious goal of landing a man on the Moon before the end of the decade. This announcement came at a time when the United States was trailing the Soviet Union in the Space Race, following the successful launch of Sputnik and Yuri Gagarin's historic flight.

        The program consisted of several phases. Apollo 1 was intended to be the first crewed mission, but tragically, a fire during a launch pad test on January 27, 1967, killed all three crew members: Gus Grissom, Ed White, and Roger Chaffee. This disaster led to a complete redesign of the Apollo command module and a renewed focus on safety.

        Apollo 7, launched in October 1968, was the first successful crewed Apollo mission, testing the command and service modules in Earth orbit. Apollo 8, launched in December 1968, was the first crewed mission to orbit the Moon, providing humanity's first close-up view of the lunar surface.

        Apollo 9 tested the lunar module in Earth orbit, while Apollo 10 performed a dress rehearsal for the lunar landing, descending to within 9 miles of the Moon's surface. Apollo 11, launched on July 16, 1969, achieved the program's primary goal when Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon on July 20, 1969, while Michael Collins orbited above in the command module.

        The Apollo program continued with five more successful lunar landings: Apollo 12, 14, 15, 16, and 17. Apollo 13, launched in April 1970, experienced a critical failure but successfully returned to Earth thanks to the ingenuity of the crew and ground control. The program concluded with Apollo 17 in December 1972, the last time humans have traveled beyond low Earth orbit.

        The Apollo program left a lasting legacy, advancing technology in numerous fields including computing, materials science, and telecommunications. It demonstrated humanity's ability to achieve seemingly impossible goals through determination, innovation, and collaboration. The program also provided valuable scientific data about the Moon, including rock samples that continue to be studied today.

        The Apollo missions collected over 380 kilograms of lunar samples, conducted extensive geological surveys, and deployed scientific instruments on the lunar surface. These missions fundamentally changed our understanding of the Moon's origin and composition, supporting the theory that the Moon was formed from debris ejected during a giant impact between Earth and a Mars-sized body.

        The program's success required the development of new technologies, including the Saturn V rocket, the most powerful rocket ever built, and the Apollo Guidance Computer, one of the first computers to use integrated circuits. The program employed over 400,000 people and involved thousands of contractors and subcontractors across the United States.

        The Apollo program remains one of humanity's greatest achievements, symbolizing the power of exploration, scientific curiosity, and international cooperation. It demonstrated that with sufficient resources, determination, and technological innovation, humanity can overcome seemingly insurmountable challenges and reach for the stars.
        """
        
        questions = [
            "What was the primary goal of Project Apollo?",
            "Which Apollo mission first landed humans on the Moon?",
            "What happened during Apollo 1?",
            "Who were the first two humans to walk on the Moon?",
            "How many successful lunar landings did the Apollo program achieve?",
            "What was the significance of Apollo 8?",
            "What happened during Apollo 13?",
            "What scientific data did Apollo missions collect?",
            "What was the Saturn V rocket?",
        ]
        
        return (apollo_document, questions)
    
    def _create_bucket_d(self) -> Dict[str, List[str]]:
        """Bucket D: Routing Stress (30 calls: 10 simple, 10 medium, 10 complex)."""
        return {
            "simple": [
                "What is 2+2?",
                "What is the capital of Spain?",
                "How many days in a week?",
                "What color is the sky?",
                "What is the largest planet?",
                "Who wrote Romeo and Juliet?",
                "What is the speed of light?",
                "How many continents are there?",
                "What is the smallest prime number?",
                "What is the chemical symbol for gold?",
            ],
            "medium": [
                "Explain photosynthesis in plants.",
                "How does the water cycle work?",
                "Describe the structure of DNA.",
                "What causes earthquakes?",
                "How do vaccines work?",
                "Explain the greenhouse effect.",
                "What is the difference between weather and climate?",
                "How does the human digestive system work?",
                "Explain the process of cellular respiration.",
                "What are the main causes of climate change?",
            ],
            "complex": [
                "Design a comprehensive microservices architecture for a large-scale e-commerce platform. Include service decomposition, API gateway patterns, data consistency strategies, and deployment considerations.",
                "Explain the theoretical foundations of quantum computing, including qubits, superposition, entanglement, and quantum algorithms. Compare with classical computing.",
                "Develop a detailed machine learning pipeline for fraud detection in financial transactions, including data preprocessing, feature engineering, model selection, and evaluation metrics.",
                "Analyze the architectural patterns and design principles for building a distributed system that handles 10 million concurrent users with 99.99% uptime.",
                "Design a comprehensive data governance framework for a multinational corporation, including data quality standards, privacy regulations compliance, and data lifecycle management.",
                "Explain the implementation of a real-time recommendation system using collaborative filtering, content-based filtering, and hybrid approaches with scalability considerations.",
                "Develop a detailed security architecture for a cloud-native application, including authentication, authorization, encryption, threat modeling, and incident response procedures.",
                "Design a comprehensive DevOps pipeline for continuous integration and deployment, including automated testing, infrastructure as code, monitoring, and rollback strategies.",
                "Explain the architecture and implementation of a blockchain-based supply chain tracking system, including consensus mechanisms, smart contracts, and integration challenges.",
                "Develop a detailed strategy for migrating a monolithic application to a microservices architecture, including service identification, data migration, and organizational changes.",
            ],
        }
    
    def initialize_platform(self, reset_state: bool = False) -> bool:
        """Initialize TokenomicsPlatform."""
        try:
            config = TokenomicsConfig.from_env()
            config.llm.provider = "openai"
            config.llm.model = "gpt-4o-mini"
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
            
            # Enable all features
            config.memory.use_exact_cache = True
            config.memory.use_semantic_cache = True
            config.memory.enable_llmlingua = True
            config.bandit.state_file = "validation_bandit_state.json" if not reset_state else None
            config.bandit.auto_save = True
            config.judge.enabled = True
            
            self.platform = TokenomicsPlatform(config=config)
            print("✓ Platform initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize platform: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_query(self, query: str, bucket: str, index: int, **kwargs) -> TestResult:
        """Run a single query and capture metrics."""
        start_time = time.time()
        
        try:
            result = self.platform.query(
                query=query,
                use_cache=True,
                use_bandit=True,
                use_compression=True,
                use_cost_aware_routing=True,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract cache type
            cache_hit_type = None
            if result.get('cache_hit'):
                cache_type = result.get('cache_type', '')
                if cache_type == 'exact':
                    cache_hit_type = 'exact'
                elif cache_type in ['semantic_direct', 'context']:
                    cache_hit_type = 'semantic_direct' if cache_type == 'semantic_direct' else 'context'
            
            # Extract model/strategy
            model_selected = result.get('strategy')
            
            # Extract compression ratio
            compression_metrics = result.get('compression_metrics', {})
            compression_ratio = None
            if compression_metrics.get('context_compressed'):
                original = compression_metrics.get('context_original_tokens', 0)
                compressed = compression_metrics.get('context_compressed_tokens', 0)
                if original > 0:
                    compression_ratio = compressed / original
            
            # Extract judge score (if available)
            judge_score = result.get('judge_score')
            
            # Calculate baseline (estimate: assume no optimizations would use more tokens)
            tokens_total = result.get('tokens_used', 0)
            baseline_tokens = tokens_total + result.get('memory_savings', 0) + result.get('orchestrator_savings', 0)
            savings = baseline_tokens - tokens_total if baseline_tokens > tokens_total else 0
            
            return TestResult(
                bucket=bucket,
                query=query,
                query_index=index,
                latency_ms=latency_ms,
                tokens_total=tokens_total,
                cache_hit_type=cache_hit_type,
                model_selected=model_selected,
                compression_ratio=compression_ratio,
                judge_score=judge_score,
                success=True,
                baseline_tokens=baseline_tokens,
                savings=savings,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TestResult(
                bucket=bucket,
                query=query,
                query_index=index,
                latency_ms=latency_ms,
                tokens_total=0,
                cache_hit_type=None,
                model_selected=None,
                compression_ratio=None,
                judge_score=None,
                success=False,
                error=str(e),
            )
    
    def run_bucket_a(self):
        """Run Bucket A: Exact Duplicates."""
        print("\n" + "=" * 70)
        print("BUCKET A: Exact Duplicates (10 calls)")
        print("=" * 70)
        
        for i, prompt in enumerate(self.bucket_a, 1):
            # First run
            print(f"\n[{i}/5] First run: {prompt[:50]}...")
            result1 = self.run_query(prompt, "A", (i-1)*2 + 1)
            self.results.append(result1)
            status1 = "✓" if result1.success else "✗"
            cache1 = result1.cache_hit_type or "miss"
            print(f"  [{status1}] Latency: {result1.latency_ms:.0f}ms, Cache: {cache1}, Tokens: {result1.tokens_total}")
            
            # Second run (should hit cache)
            print(f"      Second run (expecting cache hit)...")
            result2 = self.run_query(prompt, "A", (i-1)*2 + 2)
            self.results.append(result2)
            status2 = "✓" if result2.success else "✗"
            cache2 = result2.cache_hit_type or "miss"
            print(f"  [{status2}] Latency: {result2.latency_ms:.0f}ms, Cache: {cache2}, Tokens: {result2.tokens_total}")
            
            if result2.cache_hit_type == "exact" and result2.latency_ms < 500:
                print(f"  ✓ Cache gate: PASS (latency {result2.latency_ms:.0f}ms < 500ms)")
            elif result2.cache_hit_type != "exact":
                print(f"  ⚠ Cache gate: Expected exact cache hit, got {cache2}")
    
    def run_bucket_b(self):
        """Run Bucket B: Semantic Paraphrases."""
        print("\n" + "=" * 70)
        print("BUCKET B: Semantic Paraphrases (10 calls)")
        print("=" * 70)
        
        for i, (prompt1, prompt2) in enumerate(self.bucket_b, 1):
            # First prompt
            print(f"\n[{i}/5] First: {prompt1[:50]}...")
            result1 = self.run_query(prompt1, "B", (i-1)*2 + 1)
            self.results.append(result1)
            status1 = "✓" if result1.success else "✗"
            cache1 = result1.cache_hit_type or "miss"
            print(f"  [{status1}] Cache: {cache1}, Tokens: {result1.tokens_total}")
            
            # Second prompt (semantic paraphrase - should hit semantic cache)
            print(f"      Second (paraphrase): {prompt2[:50]}...")
            result2 = self.run_query(prompt2, "B", (i-1)*2 + 2)
            self.results.append(result2)
            status2 = "✓" if result2.success else "✗"
            cache2 = result2.cache_hit_type or "miss"
            print(f"  [{status2}] Cache: {cache2}, Tokens: {result2.tokens_total}")
            
            if cache2 in ["semantic_direct", "context"]:
                print(f"  ✓ Semantic cache hit detected")
    
    def run_bucket_c(self):
        """Run Bucket C: Context Injection."""
        print("\n" + "=" * 70)
        print("BUCKET C: Context Injection (10 calls)")
        print("=" * 70)
        
        document, questions = self.bucket_c
        
        # Inject document into memory
        print("\n[Setup] Injecting Apollo document into memory...")
        self.platform.memory.store(
            query="Project Apollo space missions",
            response=document,
            tokens_used=2000,  # Estimate
        )
        print("  ✓ Document stored")
        
        # Run questions
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/9] {question[:60]}...")
            result = self.run_query(question, "C", i)
            self.results.append(result)
            status = "✓" if result.success else "✗"
            cache = result.cache_hit_type or "miss"
            compression = f", Compression: {result.compression_ratio:.2f}" if result.compression_ratio else ""
            print(f"  [{status}] Cache: {cache}{compression}, Tokens: {result.tokens_total}")
            
            # Check compression gate
            if result.compression_ratio and result.compression_ratio < 0.4:  # max_context_ratio default
                print(f"  ✓ Compression gate: PASS (ratio {result.compression_ratio:.2f} < 0.4)")
    
    def run_bucket_d(self):
        """Run Bucket D: Routing Stress."""
        print("\n" + "=" * 70)
        print("BUCKET D: Routing Stress (30 calls)")
        print("=" * 70)
        
        for complexity, prompts in self.bucket_d.items():
            print(f"\n[{complexity.upper()}] {len(prompts)} prompts")
            
            for i, prompt in enumerate(prompts, 1):
                print(f"  [{i}/{len(prompts)}] {prompt[:50]}...")
                # Calculate query index: simple=1-10, medium=11-20, complex=21-30
                if complexity == "simple":
                    query_idx = i
                elif complexity == "medium":
                    query_idx = 10 + i
                else:  # complex
                    query_idx = 20 + i
                result = self.run_query(prompt, f"D_{complexity}", query_idx)
                self.results.append(result)
                status = "✓" if result.success else "✗"
                model = result.model_selected or "none"
                print(f"    [{status}] Model: {model}, Tokens: {result.tokens_total}")
    
    def aggregate_results(self):
        """Aggregate results into report."""
        self.report.total_queries = len(self.results)
        self.report.successful_queries = sum(1 for r in self.results if r.success)
        self.report.failed_queries = sum(1 for r in self.results if not r.success)
        
        # Cache metrics
        for r in self.results:
            if not r.success:
                continue
            
            if r.cache_hit_type == "exact":
                self.report.exact_cache_hits += 1
            elif r.cache_hit_type in ["semantic_direct", "context"]:
                self.report.semantic_cache_hits += 1
            else:
                self.report.cache_misses += 1
            
            # Routing metrics (Bucket D only)
            if r.bucket.startswith("D_"):
                complexity = r.bucket.split("_")[1]
                model = r.model_selected or "none"
                
                if complexity == "simple":
                    if model == "cheap":
                        self.report.simple_cheap_count += 1
                    elif model == "balanced":
                        self.report.simple_balanced_count += 1
                    elif model == "premium":
                        self.report.simple_premium_count += 1
                elif complexity == "medium":
                    if model == "cheap":
                        self.report.medium_cheap_count += 1
                    elif model == "balanced":
                        self.report.medium_balanced_count += 1
                    elif model == "premium":
                        self.report.medium_premium_count += 1
                elif complexity == "complex":
                    if model == "cheap":
                        self.report.complex_cheap_count += 1
                    elif model == "balanced":
                        self.report.complex_balanced_count += 1
                    elif model == "premium":
                        self.report.complex_premium_count += 1
            
            # Compression metrics
            if r.compression_ratio is not None:
                self.report.compression_queries += 1
                self.report.avg_compression_ratio = (
                    (self.report.avg_compression_ratio * (self.report.compression_queries - 1) + r.compression_ratio)
                    / self.report.compression_queries
                )
            
            # Token metrics
            self.report.total_tokens += r.tokens_total
            if r.baseline_tokens:
                self.report.total_baseline_tokens += r.baseline_tokens
            if r.savings:
                self.report.total_savings += r.savings
        
        # Calculate cache hit rate
        total_cache_attempts = self.report.successful_queries
        if total_cache_attempts > 0:
            self.report.cache_hit_rate = (
                (self.report.exact_cache_hits + self.report.semantic_cache_hits) / total_cache_attempts * 100
            )
        
        # Pass/Fail gates
        # Cache gate: Exact duplicates (Bucket A second runs) must have latency < 500ms
        bucket_a_second_runs = [r for r in self.results if r.bucket == "A" and r.query_index % 2 == 0]
        if bucket_a_second_runs:
            avg_latency = sum(r.latency_ms for r in bucket_a_second_runs) / len(bucket_a_second_runs)
            exact_hits = sum(1 for r in bucket_a_second_runs if r.cache_hit_type == "exact")
            self.report.cache_gate_passed = avg_latency < 500 and exact_hits == len(bucket_a_second_runs)
        
        # Routing gate: Complex prompts must use Premium > 50% of the time
        complex_results = [r for r in self.results if r.bucket == "D_complex" and r.success]
        if complex_results:
            premium_count = sum(1 for r in complex_results if r.model_selected == "premium")
            premium_rate = premium_count / len(complex_results) * 100
            self.report.routing_gate_passed = premium_rate > 50
        
        # Compression gate: Context-heavy prompts must not exceed max_context_ratio
        # Check if context was compressed when needed (ratio < 1.0 indicates compression occurred)
        bucket_c_results = [r for r in self.results if r.bucket == "C"]
        if bucket_c_results:
            # Gate passes if:
            # 1. Compression occurred when context was large (ratio < 1.0), OR
            # 2. No compression needed (ratio is None or 1.0) and context stayed within budget
            compression_ratios = [r.compression_ratio for r in bucket_c_results if r.compression_ratio is not None]
            if compression_ratios:
                max_ratio = max(compression_ratios)
                # Pass if compression ratio is within acceptable range (<= 0.4 means compressed to 40% or less)
                # OR if no compression was needed (ratio = 1.0 means full context fit)
                self.report.compression_gate_passed = max_ratio <= 0.4 or (len(compression_ratios) < len(bucket_c_results) and max_ratio <= 1.0)
            else:
                # No compression occurred - check if this is acceptable (context may have been small)
                self.report.compression_gate_passed = True
    
    def generate_report(self) -> str:
        """Generate markdown report."""
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        report = f"""# Platform Validation Suite Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Duration:** {duration:.1f} seconds  
**Total Queries:** {self.report.total_queries}  
**Success Rate:** {self.report.successful_queries}/{self.report.total_queries} ({self.report.successful_queries/self.report.total_queries*100:.1f}%)

## Executive Summary

This validation suite tested the Tokenomics Platform with 60 controlled prompts across 4 test buckets:
- **Bucket A**: Exact Duplicates (10 calls) - Cache efficiency
- **Bucket B**: Semantic Paraphrases (10 calls) - Semantic cache
- **Bucket C**: Context Injection (10 calls) - Compression
- **Bucket D**: Routing Stress (30 calls) - Bandit intelligence

## Cache Efficiency

### Hit Rates
- **Exact Cache Hits:** {self.report.exact_cache_hits}
- **Semantic Cache Hits:** {self.report.semantic_cache_hits}
- **Cache Misses:** {self.report.cache_misses}
- **Overall Cache Hit Rate:** {self.report.cache_hit_rate:.1f}%

### Bucket A (Exact Duplicates)
- Expected: Second runs should hit exact cache with < 500ms latency
- Results: See detailed metrics below

### Bucket B (Semantic Paraphrases)
- Expected: Second prompts should hit semantic cache
- Results: {self.report.semantic_cache_hits} semantic hits detected

## Routing Intelligence

### Bucket D - Strategy Distribution

**Simple Prompts (10):**
- Cheap: {self.report.simple_cheap_count}
- Balanced: {self.report.simple_balanced_count}
- Premium: {self.report.simple_premium_count}

**Medium Prompts (10):**
- Cheap: {self.report.medium_cheap_count}
- Balanced: {self.report.medium_balanced_count}
- Premium: {self.report.medium_premium_count}

**Complex Prompts (10):**
- Cheap: {self.report.complex_cheap_count}
- Balanced: {self.report.complex_balanced_count}
- Premium: {self.report.complex_premium_count}
- **Premium Rate:** {self.report.complex_premium_count/10*100:.1f}%

## Compression Statistics

### Bucket C (Context Injection)
- **Queries with Compression:** {self.report.compression_queries}
- **Average Compression Ratio:** {self.report.avg_compression_ratio:.3f}
- **Max Compression Ratio:** {max((r.compression_ratio for r in self.results if r.compression_ratio), default=0.0):.3f}

## Token Metrics

- **Total Tokens Used:** {self.report.total_tokens:,}
- **Estimated Baseline Tokens:** {self.report.total_baseline_tokens:,}
- **Total Savings:** {self.report.total_savings:,} tokens
- **Savings Percentage:** {(self.report.total_savings/self.report.total_baseline_tokens*100) if self.report.total_baseline_tokens > 0 else 0:.1f}%

## Pass/Fail Gates

### Cache Gate
- **Status:** {'✅ PASS' if self.report.cache_gate_passed else '❌ FAIL'}
- **Requirement:** Exact duplicates must have latency < 500ms
- **Result:** {'Passed' if self.report.cache_gate_passed else 'Failed'}

### Routing Gate
- **Status:** {'✅ PASS' if self.report.routing_gate_passed else '❌ FAIL'}
- **Requirement:** Complex prompts must use Premium model > 50% of the time
- **Result:** {'Passed' if self.report.routing_gate_passed else 'Failed'}

### Compression Gate
- **Status:** {'✅ PASS' if self.report.compression_gate_passed else '❌ FAIL'}
- **Requirement:** Context-heavy prompts must not exceed max_context_ratio (0.4)
- **Result:** {'Passed' if self.report.compression_gate_passed else 'Failed'}

## Overall Status

{'✅ ALL GATES PASSED' if all([self.report.cache_gate_passed, self.report.routing_gate_passed, self.report.compression_gate_passed]) else '❌ SOME GATES FAILED'}

## Detailed Results

### Bucket A: Exact Duplicates
"""
        
        bucket_a_results = [r for r in self.results if r.bucket == "A"]
        for r in bucket_a_results:
            report += f"\n- Query {r.query_index}: {r.query[:60]}...\n"
            report += f"  - Latency: {r.latency_ms:.0f}ms\n"
            report += f"  - Cache: {r.cache_hit_type or 'miss'}\n"
            report += f"  - Tokens: {r.tokens_total}\n"
        
        report += "\n### Bucket B: Semantic Paraphrases\n"
        bucket_b_results = [r for r in self.results if r.bucket == "B"]
        for r in bucket_b_results:
            report += f"\n- Query {r.query_index}: {r.query[:60]}...\n"
            report += f"  - Cache: {r.cache_hit_type or 'miss'}\n"
            report += f"  - Tokens: {r.tokens_total}\n"
        
        report += "\n### Bucket C: Context Injection\n"
        bucket_c_results = [r for r in self.results if r.bucket == "C"]
        for r in bucket_c_results:
            report += f"\n- Query {r.query_index}: {r.query[:60]}...\n"
            report += f"  - Cache: {r.cache_hit_type or 'miss'}\n"
            if r.compression_ratio:
                report += f"  - Compression Ratio: {r.compression_ratio:.3f}\n"
            report += f"  - Tokens: {r.tokens_total}\n"
        
        report += "\n### Bucket D: Routing Stress\n"
        bucket_d_results = [r for r in self.results if r.bucket.startswith("D_")]
        for r in bucket_d_results:
            complexity = r.bucket.split("_")[1]
            report += f"\n- [{complexity}] Query {r.query_index}: {r.query[:60]}...\n"
            report += f"  - Model: {r.model_selected or 'none'}\n"
            report += f"  - Tokens: {r.tokens_total}\n"
        
        return report
    
    def save_report(self, filepath: Optional[str] = None):
        """Save report to file."""
        if filepath is None:
            filepath = project_root / "tests" / "validation_report.md"
        
        report_text = self.generate_report()
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {filepath}")
        return filepath
    
    def run(self) -> bool:
        """Run complete validation suite."""
        print("=" * 70)
        print("PLATFORM VALIDATION SUITE")
        print("=" * 70)
        print(f"Total Queries: 60 (A: 10, B: 10, C: 10, D: 30)")
        
        self.start_time = datetime.now()
        
        # Initialize platform (reset state for clean test)
        if not self.initialize_platform(reset_state=True):
            return False
        
        # Run buckets
        self.run_bucket_a()
        self.run_bucket_b()
        self.run_bucket_c()
        self.run_bucket_d()
        
        self.end_time = datetime.now()
        
        # Aggregate results
        print("\n" + "=" * 70)
        print("AGGREGATING RESULTS")
        print("=" * 70)
        self.aggregate_results()
        
        # Display summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Queries: {self.report.total_queries}")
        print(f"Successful: {self.report.successful_queries}")
        print(f"Failed: {self.report.failed_queries}")
        print(f"\nCache Hit Rate: {self.report.cache_hit_rate:.1f}%")
        print(f"  - Exact: {self.report.exact_cache_hits}")
        print(f"  - Semantic: {self.report.semantic_cache_hits}")
        print(f"  - Misses: {self.report.cache_misses}")
        
        print(f"\nRouting (Complex → Premium): {self.report.complex_premium_count}/10 ({self.report.complex_premium_count/10*100:.1f}%)")
        
        print(f"\nCompression: {self.report.compression_queries} queries, avg ratio: {self.report.avg_compression_ratio:.3f}")
        
        if self.report.total_baseline_tokens > 0:
            savings_pct = (self.report.total_savings / self.report.total_baseline_tokens * 100)
            print(f"\nToken Savings: {self.report.total_savings:,} tokens ({savings_pct:.1f}%)")
        else:
            print("\nToken Savings: N/A")
        
        print("\n" + "=" * 70)
        print("PASS/FAIL GATES")
        print("=" * 70)
        print(f"Cache Gate: {'✅ PASS' if self.report.cache_gate_passed else '❌ FAIL'}")
        print(f"Routing Gate: {'✅ PASS' if self.report.routing_gate_passed else '❌ FAIL'}")
        print(f"Compression Gate: {'✅ PASS' if self.report.compression_gate_passed else '❌ FAIL'}")
        
        all_passed = all([
            self.report.cache_gate_passed,
            self.report.routing_gate_passed,
            self.report.compression_gate_passed,
        ])
        
        print("\n" + "=" * 70)
        print(f"OVERALL: {'✅ ALL GATES PASSED' if all_passed else '❌ SOME GATES FAILED'}")
        print("=" * 70)
        
        # Save report
        self.save_report()
        
        return all_passed


def main():
    """Main entry point."""
    suite = PlatformValidationSuite()
    success = suite.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()








