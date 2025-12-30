"""Usage tracker to document all API calls, tokens, and costs."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
import structlog
import re

logger = structlog.get_logger()


@dataclass
class UsageRecord:
    """Record of a single API call."""
    timestamp: str
    query: str
    response: str
    response_length: int
    tokens_used: int
    latency_ms: float
    cache_hit: bool
    cache_type: str
    strategy: str
    model: str
    cost_estimate: float  # Estimated cost in USD
    quality_score: float = 0.0  # Quality score (0-1)
    quality_metrics: Dict = None  # Detailed quality metrics


class UsageTracker:
    """Track all API usage for documentation."""
    
    def __init__(self, output_file: str = "usage_report.json"):
        """Initialize usage tracker."""
        self.records: List[UsageRecord] = []
        self.output_file = Path(output_file)
        self.start_time = datetime.now()
        
        # Cost estimates (per 1M tokens) - approximate Gemini pricing
        self.cost_per_1M_input_tokens = 0.0  # Free tier
        self.cost_per_1M_output_tokens = 0.0  # Free tier
        
        logger.info("UsageTracker initialized", output_file=output_file)
    
    def compute_quality_metrics(self, query: str, response: str) -> Dict:
        """Compute quality metrics for a response."""
        metrics = {}
        
        # Response length (longer often = more complete)
        metrics['response_length'] = len(response)
        metrics['response_length_score'] = min(1.0, len(response) / 500)  # Normalize to 500 chars
        
        # Word count
        word_count = len(response.split())
        metrics['word_count'] = word_count
        metrics['word_count_score'] = min(1.0, word_count / 100)  # Normalize to 100 words
        
        # Sentence count (more sentences = more structured)
        sentence_count = len(re.split(r'[.!?]+', response))
        metrics['sentence_count'] = sentence_count
        metrics['sentence_count_score'] = min(1.0, sentence_count / 5)  # Normalize to 5 sentences
        
        # Check for common quality indicators
        has_definition = any(word in response.lower() for word in ['is', 'are', 'means', 'refers'])
        has_explanation = any(word in response.lower() for word in ['because', 'since', 'due to', 'explain'])
        has_examples = any(word in response.lower() for word in ['example', 'instance', 'such as', 'like'])
        has_structure = any(marker in response for marker in ['**', '##', '-', '*', '1.', '2.'])
        
        metrics['has_definition'] = has_definition
        metrics['has_explanation'] = has_explanation
        metrics['has_examples'] = has_examples
        metrics['has_structure'] = has_structure
        
        # Quality indicators score
        quality_indicators = sum([has_definition, has_explanation, has_examples, has_structure])
        metrics['quality_indicators_score'] = quality_indicators / 4.0
        
        # Overall quality score (weighted average)
        quality_score = (
            metrics['response_length_score'] * 0.3 +
            metrics['word_count_score'] * 0.2 +
            metrics['sentence_count_score'] * 0.2 +
            metrics['quality_indicators_score'] * 0.3
        )
        metrics['overall_quality_score'] = min(1.0, quality_score)
        
        return metrics
    
    def record_query(
        self,
        query: str,
        response: str,
        tokens_used: int,
        latency_ms: float,
        cache_hit: bool,
        cache_type: str,
        strategy: str,
        model: str,
    ):
        """Record a query execution."""
        # Estimate cost (rough approximation)
        # Assuming 50/50 input/output split for simplicity
        input_tokens = tokens_used // 2
        output_tokens = tokens_used - input_tokens
        cost = (input_tokens * self.cost_per_1M_input_tokens / 1_000_000 +
                output_tokens * self.cost_per_1M_output_tokens / 1_000_000)
        
        # Compute quality metrics
        quality_metrics = self.compute_quality_metrics(query, response)
        quality_score = quality_metrics['overall_quality_score']
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            response_length=len(response),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            cache_type=cache_type or "none",
            strategy=strategy or "none",
            model=model,
            cost_estimate=cost,
            quality_score=quality_score,
            quality_metrics=quality_metrics,
        )
        
        self.records.append(record)
        
        logger.info(
            "Usage recorded",
            query=query[:50],
            tokens=tokens_used,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
        )
    
    def get_summary(self) -> Dict:
        """Get usage summary statistics."""
        if not self.records:
            return {}
        
        total_queries = len(self.records)
        cache_hits = sum(1 for r in self.records if r.cache_hit)
        cache_misses = total_queries - cache_hits
        
        # Calculate tokens: cache hits use 0 tokens, misses use actual tokens
        total_tokens = sum(r.tokens_used for r in self.records)
        tokens_without_cache = sum(r.tokens_used for r in self.records)  # All queries would use tokens without cache
        tokens_with_cache = sum(r.tokens_used for r in self.records if not r.cache_hit)  # Only misses use tokens
        tokens_saved = tokens_without_cache - tokens_with_cache  # Difference is what we saved
        
        total_latency = sum(r.latency_ms for r in self.records)
        avg_latency = total_latency / total_queries if total_queries > 0 else 0
        
        total_cost = sum(r.cost_estimate for r in self.records)
        cost_saved = sum(r.cost_estimate for r in self.records if r.cache_hit)
        
        cache_hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0
        token_savings_rate = (tokens_saved / total_tokens * 100) if total_tokens > 0 else 0
        
        # Quality metrics
        avg_quality = sum(r.quality_score for r in self.records) / total_queries if total_queries > 0 else 0
        cached_quality = sum(r.quality_score for r in self.records if r.cache_hit) / cache_hits if cache_hits > 0 else 0
        non_cached_quality = sum(r.quality_score for r in self.records if not r.cache_hit) / cache_misses if cache_misses > 0 else 0
        
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "total_tokens_used": total_tokens,
            "tokens_with_cache": tokens_with_cache,
            "tokens_saved": tokens_saved,
            "token_savings_rate_percent": round(token_savings_rate, 2),
            "total_latency_ms": round(total_latency, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_cost_estimate_usd": round(total_cost, 6),
            "cost_saved_usd": round(cost_saved, 6),
            "cost_savings_rate_percent": round((cost_saved / total_cost * 100) if total_cost > 0 else 0, 2),
            "average_quality_score": round(avg_quality, 3),
            "cached_quality_score": round(cached_quality, 3),
            "non_cached_quality_score": round(non_cached_quality, 3),
            "quality_difference": round(cached_quality - non_cached_quality, 3),
        }
    
    def save_report(self):
        """Save detailed usage report."""
        report = {
            "summary": self.get_summary(),
            "records": [asdict(r) for r in self.records],
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Usage report saved", file=str(self.output_file))
        return self.output_file
    
    def print_summary(self):
        """Print a formatted summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("USAGE TRACKING SUMMARY")
        print("=" * 80)
        print(f"Start Time: {summary.get('start_time', 'N/A')}")
        print(f"End Time: {summary.get('end_time', 'N/A')}")
        print()
        print("QUERY STATISTICS:")
        print(f"  Total Queries: {summary.get('total_queries', 0)}")
        print(f"  Cache Hits: {summary.get('cache_hits', 0)}")
        print(f"  Cache Misses: {summary.get('cache_misses', 0)}")
        print(f"  Cache Hit Rate: {summary.get('cache_hit_rate_percent', 0)}%")
        print()
        print("TOKEN USAGE:")
        print(f"  Total Tokens Used: {summary.get('total_tokens_used', 0):,}")
        print(f"  Tokens (with cache): {summary.get('tokens_with_cache', 0):,}")
        print(f"  Tokens Saved: {summary.get('tokens_saved', 0):,}")
        print(f"  Token Savings Rate: {summary.get('token_savings_rate_percent', 0)}%")
        print()
        print("PERFORMANCE:")
        print(f"  Total Latency: {summary.get('total_latency_ms', 0):.2f} ms")
        print(f"  Average Latency: {summary.get('average_latency_ms', 0):.2f} ms")
        print()
        print("COST ESTIMATES:")
        print(f"  Total Cost: ${summary.get('total_cost_estimate_usd', 0):.6f}")
        print(f"  Cost Saved: ${summary.get('cost_saved_usd', 0):.6f}")
        print(f"  Cost Savings Rate: {summary.get('cost_savings_rate_percent', 0)}%")
        print()
        print("QUALITY METRICS:")
        print(f"  Average Quality Score: {summary.get('average_quality_score', 0):.3f}/1.0")
        print(f"  Cached Responses Quality: {summary.get('cached_quality_score', 0):.3f}/1.0")
        print(f"  Non-Cached Responses Quality: {summary.get('non_cached_quality_score', 0):.3f}/1.0")
        print(f"  Quality Difference: {summary.get('quality_difference', 0):+.3f}")
        print("=" * 80)

