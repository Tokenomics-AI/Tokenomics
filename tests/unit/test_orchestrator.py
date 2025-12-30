"""Tests for orchestrator."""

import pytest
from tokenomics.orchestrator import TokenAwareOrchestrator, QueryComplexity


def test_complexity_analysis():
    """Test query complexity analysis."""
    orchestrator = TokenAwareOrchestrator()
    
    assert orchestrator.analyze_complexity("short") == QueryComplexity.SIMPLE
    assert orchestrator.analyze_complexity("a" * 200) == QueryComplexity.MEDIUM
    assert orchestrator.analyze_complexity("a" * 1000) == QueryComplexity.COMPLEX


def test_token_allocation():
    """Test token allocation."""
    orchestrator = TokenAwareOrchestrator(default_token_budget=1000)
    
    components = {
        "system": {"cost": 100, "utility": 1.0},
        "query": {"cost": 50, "utility": 1.0},
        "context": {"cost": 500, "utility": 0.8},
    }
    
    allocations = orchestrator.allocate_tokens_greedy(components, budget=1000)
    assert len(allocations) > 0
    
    total_allocated = sum(a.tokens for a in allocations)
    assert total_allocated <= 1000


def test_query_plan():
    """Test query planning."""
    orchestrator = TokenAwareOrchestrator(default_token_budget=2000)
    
    plan = orchestrator.plan_query("test query")
    assert plan.query == "test query"
    assert plan.token_budget == 2000
    assert len(plan.allocations) > 0

