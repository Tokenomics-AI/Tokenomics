"""Tests for bandit optimizer."""

import pytest
from tokenomics.bandit import BanditOptimizer, Strategy


def test_bandit_initialization():
    """Test bandit initialization."""
    bandit = BanditOptimizer(algorithm="ucb")
    assert bandit.algorithm.value == "ucb"


def test_strategy_selection():
    """Test strategy selection."""
    bandit = BanditOptimizer(algorithm="ucb")
    
    strategies = [
        Strategy(arm_id="s1", model="m1", max_tokens=100),
        Strategy(arm_id="s2", model="m2", max_tokens=200),
    ]
    
    bandit.add_strategies(strategies)
    
    # Should select a strategy
    strategy = bandit.select_strategy()
    assert strategy is not None
    assert strategy.arm_id in ["s1", "s2"]


def test_bandit_update():
    """Test bandit update."""
    bandit = BanditOptimizer(algorithm="ucb")
    
    strategy = Strategy(arm_id="s1", model="m1", max_tokens=100)
    bandit.add_strategy(strategy)
    
    # Select and update
    selected = bandit.select_strategy()
    assert selected is not None
    
    reward = bandit.compute_reward(quality_score=0.9, tokens_used=100)
    bandit.update(selected.arm_id, reward)
    
    # Check stats
    stats = bandit.stats()
    assert stats["total_pulls"] > 0

