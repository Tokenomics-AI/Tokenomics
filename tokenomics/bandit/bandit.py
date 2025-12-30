"""Multi-armed bandit optimizer for strategy selection with RouterBench cost-quality routing."""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import numpy as np
import structlog

logger = structlog.get_logger()


class BanditAlgorithm(Enum):
    """Bandit algorithm types."""
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson"


# RouterBench-style model cost per 1M tokens (input + output averaged)
MODEL_COSTS = {
    # OpenAI models
    "gpt-4o-mini": 0.15,  # $0.15 per 1M tokens
    "gpt-4o": 2.50,  # $2.50 per 1M tokens
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
    # Gemini models
    "gemini-flash": 0.075,
    "gemini-pro": 1.25,
    "gemini-2.0-flash-exp": 0.075,
    # Default
    "default": 0.10,
}


@dataclass
class Strategy:
    """A strategy configuration (arm)."""
    arm_id: str
    model: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1  # Number of completions
    rerank: bool = False
    memory_mode: str = "light"  # off, light, rich
    metadata: Dict = field(default_factory=dict)


@dataclass
class RoutingMetrics:
    """RouterBench-style metrics for cost-quality evaluation."""
    total_cost: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_quality: float = 0.0
    query_count: int = 0
    
    @property
    def avg_cost_per_query(self) -> float:
        return self.total_cost / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def cost_quality_ratio(self) -> float:
        """Higher is better: quality per unit cost."""
        if self.total_cost == 0:
            return float('inf') if self.avg_quality > 0 else 0.0
        return self.avg_quality / (self.avg_cost_per_query * 1000)  # Scale for readability
    
    @property
    def efficiency_score(self) -> float:
        """Combined efficiency: balances quality, cost, and speed."""
        quality_component = self.avg_quality * 0.5
        cost_component = (1.0 / (1.0 + self.avg_cost_per_query * 100)) * 0.3
        speed_component = (1.0 / (1.0 + self.avg_latency / 5000)) * 0.2
        return quality_component + cost_component + speed_component
    
    def update(self, cost: float, tokens: int, latency_ms: float, quality: float):
        """Update metrics with new observation."""
        self.total_cost += cost
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.total_quality += quality
        self.query_count += 1


@dataclass
class BanditArm:
    """Statistics for a bandit arm with RouterBench metrics."""
    strategy: Strategy
    pulls: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    routing_metrics: RoutingMetrics = field(default_factory=RoutingMetrics)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, reward: float):
        """Update arm statistics."""
        self.pulls += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.pulls
    
    def update_routing(self, cost: float, tokens: int, latency_ms: float, quality: float):
        """Update RouterBench metrics."""
        self.routing_metrics.update(cost, tokens, latency_ms, quality)


class BanditOptimizer:
    """Multi-armed bandit for strategy selection."""
    
    def __init__(
        self,
        algorithm: str = "ucb",
        exploration_rate: float = 0.1,
        reward_lambda: float = 0.001,
        reset_frequency: Optional[int] = None,
        contextual: bool = False,
        state_file: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Initialize bandit optimizer.
        
        Args:
            algorithm: "ucb", "epsilon_greedy", or "thompson"
            exploration_rate: Exploration rate for epsilon-greedy
            reward_lambda: Lambda for reward = quality - lambda * tokens
            reset_frequency: Reset after N queries (None = never)
            contextual: Use contextual bandits by query type
            state_file: Path to state persistence file
            auto_save: Automatically save state after each update
        """
        self.algorithm = BanditAlgorithm(algorithm)
        self.exploration_rate = exploration_rate
        self.reward_lambda = reward_lambda
        self.reset_frequency = reset_frequency
        self.contextual = contextual
        self.state_file = state_file
        self.auto_save = auto_save
        
        self.arms: Dict[str, BanditArm] = {}
        self.total_pulls = 0
        self.query_count = 0
        
        logger.info(
            "BanditOptimizer initialized",
            algorithm=algorithm,
            exploration_rate=exploration_rate,
            reward_lambda=reward_lambda,
            state_file=state_file,
            auto_save=auto_save,
        )
    
    def add_strategy(self, strategy: Strategy):
        """Add a strategy (arm) to the bandit."""
        arm = BanditArm(strategy=strategy)
        self.arms[strategy.arm_id] = arm
        logger.debug("Added strategy", arm_id=strategy.arm_id)
    
    def add_strategies(self, strategies: List[Strategy]):
        """Add multiple strategies."""
        for strategy in strategies:
            self.add_strategy(strategy)
    
    def compute_reward(
        self,
        quality_score: float,
        tokens_used: int,
        latency_ms: Optional[float] = None,
    ) -> float:
        """
        Compute reward signal.
        
        Args:
            quality_score: Quality score (0-1)
            tokens_used: Number of tokens used
            latency_ms: Latency in milliseconds (optional)
        
        Returns:
            Reward value
        """
        # Base reward: quality - lambda * tokens
        reward = quality_score - self.reward_lambda * tokens_used
        
        # Optionally penalize latency
        if latency_ms is not None:
            latency_penalty = latency_ms / 10000.0  # Normalize
            reward -= 0.1 * latency_penalty
        
        return reward
    
    # =========================================================================
    # RouterBench Cost-Quality Routing
    # =========================================================================
    
    def get_model_cost(self, model: str) -> float:
        """Get cost per 1M tokens for a model."""
        return MODEL_COSTS.get(model, MODEL_COSTS["default"])
    
    def compute_reward_routerbench(
        self,
        arm_id: str,
        quality_score: float,
        tokens_used: int,
        latency_ms: float,
        model: str,
    ) -> float:
        """
        RouterBench-style reward computation considering cost, quality, and latency.
        
        Args:
            arm_id: Strategy arm ID
            quality_score: Quality score (0-1)
            tokens_used: Number of tokens used
            latency_ms: Latency in milliseconds
            model: Model name for cost lookup
        
        Returns:
            Reward value optimized for cost-quality tradeoff
        """
        # Calculate actual cost
        cost_per_million = self.get_model_cost(model)
        actual_cost = (tokens_used / 1_000_000) * cost_per_million
        
        # Update routing metrics for this arm
        if arm_id in self.arms:
            self.arms[arm_id].update_routing(actual_cost, tokens_used, latency_ms, quality_score)
        
        # RouterBench reward formula:
        # Reward = quality * (1 - cost_weight * normalized_cost) * (1 - latency_weight * normalized_latency)
        cost_weight = 0.3
        latency_weight = 0.2
        
        # Normalize cost (assume max cost ~$0.01 per query)
        normalized_cost = min(1.0, actual_cost / 0.01)
        
        # Normalize latency (assume max acceptable latency ~30s)
        normalized_latency = min(1.0, latency_ms / 30000)
        
        reward = quality_score * (1 - cost_weight * normalized_cost) * (1 - latency_weight * normalized_latency)
        
        logger.debug(
            "RouterBench reward computed",
            arm_id=arm_id,
            quality=quality_score,
            cost=f"${actual_cost:.6f}",
            latency_ms=f"{latency_ms:.0f}",
            reward=f"{reward:.4f}",
        )
        
        return reward
    
    def select_strategy_cost_aware(
        self,
        query_complexity: str = "medium",
        budget_constraint: Optional[float] = None,
        context_quality_score: float = 1.0,
    ) -> Optional[Strategy]:
        """
        RouterBench-style strategy selection considering cost-quality tradeoff.
        
        Args:
            query_complexity: "simple", "medium", or "complex"
            budget_constraint: Maximum cost per query (optional)
        
        Returns:
            Selected strategy optimized for cost-quality
        """
        if not self.arms:
            logger.warning("No strategies available")
            return None
        
        # Filter by budget constraint if specified
        candidates = {}
        for arm_id, arm in self.arms.items():
            strategy = arm.strategy
            
            # Estimate cost for this strategy
            cost_per_million = self.get_model_cost(strategy.model)
            estimated_cost = (strategy.max_tokens / 1_000_000) * cost_per_million
            
            # Check budget constraint
            if budget_constraint and estimated_cost > budget_constraint:
                continue
            
            # Soft scoring based on complexity (penalize but don't exclude)
            complexity_penalty = 0.0
            if query_complexity == "simple":
                # Penalize expensive strategies for simple queries, but don't exclude
                if strategy.max_tokens > 500:
                    complexity_penalty = 0.3  # 30% penalty
                elif strategy.max_tokens > 300:
                    complexity_penalty = 0.1  # 10% penalty
            elif query_complexity == "medium":
                # Slight preference for balanced strategies
                if strategy.max_tokens < 300:
                    complexity_penalty = 0.1
            elif query_complexity == "complex":
                # Penalize cheap strategies for complex queries
                if strategy.max_tokens < 500:
                    complexity_penalty = 0.2
            
            # Context-aware penalty: prefer premium strategies when context is heavily compressed
            context_penalty = 0.0
            if context_quality_score < 0.7:
                # Context is heavily compressed - prefer premium strategies
                if strategy.max_tokens < 500:  # Cheap/balanced strategies
                    context_penalty = 0.2 * (1.0 - context_quality_score)  # Up to 20% penalty
                # Premium strategies get no penalty (or slight boost)
                elif strategy.max_tokens >= 1000:
                    context_penalty = -0.1  # Slight boost for premium
            
            # Score based on efficiency
            if arm.routing_metrics.query_count > 0:
                base_score = arm.routing_metrics.efficiency_score
                score = base_score * (1.0 - complexity_penalty) * (1.0 - context_penalty)
            else:
                # For unexplored arms, apply penalty to initial score
                base_score = 0.8 - (estimated_cost * 10)  # Cheaper = higher initial score
                score = base_score * (1.0 - complexity_penalty) * (1.0 - context_penalty)
            
            candidates[arm_id] = (arm, score)
        
        if not candidates:
            # Fallback to standard selection if no candidates pass filters
            return self.select_strategy()
        
        # Exploration vs exploitation
        if self.total_pulls < len(self.arms) * 2:
            # Exploration phase: try each arm at least twice
            unexplored = [aid for aid, (arm, _) in candidates.items() if arm.pulls < 2]
            if unexplored:
                selected_id = np.random.choice(unexplored)
                return self.arms[selected_id].strategy
        
        # Exploitation: select best efficiency score
        best_arm_id = max(candidates.items(), key=lambda x: x[1][1])[0]
        
        logger.debug(
            "Cost-aware strategy selected",
            arm_id=best_arm_id,
            complexity=query_complexity,
            context_quality_score=context_quality_score,
        )
        
        return self.arms[best_arm_id].strategy
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get RouterBench-style routing statistics."""
        stats = {
            "total_queries": sum(arm.routing_metrics.query_count for arm in self.arms.values()),
            "total_cost": sum(arm.routing_metrics.total_cost for arm in self.arms.values()),
            "total_tokens": sum(arm.routing_metrics.total_tokens for arm in self.arms.values()),
            "arms": {},
        }
        
        best_efficiency = None
        best_arm_id = None
        
        for arm_id, arm in self.arms.items():
            metrics = arm.routing_metrics
            arm_stats = {
                "queries": metrics.query_count,
                "avg_cost": metrics.avg_cost_per_query,
                "avg_tokens": metrics.avg_tokens,
                "avg_latency_ms": metrics.avg_latency,
                "avg_quality": metrics.avg_quality,
                "cost_quality_ratio": metrics.cost_quality_ratio,
                "efficiency_score": metrics.efficiency_score,
            }
            stats["arms"][arm_id] = arm_stats
            
            if metrics.query_count > 0:
                if best_efficiency is None or metrics.efficiency_score > best_efficiency:
                    best_efficiency = metrics.efficiency_score
                    best_arm_id = arm_id
        
        stats["best_efficiency_arm"] = best_arm_id
        stats["best_efficiency_score"] = best_efficiency
        
        return stats
    
    def select_arm_ucb(self) -> Optional[str]:
        """Select arm using Upper Confidence Bound."""
        if not self.arms:
            return None
        
        if self.total_pulls == 0:
            # First pull: select randomly
            return np.random.choice(list(self.arms.keys()))
        
        # UCB formula: average_reward + c * sqrt(ln(total_pulls) / arm_pulls)
        c = 2.0  # Exploration constant
        best_arm = None
        best_value = float('-inf')
        
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                # Unpulled arm gets infinite value
                return arm_id
            
            ucb_value = (
                arm.average_reward +
                c * np.sqrt(np.log(self.total_pulls) / arm.pulls)
            )
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_arm = arm_id
        
        return best_arm
    
    def select_arm_epsilon_greedy(self) -> Optional[str]:
        """Select arm using epsilon-greedy."""
        if not self.arms:
            return None
        
        if np.random.random() < self.exploration_rate:
            # Explore: random selection
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: best average reward
            best_arm = None
            best_reward = float('-inf')
            
            for arm_id, arm in self.arms.items():
                if arm.pulls == 0:
                    # Unpulled arm gets a chance
                    if np.random.random() < 0.5:
                        return arm_id
                
                if arm.average_reward > best_reward:
                    best_reward = arm.average_reward
                    best_arm = arm_id
            
            return best_arm or np.random.choice(list(self.arms.keys()))
    
    def select_arm_thompson(self) -> Optional[str]:
        """Select arm using Thompson Sampling (Beta distribution)."""
        if not self.arms:
            return None
        
        # Thompson Sampling with Beta distribution
        # Assume rewards are in [0, 1], use Beta(alpha, beta)
        # alpha = successes + 1, beta = failures + 1
        samples = {}
        
        for arm_id, arm in self.arms.items():
            if arm.pulls == 0:
                # Unpulled arm: use uniform prior
                alpha, beta = 1, 1
            else:
                # Estimate success rate from average reward
                # Convert average reward to success count
                successes = max(1, int(arm.average_reward * arm.pulls))
                failures = max(1, arm.pulls - successes)
                alpha = successes + 1
                beta = failures + 1
            
            # Sample from Beta distribution
            samples[arm_id] = np.random.beta(alpha, beta)
        
        # Select arm with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def select_strategy(self, query_type: Optional[str] = None) -> Optional[Strategy]:
        """
        Select a strategy (arm) to use.
        
        Args:
            query_type: Query type for contextual bandits (optional)
        
        Returns:
            Selected strategy
        """
        if not self.arms:
            logger.warning("No strategies available")
            return None
        
        # Contextual bandits: filter by query type if enabled
        available_arms = self.arms
        if self.contextual and query_type:
            # Filter arms by query type (if metadata includes query_type)
            available_arms = {
                k: v for k, v in self.arms.items()
                if v.strategy.metadata.get("query_type") == query_type
            }
            if not available_arms:
                available_arms = self.arms  # Fallback to all
        
        # Select arm based on algorithm
        if self.algorithm == BanditAlgorithm.UCB:
            arm_id = self.select_arm_ucb()
        elif self.algorithm == BanditAlgorithm.EPSILON_GREEDY:
            arm_id = self.select_arm_epsilon_greedy()
        elif self.algorithm == BanditAlgorithm.THOMPSON_SAMPLING:
            arm_id = self.select_arm_thompson()
        else:
            arm_id = np.random.choice(list(available_arms.keys()))
        
        if arm_id and arm_id in available_arms:
            arm = available_arms[arm_id]
            logger.debug("Selected strategy", arm_id=arm_id, pulls=arm.pulls)
            return arm.strategy
        
        return None
    
    def update(
        self,
        arm_id: str,
        reward: float,
    ):
        """
        Update arm statistics after pulling.
        
        Args:
            arm_id: ID of the arm that was pulled
            reward: Observed reward
        """
        if arm_id not in self.arms:
            logger.warning("Unknown arm_id", arm_id=arm_id)
            return
        
        arm = self.arms[arm_id]
        arm.update(reward)
        self.total_pulls += 1
        self.query_count += 1
        
        # Reset if needed
        if self.reset_frequency and self.query_count >= self.reset_frequency:
            self.reset()
        
        logger.debug(
            "Updated arm",
            arm_id=arm_id,
            reward=reward,
            average_reward=arm.average_reward,
            pulls=arm.pulls,
        )
        
        # Auto-save state if enabled
        if self.auto_save and self.state_file:
            try:
                self.save_state()
            except Exception as e:
                logger.warning("Failed to auto-save bandit state", error=str(e))
    
    def reset(self):
        """Reset all arm statistics."""
        for arm in self.arms.values():
            arm.pulls = 0
            arm.total_reward = 0.0
            arm.average_reward = 0.0
        
        self.total_pulls = 0
        self.query_count = 0
        logger.info("Bandit reset")
    
    def get_best_strategy(self) -> Optional[Strategy]:
        """Get the strategy with highest average reward."""
        if not self.arms:
            return None
        
        best_arm = max(self.arms.values(), key=lambda a: a.average_reward)
        return best_arm.strategy
    
    def save_state(self, filepath: Optional[str] = None) -> None:
        """
        Save bandit state to JSON file.
        
        Args:
            filepath: Path to save state (uses self.state_file if None)
        """
        save_path = filepath or self.state_file
        if not save_path:
            logger.warning("No state file path specified, cannot save state")
            return
        
        try:
            state = {
                "total_pulls": self.total_pulls,
                "query_count": self.query_count,
                "arms": {},
            }
            
            # Serialize arm statistics (NOT strategy configuration)
            for arm_id, arm in self.arms.items():
                arm_data = {
                    "pulls": arm.pulls,
                    "total_reward": arm.total_reward,
                    "average_reward": arm.average_reward,
                    "routing_metrics": {
                        "total_cost": arm.routing_metrics.total_cost,
                        "total_tokens": arm.routing_metrics.total_tokens,
                        "total_latency_ms": arm.routing_metrics.total_latency_ms,
                        "total_quality": arm.routing_metrics.total_quality,
                        "query_count": arm.routing_metrics.query_count,
                    },
                }
                state["arms"][arm_id] = arm_data
            
            # Write to file
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info("Bandit state saved", filepath=save_path, arms=len(state["arms"]))
        except Exception as e:
            logger.error("Failed to save bandit state", error=str(e), filepath=save_path)
            raise
    
    def load_state(self, filepath: Optional[str] = None) -> None:
        """
        Load bandit state from JSON file (MERGE strategy - only statistics, not config).
        
        CRITICAL: This method uses MERGE strategy, not overwrite:
        - Strategies are initialized from config.py first (Source of Truth)
        - Only learning statistics are restored from file
        - Strategy configuration (model, max_tokens, etc.) is NOT overwritten
        
        Args:
            filepath: Path to load state from (uses self.state_file if None)
        """
        load_path = filepath or self.state_file
        if not load_path or not Path(load_path).exists():
            logger.debug("No state file found or path not specified", filepath=load_path)
            return
        
        try:
            with open(load_path, 'r') as f:
                state = json.load(f)
            
            # Restore global statistics
            self.total_pulls = state.get("total_pulls", 0)
            self.query_count = state.get("query_count", 0)
            
            # MERGE: Only update statistics for arms that exist in current config
            loaded_arms = 0
            skipped_arms = 0
            
            for arm_id, arm_data in state.get("arms", {}).items():
                if arm_id in self.arms:
                    # Arm exists in current config - merge statistics
                    arm = self.arms[arm_id]
                    
                    # Update learning statistics
                    arm.pulls = arm_data.get("pulls", 0)
                    arm.total_reward = arm_data.get("total_reward", 0.0)
                    arm.average_reward = arm_data.get("average_reward", 0.0)
                    
                    # Restore routing metrics
                    routing_data = arm_data.get("routing_metrics", {})
                    arm.routing_metrics.total_cost = routing_data.get("total_cost", 0.0)
                    arm.routing_metrics.total_tokens = routing_data.get("total_tokens", 0)
                    arm.routing_metrics.total_latency_ms = routing_data.get("total_latency_ms", 0.0)
                    arm.routing_metrics.total_quality = routing_data.get("total_quality", 0.0)
                    arm.routing_metrics.query_count = routing_data.get("query_count", 0)
                    
                    # DO NOT update arm.strategy (model, max_tokens, etc.) - keep from config
                    loaded_arms += 1
                else:
                    # Arm doesn't exist in current config - skip it
                    skipped_arms += 1
                    logger.debug("Skipping arm not in current config", arm_id=arm_id)
            
            logger.info(
                "Bandit state loaded (merged)",
                filepath=load_path,
                loaded_arms=loaded_arms,
                skipped_arms=skipped_arms,
                total_pulls=self.total_pulls,
                query_count=self.query_count,
            )
        except json.JSONDecodeError as e:
            logger.warning("Corrupt state file, starting fresh", error=str(e), filepath=load_path)
        except Exception as e:
            logger.error("Failed to load bandit state", error=str(e), filepath=load_path)
            # Don't raise - allow bandit to start fresh if state loading fails
    
    def stats(self) -> Dict:
        """Get bandit statistics including RouterBench metrics."""
        base_stats = {
            "algorithm": self.algorithm.value,
            "total_pulls": self.total_pulls,
            "num_arms": len(self.arms),
            "arms": {
                arm_id: {
                    "pulls": arm.pulls,
                    "average_reward": arm.average_reward,
                    "total_reward": arm.total_reward,
                    "routing": {
                        "queries": arm.routing_metrics.query_count,
                        "avg_cost": arm.routing_metrics.avg_cost_per_query,
                        "avg_tokens": arm.routing_metrics.avg_tokens,
                        "efficiency": arm.routing_metrics.efficiency_score,
                    } if arm.routing_metrics.query_count > 0 else None,
                }
                for arm_id, arm in self.arms.items()
            },
        }
        
        # Add summary routing stats
        routing_stats = self.get_routing_stats()
        base_stats["routing_summary"] = {
            "total_cost": routing_stats["total_cost"],
            "total_tokens": routing_stats["total_tokens"],
            "best_arm": routing_stats["best_efficiency_arm"],
        }
        
        return base_stats
    
    def predict_escalation_likelihood(
        self,
        query_complexity: str,
        context_quality_score: float,
        query_tokens: int,
        cheap_arm_id: Optional[str] = None,
    ) -> float:
        """
        DEPRECATED: Use EscalationPredictor instead.
        
        This method is kept for backward compatibility but is deprecated.
        The platform now uses ML-based escalation prediction via EscalationPredictor.
        
        Predict probability (0.0-1.0) that escalation will be needed.
        
        Uses historical escalation data from cheap model arm to predict
        if current query will likely need premium model.
        
        Args:
            query_complexity: Query complexity ("simple", "medium", "complex")
            context_quality_score: Context quality score (0.0-1.0)
            query_tokens: Number of tokens in query
            cheap_arm_id: Optional arm ID for cheap model (auto-detected if None)
        
        Returns:
            Escalation probability (0.0 = unlikely, 1.0 = very likely)
        """
        import warnings
        warnings.warn(
            "BanditOptimizer.predict_escalation_likelihood() is deprecated. "
            "Use EscalationPredictor instead for ML-based prediction.",
            DeprecationWarning,
            stacklevel=2
        )
        # Find cheap model arm
        if cheap_arm_id is None:
            # Try to find cheap model arm automatically
            for arm_id, arm in self.arms.items():
                # Check if this looks like a cheap model (heuristic: low max_tokens)
                if arm.strategy.max_tokens < 500:
                    cheap_arm_id = arm_id
                    break
        
        if cheap_arm_id is None or cheap_arm_id not in self.arms:
            # No cheap arm found or no historical data - return conservative default
            logger.debug("No cheap arm found for escalation prediction, using default")
            return 0.3  # Conservative: assume 30% escalation likelihood
        
        arm = self.arms[cheap_arm_id]
        
        # If no escalation history, use heuristics based on query characteristics
        if not arm.escalation_history or len(arm.escalation_history) == 0:
            return self._predict_escalation_heuristic(
                query_complexity, context_quality_score, query_tokens
            )
        
        # Calculate escalation rate for similar queries
        escalation_rates = {
            "by_complexity": {},
            "by_context_quality": {},
            "by_query_tokens": {},
        }
        
        total_escalations = 0
        total_queries = len(arm.escalation_history)
        
        # Analyze historical data
        for record in arm.escalation_history:
            complexity = record.get("complexity", "medium")
            context_quality = record.get("context_quality", 1.0)
            tokens = record.get("query_tokens", 0)
            escalated = record.get("escalated", False)
            
            if escalated:
                total_escalations += 1
            
            # Group by complexity
            if complexity not in escalation_rates["by_complexity"]:
                escalation_rates["by_complexity"][complexity] = {"escalated": 0, "total": 0}
            escalation_rates["by_complexity"][complexity]["total"] += 1
            if escalated:
                escalation_rates["by_complexity"][complexity]["escalated"] += 1
            
            # Group by context quality ranges
            quality_range = "high" if context_quality >= 0.8 else "medium" if context_quality >= 0.5 else "low"
            if quality_range not in escalation_rates["by_context_quality"]:
                escalation_rates["by_context_quality"][quality_range] = {"escalated": 0, "total": 0}
            escalation_rates["by_context_quality"][quality_range]["total"] += 1
            if escalated:
                escalation_rates["by_context_quality"][quality_range]["escalated"] += 1
            
            # Group by query token ranges
            token_range = "short" if tokens < 50 else "medium" if tokens < 200 else "long"
            if token_range not in escalation_rates["by_query_tokens"]:
                escalation_rates["by_query_tokens"][token_range] = {"escalated": 0, "total": 0}
            escalation_rates["by_query_tokens"][token_range]["total"] += 1
            if escalated:
                escalation_rates["by_query_tokens"][token_range]["escalated"] += 1
        
        # Calculate weighted probability
        weights = {"complexity": 0.4, "context_quality": 0.35, "query_tokens": 0.25}
        probabilities = []
        
        # Complexity-based probability
        complexity_rate = escalation_rates["by_complexity"].get(query_complexity, {"escalated": 0, "total": 0})
        if complexity_rate["total"] > 0:
            prob_complexity = complexity_rate["escalated"] / complexity_rate["total"]
        else:
            # Fallback to overall rate
            prob_complexity = total_escalations / total_queries if total_queries > 0 else 0.3
        probabilities.append(prob_complexity * weights["complexity"])
        
        # Context quality-based probability
        quality_range = "high" if context_quality_score >= 0.8 else "medium" if context_quality_score >= 0.5 else "low"
        quality_rate = escalation_rates["by_context_quality"].get(quality_range, {"escalated": 0, "total": 0})
        if quality_rate["total"] > 0:
            prob_quality = quality_rate["escalated"] / quality_rate["total"]
        else:
            prob_quality = total_escalations / total_queries if total_queries > 0 else 0.3
        probabilities.append(prob_quality * weights["context_quality"])
        
        # Query tokens-based probability
        token_range = "short" if query_tokens < 50 else "medium" if query_tokens < 200 else "long"
        token_rate = escalation_rates["by_query_tokens"].get(token_range, {"escalated": 0, "total": 0})
        if token_rate["total"] > 0:
            prob_tokens = token_rate["escalated"] / token_rate["total"]
        else:
            prob_tokens = total_escalations / total_queries if total_queries > 0 else 0.3
        probabilities.append(prob_tokens * weights["query_tokens"])
        
        # Combine probabilities
        predicted_probability = sum(probabilities)
        
        # Clamp to [0.0, 1.0]
        predicted_probability = max(0.0, min(1.0, predicted_probability))
        
        logger.debug(
            "Escalation likelihood predicted",
            probability=predicted_probability,
            complexity=query_complexity,
            context_quality=context_quality_score,
            query_tokens=query_tokens,
            history_size=len(arm.escalation_history),
        )
        
        return predicted_probability
    
    def _predict_escalation_heuristic(
        self,
        query_complexity: str,
        context_quality_score: float,
        query_tokens: int,
    ) -> float:
        """
        DEPRECATED: Internal method for deprecated predict_escalation_likelihood().
        
        Heuristic escalation prediction when no historical data is available.
        
        Args:
            query_complexity: Query complexity
            context_quality_score: Context quality score
            query_tokens: Query token count
        
        Returns:
            Escalation probability (0.0-1.0)
        """
        probability = 0.0
        
        # Complexity factor
        complexity_weights = {"simple": 0.1, "medium": 0.3, "complex": 0.6}
        probability += complexity_weights.get(query_complexity, 0.3)
        
        # Context quality factor (low quality = higher escalation likelihood)
        if context_quality_score < 0.5:
            probability += 0.3
        elif context_quality_score < 0.7:
            probability += 0.15
        
        # Query length factor (very long queries might need premium)
        if query_tokens > 200:
            probability += 0.1
        elif query_tokens > 100:
            probability += 0.05
        
        # Normalize and clamp
        probability = max(0.0, min(1.0, probability))
        
        return probability
    
    def record_escalation_outcome(
        self,
        arm_id: str,
        query_complexity: str,
        context_quality_score: float,
        query_tokens: int,
        escalated: bool,
    ):
        """
        DEPRECATED: Use EscalationPredictor.record_outcome() instead.
        
        This method is kept for backward compatibility but is deprecated.
        The platform now uses ML-based escalation prediction with SQLite persistence.
        
        Record escalation outcome for learning.
        
        Stores escalation patterns to improve future predictions.
        
        Args:
            arm_id: Arm ID that was used
            query_complexity: Query complexity
            context_quality_score: Context quality score
            query_tokens: Query token count
            escalated: Whether escalation actually occurred
        """
        import warnings
        warnings.warn(
            "BanditOptimizer.record_escalation_outcome() is deprecated. "
            "Use EscalationPredictor.record_outcome() instead for ML-based learning.",
            DeprecationWarning,
            stacklevel=2
        )
        if arm_id not in self.arms:
            logger.warning("Unknown arm_id for escalation recording", arm_id=arm_id)
            return
        
        arm = self.arms[arm_id]
        
        # Store escalation history
        arm.escalation_history.append({
            "complexity": query_complexity,
            "context_quality": context_quality_score,
            "query_tokens": query_tokens,
            "escalated": escalated,
        })
        
        # Keep only recent history (last 100)
        if len(arm.escalation_history) > 100:
            arm.escalation_history = arm.escalation_history[-100:]
        
        logger.debug(
            "Escalation outcome recorded",
            arm_id=arm_id,
            escalated=escalated,
            history_size=len(arm.escalation_history),
        )

