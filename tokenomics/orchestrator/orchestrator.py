"""Token-aware orchestrator for dynamic budget allocation."""

import tiktoken
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger()


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class TokenAllocation:
    """Token allocation for a component."""
    component: str
    tokens: int
    priority: int = 0
    utility: float = 0.0


@dataclass
class QueryPlan:
    """Plan for executing a query."""
    query: str
    complexity: QueryComplexity
    token_budget: int
    allocations: List[TokenAllocation] = field(default_factory=list)
    model: Optional[str] = None
    use_retrieval: bool = False
    retrieved_context: List[str] = field(default_factory=list)
    compressed_prompt: Optional[str] = None
    context_quality_score: float = 1.0  # 1.0 = full context, 0.0 = no context
    context_compression_ratio: Optional[float] = None  # compressed/original
    context_original_tokens: int = 0
    context_allocated_tokens: int = 0


class TokenAwareOrchestrator:
    """Orchestrator that allocates token budgets optimally."""
    
    def __init__(
        self,
        default_token_budget: int = 4000,
        max_context_tokens: int = 8000,
        use_knapsack_optimization: bool = True,
        compression_ratio: float = 0.5,
        enable_multi_model_routing: bool = True,
        provider: str = "gemini",
        max_context_ratio: float = 0.4,
        min_response_ratio: float = 0.3,
    ):
        """
        Initialize token-aware orchestrator.
        
        Args:
            default_token_budget: Default token budget per query
            max_context_tokens: Maximum context window size
            use_knapsack_optimization: Use knapsack solver for allocation
            compression_ratio: Ratio for compressing low-value content
            enable_multi_model_routing: Enable multi-model routing
            provider: LLM provider name (gemini, openai, vllm)
            max_context_ratio: Maximum ratio of budget allocated to context (0.0-1.0)
            min_response_ratio: Minimum ratio of budget reserved for response generation
        """
        self.default_token_budget = default_token_budget
        self.max_context_tokens = max_context_tokens
        self.use_knapsack_optimization = use_knapsack_optimization
        self.compression_ratio = compression_ratio
        self.enable_multi_model_routing = enable_multi_model_routing
        self.provider = provider
        self.max_context_ratio = max_context_ratio
        self.min_response_ratio = min_response_ratio
        
        # Initialize tokenizer (using GPT-4 tokenizer as approximation)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Could not load tiktoken, using fallback")
            self.tokenizer = None
        
        logger.info(
            "TokenAwareOrchestrator initialized",
            default_budget=default_token_budget,
            max_context=max_context_tokens,
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Better fallback: account for punctuation, whitespace
        # Average: ~0.75 tokens per word for English
        # Add 20% for punctuation and special characters
        words = len(text.split())
        if words == 0:
            return 0
        # More accurate estimate: words * 0.75 * 1.2 for punctuation
        return int(words * 0.75 * 1.2)
    
    def analyze_complexity(self, query: str, complexity_classifier=None, query_embedding=None) -> QueryComplexity:
        """
        Analyze query complexity using ML classifier if available.
        
        Args:
            query: User query
            complexity_classifier: Optional ComplexityClassifier instance
            query_embedding: Optional query embedding vector
        
        Returns:
            QueryComplexity enum (SIMPLE, MEDIUM, COMPLEX)
        """
        if complexity_classifier:
            try:
                predicted = complexity_classifier.predict(query, query_embedding)
                return QueryComplexity[predicted.upper()]
            except Exception as e:
                logger.warning("ML complexity prediction failed, using heuristic", error=str(e))
        
        # Fallback to heuristic (current implementation)
        return self._analyze_complexity_heuristic(query)
    
    def _analyze_complexity_heuristic(self, query: str) -> QueryComplexity:
        """Analyze query complexity with enhanced semantic indicators (heuristic fallback)."""
        query_length = len(query)
        token_count = self.count_tokens(query)
        query_lower = query.lower()
        
        # Enhanced keyword-based complexity indicators
        complex_indicators = [
            "design", "architecture", "compare", "analyze", "comprehensive",
            "detailed", "system", "pipeline", "implement", "optimize",
            "production", "enterprise", "scalable", "microservices",
            "explain the differences", "provide a comprehensive",
            "write a detailed", "create a complete"
        ]
        medium_indicators = [
            "how does", "explain", "difference", "work", "what are",
            "benefits", "use cases", "advantages", "disadvantages"
        ]
        
        # Count indicator matches
        complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
        medium_score = sum(1 for ind in medium_indicators if ind in query_lower)
        
        # Count question marks (multiple questions = complex)
        question_count = query.count('?')
        
        # Check for comparison/analysis patterns
        comparison_patterns = ["vs", "versus", "difference between", "compare", "contrast"]
        has_comparison = any(pattern in query_lower for pattern in comparison_patterns)
        
        # Enhanced classification logic
        # Complex if:
        # - Multiple complex indicators (>= 2)
        # - Long query (>= 50 tokens) with at least one complex indicator
        # - Multiple questions (>= 2)
        # - Comparison pattern with substantial length (>= 30 tokens)
        if complex_score >= 2 or (token_count >= 50 and complex_score >= 1) or \
           (question_count >= 2) or (has_comparison and token_count >= 30):
            return QueryComplexity.COMPLEX
        # Medium if:
        # - At least one medium indicator
        # - Token count >= 20
        # - At least one complex indicator (but not enough for complex)
        elif token_count >= 20 or complex_score >= 1 or medium_score >= 1:
            return QueryComplexity.MEDIUM
        # Simple otherwise
        else:
            return QueryComplexity.SIMPLE
    
    # NOTE: select_model() method removed - model selection is now handled by Bandit optimizer
    # This ensures clear separation: Orchestrator plans, Bandit routes
    # If you need model selection, use the Bandit's strategy selection instead
    
    def allocate_tokens_knapsack(
        self,
        candidates: List[Tuple[str, int, float]],  # (component, cost, utility)
        budget: int,
    ) -> List[TokenAllocation]:
        """
        Allocate tokens using knapsack optimization.
        
        Args:
            candidates: List of (component_name, token_cost, utility_score)
            budget: Token budget
        
        Returns:
            List of token allocations
        """
        if not candidates:
            return []
        
        # Use greedy approach: sort by utility density (utility / cost)
        candidates_with_density = [
            (name, cost, util, util / cost if cost > 0 else 0)
            for name, cost, util in candidates
        ]
        candidates_with_density.sort(key=lambda x: x[3], reverse=True)
        
        allocations = []
        remaining_budget = budget
        
        logger.debug(
            "Knapsack optimization starting",
            budget=budget,
            num_candidates=len(candidates_with_density),
            candidates=[(name, cost, f"{util:.2f}", f"{density:.4f}") for name, cost, util, density in candidates_with_density],
        )
        
        for name, cost, util, density in candidates_with_density:
            if cost <= remaining_budget:
                allocations.append(TokenAllocation(
                    component=name,
                    tokens=cost,
                    utility=util,
                ))
                remaining_budget -= cost
                logger.debug(
                    "Allocated tokens",
                    component=name,
                    tokens=cost,
                    utility=util,
                    utility_density=f"{density:.4f}",
                    remaining_budget=remaining_budget,
                )
        
        total_allocated = sum(a.tokens for a in allocations)
        logger.info(
            "Knapsack optimization completed",
            total_allocated=total_allocated,
            budget=budget,
            remaining_budget=remaining_budget,
            utilization_percent=(total_allocated / budget * 100) if budget > 0 else 0,
            num_allocations=len(allocations),
        )
        
        # Verify allocations sum correctly
        if total_allocated > budget:
            logger.warning(
                "Knapsack allocation exceeds budget",
                total_allocated=total_allocated,
                budget=budget,
                excess=total_allocated - budget,
            )
        
        return allocations
    
    def allocate_tokens_greedy(
        self,
        components: Dict[str, Dict[str, float]],  # {component: {cost: int, utility: float}}
        budget: int,
    ) -> List[TokenAllocation]:
        """Greedy token allocation."""
        candidates = [
            (name, int(info["cost"]), info["utility"])
            for name, info in components.items()
        ]
        return self.allocate_tokens_knapsack(candidates, budget)
    
    def compress_text(self, text: str, target_tokens: int) -> str:
        """Compress text to target token count."""
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= target_tokens:
            return text
        
        # Simple compression: truncate and add summary indicator
        # In production, use LLM to summarize
        ratio = target_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.9)  # 90% to leave room
        
        compressed = text[:target_chars]
        if len(text) > target_chars:
            compressed += "... [truncated]"
        
        return compressed
    
    def plan_query(
        self,
        query: str,
        token_budget: Optional[int] = None,
        retrieved_context: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> QueryPlan:
        """
        Create execution plan for a query.
        
        Args:
            query: User query
            token_budget: Token budget (uses default if None)
            retrieved_context: Retrieved context from memory layer
            metadata: Additional metadata
        
        Returns:
            QueryPlan with token allocations
        """
        budget = token_budget or self.default_token_budget
        complexity = self.analyze_complexity(query)
        
        # Estimate token costs
        query_tokens = self.count_tokens(query)
        
        # Allocate tokens across components
        components = {}
        
        # System prompt (fixed, high priority)
        system_prompt_tokens = 100
        components["system_prompt"] = {
            "cost": system_prompt_tokens,
            "utility": 1.0,  # High utility
        }
        
        # User query (required)
        components["user_query"] = {
            "cost": query_tokens,
            "utility": 1.0,
        }
        
        # Retrieved context (variable, based on relevance)
        if retrieved_context:
            context_tokens = sum(self.count_tokens(ctx) for ctx in retrieved_context)
            # Allocate up to max_context_ratio of budget for context
            max_context_tokens = int(budget * self.max_context_ratio)
            context_tokens = min(context_tokens, max_context_tokens)
            
            components["retrieved_context"] = {
                "cost": context_tokens,
                "utility": 0.8,  # High utility if relevant
            }
        
        # Response generation (remaining budget)
        remaining = budget - sum(c["cost"] for c in components.values())
        
        # Ensure response gets minimum allocation
        min_response_tokens = int(budget * self.min_response_ratio)
        if remaining < min_response_tokens:
            # Adjust context allocation to ensure response minimum
            if "retrieved_context" in components:
                max_context_tokens = int(budget * self.max_context_ratio)
                # Reduce context allocation to free up tokens for response
                available_for_context = budget - system_prompt_tokens - query_tokens - min_response_tokens
                max_context_tokens = min(max_context_tokens, max(0, available_for_context))
                components["retrieved_context"]["cost"] = min(
                    components["retrieved_context"]["cost"],
                    max_context_tokens
                )
                # Recalculate remaining after adjustment
                remaining = budget - sum(c["cost"] for c in components.values())
        
        if remaining > 0:
            components["response"] = {
                "cost": max(remaining, min_response_tokens),
                "utility": 0.9,
            }
        
        # Optimize allocation
        allocations = self.allocate_tokens_greedy(components, budget)
        
        # NOTE: Model selection is now handled by Bandit optimizer, not orchestrator
        # Orchestrator only plans and allocates tokens - routing is Bandit's responsibility
        model = None  # Will be set by bandit strategy selection
        
        # Compress context if needed
        compressed_context = None
        context_quality_score = 1.0
        context_compression_ratio = None
        context_original_tokens = 0
        context_allocated_tokens = 0
        
        if retrieved_context:
            context_allocation = next(
                (a for a in allocations if a.component == "retrieved_context"),
                None
            )
            if context_allocation:
                # Calculate original context size
                context_original_tokens = sum(self.count_tokens(ctx) for ctx in retrieved_context)
                context_allocated_tokens = context_allocation.tokens
                
                # Calculate compression ratio
                if context_original_tokens > 0:
                    context_compression_ratio = context_allocated_tokens / context_original_tokens
                    # Quality score: 1.0 = full context, lower for compressed
                    # Slight boost for partial context (compression might be smart)
                    context_quality_score = min(1.0, context_compression_ratio * 1.2)
                else:
                    context_compression_ratio = 1.0
                    context_quality_score = 1.0
                
                # Compress the context
                full_context = " ".join(retrieved_context)
                compressed_context = self.compress_text(
                    full_context,
                    context_allocation.tokens,
                )
        elif not retrieved_context:
            # No context retrieved - quality score is 1.0 (no context needed)
            context_quality_score = 1.0
        
        plan = QueryPlan(
            query=query,
            complexity=complexity,
            token_budget=budget,
            allocations=allocations,
            model=model,
            use_retrieval=bool(retrieved_context),
            retrieved_context=retrieved_context or [],
            compressed_prompt=compressed_context,
            context_quality_score=context_quality_score,
            context_compression_ratio=context_compression_ratio,
            context_original_tokens=context_original_tokens,
            context_allocated_tokens=context_allocated_tokens,
        )
        
        logger.debug(
            "Query plan created",
            complexity=complexity.value,
            budget=budget,
            model=model,
            num_allocations=len(allocations),
        )
        
        return plan
    
    def build_prompt(
        self,
        plan: QueryPlan,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build prompt from plan."""
        parts = []
        
        # System prompt
        if system_prompt:
            sys_alloc = next(
                (a for a in plan.allocations if a.component == "system_prompt"),
                None
            )
            if sys_alloc:
                parts.append(system_prompt)
        
        # Retrieved context
        if plan.use_retrieval and plan.compressed_prompt:
            parts.append("Context:")
            parts.append(plan.compressed_prompt)
        
        # User query - no "Query:" prefix to avoid +2 token overhead
        parts.append(plan.query)
        
        return "\n\n".join(parts)

