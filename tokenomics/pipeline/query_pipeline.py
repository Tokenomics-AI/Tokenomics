"""Query pipeline implementation with pluggable stages."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""
    query: str
    token_budget: Optional[int] = None
    system_prompt: Optional[str] = None
    use_cache: bool = True
    use_bandit: bool = True
    use_compression: bool = True
    use_cost_aware_routing: bool = True
    
    # Stage outputs
    cache_entry: Optional[Any] = None
    compressed_context: str = ""
    preference_context: Dict[str, Any] = field(default_factory=dict)
    match_similarity: Optional[float] = None
    memory_metrics: Dict[str, Any] = field(default_factory=dict)
    
    plan: Optional[Any] = None
    strategy: Optional[Any] = None
    prompt: str = ""
    
    llm_response: Optional[Any] = None
    final_response: str = ""
    
    judge_result: Optional[Any] = None
    
    # Final result
    result: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error: Optional[Exception] = None


class PipelineStage(ABC):
    """Base class for pipeline stages."""
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this pipeline stage.
        
        Args:
            context: Pipeline context with current state
        
        Returns:
            Updated context
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get stage name for logging."""
        pass


class CacheStage(PipelineStage):
    """Stage 1: Check cache, return if hit."""
    
    def __init__(self, memory_layer):
        self.memory = memory_layer
    
    def get_name(self) -> str:
        return "CacheStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Check cache and return early if hit."""
        if not context.use_cache:
            return context
        
        try:
            cache_entry, compressed_context, preference_context, match_similarity, mem_ops = \
                self.memory.retrieve_compressed(
                    query=context.query,
                    context_token_budget=500,
                    top_k=3,
                )
            
            context.cache_entry = cache_entry
            context.compressed_context = compressed_context
            context.preference_context = preference_context
            context.match_similarity = match_similarity
            context.memory_metrics = mem_ops
            
            # If cache hit, create result and mark for early return
            if cache_entry:
                # Determine cache type
                similarity = cache_entry.metadata.get("similarity") or match_similarity
                cache_type = "semantic_direct" if similarity else "exact"
                
                # Get complexity for diagnostics
                from ..orchestrator.orchestrator import QueryComplexity
                complexity = QueryComplexity.SIMPLE  # Default, will be calculated if needed
                
                # Create early return result
                context.result = {
                    "response": cache_entry.response,
                    "tokens_used": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_hit": True,
                    "cache_type": cache_type,
                    "similarity": similarity,
                    "latency_ms": 0,
                    "strategy": None,
                    "model": None,
                    "component_savings": {
                        "memory_layer": cache_entry.tokens_used if cache_entry else 0,
                        "orchestrator": 0,
                        "bandit": 0,
                        "total_savings": cache_entry.tokens_used if cache_entry else 0,
                    },
                }
                context.error = StopIteration("Cache hit - early return")  # Signal to stop pipeline
        except Exception as e:
            logger.error("Cache stage failed", error=str(e))
            context.error = e
        
        return context


class PlanningStage(PipelineStage):
    """Stage 2: Orchestrator creates plan (complexity analysis, token allocation, prompt building)."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def get_name(self) -> str:
        return "PlanningStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Create query plan with orchestrator."""
        if context.error:
            return context
        
        try:
            # Convert compressed context to list format
            retrieved_context = [context.compressed_context] if context.compressed_context else None
            
            plan = self.orchestrator.plan_query(
                query=context.query,
                token_budget=context.token_budget,
                retrieved_context=retrieved_context,
            )
            
            context.plan = plan
        except Exception as e:
            logger.error("Planning stage failed", error=str(e))
            context.error = e
        
        return context


class RoutingStage(PipelineStage):
    """Stage 3: Bandit selects strategy (model, max_tokens, temperature)."""
    
    def __init__(self, bandit):
        self.bandit = bandit
    
    def get_name(self) -> str:
        return "RoutingStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Select strategy with bandit."""
        if context.error or not context.use_bandit:
            return context
        
        try:
            complexity = context.plan.complexity.value if context.plan else "simple"
            
            if context.use_cost_aware_routing:
                strategy = self.bandit.select_strategy_cost_aware(
                    query_complexity=complexity,
                    context_quality_score=context.plan.context_quality_score if context.plan else 1.0,
                )
            else:
                strategy = self.bandit.select_strategy()
            
            context.strategy = strategy
            
            # Apply strategy to plan
            if strategy and context.plan:
                context.plan.model = strategy.model
        except Exception as e:
            logger.error("Routing stage failed", error=str(e))
            context.error = e
        
        return context


class CompressionStage(PipelineStage):
    """Stage 4: Compress query/context (orchestrator coordinates, memory layer executes)."""
    
    def __init__(self, orchestrator, memory_layer):
        self.orchestrator = orchestrator
        self.memory = memory_layer
    
    def get_name(self) -> str:
        return "CompressionStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Build prompt and compress if needed."""
        if context.error:
            return context
        
        try:
            # Build prompt
            prompt = self.orchestrator.build_prompt(context.plan, system_prompt=context.system_prompt)
            
            # Compress query if needed
            if context.use_compression and context.plan:
                original_query = context.plan.query
                compressed_query = self.memory.compress_query_if_needed(original_query)
                
                if compressed_query != original_query:
                    prompt = prompt.replace(original_query, compressed_query)
            
            context.prompt = prompt
        except Exception as e:
            logger.error("Compression stage failed", error=str(e))
            context.error = e
        
        return context


class ExecutionStage(PipelineStage):
    """Stage 5: Call LLM provider."""
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    
    def get_name(self) -> str:
        return "ExecutionStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute LLM call."""
        if context.error:
            return context
        
        try:
            # Determine max_tokens from strategy or plan
            if context.strategy:
                max_response_tokens = context.strategy.max_tokens
            elif context.plan:
                max_response_tokens = context.plan.token_budget // 2
            else:
                max_response_tokens = 512
            
            generation_params = {"max_tokens": max_response_tokens}
            
            if context.strategy:
                generation_params.update({
                    "temperature": context.strategy.temperature,
                    "top_p": context.strategy.top_p,
                    "frequency_penalty": context.strategy.frequency_penalty,
                    "presence_penalty": context.strategy.presence_penalty,
                    "n": context.strategy.n,
                })
            
            llm_response = self.llm_provider.generate(context.prompt, **generation_params)
            context.llm_response = llm_response
            context.final_response = llm_response.text
        except Exception as e:
            logger.error("Execution stage failed", error=str(e))
            context.error = e
        
        return context


class JudgmentStage(PipelineStage):
    """Stage 6: Quality judge (optional)."""
    
    def __init__(self, quality_judge):
        self.quality_judge = quality_judge
    
    def get_name(self) -> str:
        return "JudgmentStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run quality judge if enabled."""
        if context.error or not self.quality_judge:
            return context
        
        # Quality judge requires baseline comparison, which is handled in compare_with_baseline()
        # This stage is a placeholder for future use
        return context


class StorageStage(PipelineStage):
    """Stage 7: Store in cache."""
    
    def __init__(self, memory_layer):
        self.memory = memory_layer
    
    def get_name(self) -> str:
        return "StorageStage"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Store result in cache."""
        if context.error or not context.use_cache or not context.llm_response:
            return context
        
        try:
            self.memory.store(
                query=context.query,
                response=context.final_response,
                tokens_used=context.llm_response.tokens_used,
                metadata={
                    "similarity": context.match_similarity,
                    "cache_type": context.result.get("cache_type"),
                },
            )
        except Exception as e:
            logger.warning("Storage stage failed", error=str(e))
            # Don't fail the pipeline if storage fails
        
        return context


class QueryPipeline:
    """Pipeline for processing queries through multiple stages."""
    
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
    
    def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Execute pipeline stages in order.
        
        Args:
            context: Initial pipeline context
        
        Returns:
            Final result dictionary
        """
        for stage in self.stages:
            if context.error:
                if isinstance(context.error, StopIteration):
                    # Early return (e.g., cache hit)
                    break
                # Other errors should be raised
                raise context.error
            
            logger.debug(f"Executing stage: {stage.get_name()}")
            context = stage.execute(context)
        
        return context.result







