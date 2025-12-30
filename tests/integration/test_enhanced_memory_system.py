"""
Enhanced Memory System Test
Integrates Mem0, LLM-Lingua, and RouterBench concepts

This test compares:
- Baseline: Current memory + bandit system
- Enhanced: Mem0-style memory + LLM-Lingua compression + RouterBench routing
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from tokenomics.memory.memory_layer import SmartMemoryLayer
from tokenomics.bandit.bandit import BanditOptimizer, Strategy
from tokenomics.orchestrator.orchestrator import TokenAwareOrchestrator

logger = structlog.get_logger()


# ============================================================================
# Enhanced Memory Layer (Mem0-inspired)
# ============================================================================

@dataclass
class UserPreference:
    """User preference learned from interactions."""
    preference_type: str  # "tone", "format", "detail_level", "style"
    value: str
    confidence: float = 0.5
    examples: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryEntity:
    """Structured memory entity (Mem0-style)."""
    entity_id: str
    entity_type: str  # "person", "concept", "preference", "fact"
    content: str
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # {relation_type: [entity_ids]}
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0


class EnhancedMemoryLayer:
    """
    Enhanced memory layer inspired by Mem0 architecture.
    
    Features:
    - Structured entities and relationships
    - User preference learning
    - Tone and format pattern recognition
    - Adaptive personalization
    """
    
    def __init__(
        self,
        base_memory: SmartMemoryLayer,
        enable_entity_extraction: bool = True,
        enable_preference_learning: bool = True,
        enable_relationship_tracking: bool = True,
    ):
        self.base_memory = base_memory
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_preference_learning = enable_preference_learning
        self.enable_relationship_tracking = enable_relationship_tracking
        
        # Entity storage
        self.entities: Dict[str, MemoryEntity] = {}
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # {entity_type: [entity_ids]}
        
        # User preferences
        self.user_preferences: Dict[str, UserPreference] = {}  # {preference_type: UserPreference}
        
        # Relationship graph
        self.relationships: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        logger.info(
            "EnhancedMemoryLayer initialized",
            entity_extraction=enable_entity_extraction,
            preference_learning=enable_preference_learning,
        )
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text (simplified version).
        In production, use NER model or LLM.
        
        Returns:
            List of (entity_text, entity_type)
        """
        if not self.enable_entity_extraction:
            return []
        
        # Simple heuristic extraction (in production, use proper NER)
        entities = []
        words = text.split()
        
        # Look for capitalized phrases (potential entities)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's part of a multi-word entity
                entity = word
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    entity += " " + words[i + 1]
                
                # Classify entity type (simplified)
                entity_type = "concept"
                if any(kw in word.lower() for kw in ["user", "i", "my", "me"]):
                    entity_type = "person"
                
                entities.append((entity, entity_type))
        
        return entities[:5]  # Limit to top 5
    
    def learn_preference(
        self,
        query: str,
        response: str,
        interaction_metadata: Optional[Dict] = None,
    ):
        """Learn user preferences from interaction."""
        if not self.enable_preference_learning:
            return
        
        # Analyze tone
        tone_indicators = {
            "formal": ["please", "would", "could", "thank you"],
            "casual": ["hey", "what's", "gimme", "yeah"],
            "technical": ["implement", "algorithm", "optimize", "architecture"],
            "simple": ["explain", "how", "what is", "tell me"],
        }
        
        detected_tone = "neutral"
        for tone, indicators in tone_indicators.items():
            if any(ind in query.lower() for ind in indicators):
                detected_tone = tone
                break
        
        # Update tone preference
        if detected_tone != "neutral":
            if "tone" not in self.user_preferences:
                self.user_preferences["tone"] = UserPreference(
                    preference_type="tone",
                    value=detected_tone,
                    examples=[query[:100]],
                )
            else:
                pref = self.user_preferences["tone"]
                if pref.value == detected_tone:
                    pref.confidence = min(1.0, pref.confidence + 0.1)
                else:
                    pref.confidence = max(0.0, pref.confidence - 0.05)
                pref.examples.append(query[:100])
                pref.last_updated = datetime.now()
        
        # Analyze format preference
        format_indicators = {
            "list": ["list", "steps", "items", "bullets"],
            "paragraph": ["explain", "describe", "tell me about"],
            "code": ["code", "example", "implement", "function"],
        }
        
        detected_format = "paragraph"
        for fmt, indicators in format_indicators.items():
            if any(ind in query.lower() for ind in indicators):
                detected_format = fmt
                break
        
        if "format" not in self.user_preferences:
            self.user_preferences["format"] = UserPreference(
                preference_type="format",
                value=detected_format,
                examples=[query[:100]],
            )
    
    def store_enhanced(
        self,
        query: str,
        response: str,
        tokens_used: int = 0,
        metadata: Optional[Dict] = None,
    ):
        """Store with enhanced features."""
        # Store in base memory
        entry = self.base_memory.store(query, response, tokens_used, metadata)
        
        # Extract and store entities
        if self.enable_entity_extraction:
            entities = self.extract_entities(query + " " + response)
            for entity_text, entity_type in entities:
                entity_id = f"entity_{hash(entity_text) % 1000000}"
                if entity_id not in self.entities:
                    self.entities[entity_id] = MemoryEntity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        content=entity_text,
                        metadata={"source_query": query[:100]},
                    )
                    self.entity_index[entity_type].append(entity_id)
                else:
                    self.entities[entity_id].access_count += 1
        
        # Learn preferences
        if self.enable_preference_learning:
            self.learn_preference(query, response, metadata)
        
        return entry
    
    def retrieve_enhanced(
        self,
        query: str,
        top_k: int = 5,
        use_preferences: bool = True,
    ) -> Tuple[Optional[Any], List[Any], Dict[str, Any]]:
        """
        Enhanced retrieval with preference-aware context.
        
        Returns:
            (exact_match, semantic_matches, preference_context)
        """
        # Base retrieval
        exact_match, semantic_matches = self.base_memory.retrieve(query, top_k)
        
        # Build preference context
        preference_context = {}
        if use_preferences and self.enable_preference_learning:
            for pref_type, pref in self.user_preferences.items():
                if pref.confidence > 0.5:
                    preference_context[pref_type] = pref.value
        
        return exact_match, semantic_matches, preference_context
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get user context for personalization."""
        context = {
            "preferences": {
                pref_type: {
                    "value": pref.value,
                    "confidence": pref.confidence,
                }
                for pref_type, pref in self.user_preferences.items()
            },
            "entity_count": len(self.entities),
            "entity_types": list(self.entity_index.keys()),
        }
        return context


# ============================================================================
# LLM-Lingua Style Compression
# ============================================================================

class PromptCompressor:
    """
    Prompt compression inspired by LLM-Lingua.
    
    Techniques:
    - Token-level compression
    - Importance-based filtering
    - Summarization for long contexts
    """
    
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4
    
    def compress_context(
        self,
        context: List[str],
        target_tokens: int,
        preserve_important: bool = True,
    ) -> str:
        """
        Compress context to fit token budget.
        
        Strategy:
        1. Identify important sentences (containing keywords, questions, etc.)
        2. Preserve important sentences
        3. Summarize or truncate less important parts
        """
        if not context:
            return ""
        
        full_text = " ".join(context)
        current_tokens = self.count_tokens(full_text)
        
        if current_tokens <= target_tokens:
            return full_text
        
        # Split into sentences
        sentences = [s.strip() for s in full_text.split('.') if s.strip()]
        
        # Score sentences by importance
        important_keywords = ["important", "key", "main", "primary", "essential", "critical"]
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            # Check for keywords
            if any(kw in sentence.lower() for kw in important_keywords):
                score += 2
            # Check for questions
            if '?' in sentence:
                score += 1
            # Check length (medium-length sentences often more informative)
            if 20 < len(sentence) < 200:
                score += 1
            sentence_scores.append((i, sentence, score))
        
        # Sort by importance
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Build compressed text
        compressed_sentences = []
        token_count = 0
        
        # First, add important sentences
        for i, sentence, score in sentence_scores:
            if score > 0 and token_count < target_tokens * 0.7:
                sentence_tokens = self.count_tokens(sentence)
                if token_count + sentence_tokens <= target_tokens:
                    compressed_sentences.append((i, sentence))
                    token_count += sentence_tokens
        
        # Fill remaining space with other sentences
        remaining_budget = target_tokens - token_count
        for i, sentence, score in sentence_scores:
            if (i, sentence) not in compressed_sentences:
                sentence_tokens = self.count_tokens(sentence)
                if token_count + sentence_tokens <= target_tokens:
                    compressed_sentences.append((i, sentence))
                    token_count += sentence_tokens
                else:
                    # Truncate if needed
                    if remaining_budget > 10:
                        truncated = sentence[:remaining_budget * 4]  # Rough char estimate
                        compressed_sentences.append((i, truncated + "..."))
                        break
        
        # Reorder by original position
        compressed_sentences.sort(key=lambda x: x[0])
        compressed_text = ". ".join(s for _, s in compressed_sentences)
        
        return compressed_text
    
    def compress_prompt(
        self,
        system_prompt: Optional[str],
        context: Optional[str],
        query: str,
        target_tokens: int,
    ) -> Tuple[str, int]:
        """
        Compress entire prompt to fit token budget.
        
        Returns:
            (compressed_prompt, actual_tokens)
        """
        parts = []
        
        # Always preserve query (highest priority)
        query_tokens = self.count_tokens(query)
        remaining_budget = target_tokens - query_tokens
        
        # Compress system prompt if needed
        if system_prompt:
            sys_tokens = self.count_tokens(system_prompt)
            if sys_tokens > remaining_budget * 0.3:
                # Compress system prompt
                target_sys_tokens = int(remaining_budget * 0.3)
                # Simple truncation for system prompt
                if self.tokenizer:
                    encoded = self.tokenizer.encode(system_prompt)
                    compressed_sys = self.tokenizer.decode(encoded[:target_sys_tokens])
                else:
                    compressed_sys = system_prompt[:target_sys_tokens * 4]
                parts.append(compressed_sys)
                remaining_budget -= self.count_tokens(compressed_sys)
            else:
                parts.append(system_prompt)
                remaining_budget -= sys_tokens
        
        # Compress context if needed
        if context:
            ctx_tokens = self.count_tokens(context)
            if ctx_tokens > remaining_budget:
                compressed_ctx = self.compress_context([context], remaining_budget)
                parts.append(compressed_ctx)
            else:
                parts.append(context)
        
        # Add query
        parts.append(query)
        
        final_prompt = "\n\n".join(parts)
        final_tokens = self.count_tokens(final_prompt)
        
        return final_prompt, final_tokens


# ============================================================================
# RouterBench-Style Bandit Enhancement
# ============================================================================

@dataclass
class RoutingMetrics:
    """Routing metrics for cost-quality evaluation."""
    cost_per_token: float
    quality_score: float
    latency_ms: float
    reliability: float = 1.0  # Success rate
    
    def cost_quality_ratio(self) -> float:
        """Compute cost-quality ratio (higher is better)."""
        if self.cost_per_token == 0:
            return float('inf')
        return self.quality_score / self.cost_per_token
    
    def efficiency_score(self) -> float:
        """Combined efficiency score."""
        return (
            self.quality_score * 0.5 +
            (1.0 / (1.0 + self.cost_per_token * 100)) * 0.3 +
            (1.0 / (1.0 + self.latency_ms / 1000)) * 0.2
        )


class RouterBenchBandit(BanditOptimizer):
    """
    Enhanced bandit with RouterBench-style cost-quality evaluation.
    
    Features:
    - Cost-aware routing
    - Quality metrics tracking
    - Multi-objective optimization
    - Reliability tracking
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # RouterBench metrics
        self.routing_metrics: Dict[str, RoutingMetrics] = {}
        self.model_costs: Dict[str, float] = {
            "gpt-4o-mini": 0.15 / 1_000_000,  # $0.15 per 1M tokens
            "gpt-4o": 2.50 / 1_000_000,  # $2.50 per 1M tokens
            "gemini-flash": 0.075 / 1_000_000,
            "gemini-pro": 0.50 / 1_000_000,
        }
        
        logger.info("RouterBenchBandit initialized")
    
    def compute_reward_routerbench(
        self,
        arm_id: str,
        quality_score: float,
        tokens_used: int,
        latency_ms: float,
        model: str,
    ) -> float:
        """
        Compute reward using RouterBench methodology.
        
        Considers:
        - Quality score
        - Cost efficiency
        - Latency
        - Reliability
        """
        # Get cost per token for model
        cost_per_token = self.model_costs.get(model, 0.0001)
        total_cost = cost_per_token * tokens_used
        
        # Update routing metrics
        if arm_id not in self.routing_metrics:
            self.routing_metrics[arm_id] = RoutingMetrics(
                cost_per_token=cost_per_token,
                quality_score=quality_score,
                latency_ms=latency_ms,
            )
        else:
            metrics = self.routing_metrics[arm_id]
            # Update with moving average
            alpha = 0.3
            metrics.cost_per_token = alpha * cost_per_token + (1 - alpha) * metrics.cost_per_token
            metrics.quality_score = alpha * quality_score + (1 - alpha) * metrics.quality_score
            metrics.latency_ms = alpha * latency_ms + (1 - alpha) * metrics.latency_ms
        
        # Compute efficiency score
        efficiency = self.routing_metrics[arm_id].efficiency_score()
        
        # Reward = efficiency - penalty for high cost
        reward = efficiency - (total_cost * 1000)  # Scale cost penalty
        
        return reward
    
    def select_strategy_routerbench(
        self,
        query_complexity: str = "medium",
        budget_constraint: Optional[float] = None,
    ) -> Optional[Strategy]:
        """
        Select strategy using RouterBench cost-quality routing.
        
        Considers:
        - Query complexity
        - Budget constraints
        - Historical cost-quality ratios
        """
        if not self.arms:
            return None
        
        # Filter strategies by constraints
        candidates = {}
        for arm_id, arm in self.arms.items():
            strategy = arm.strategy
            
            # Check budget constraint
            if budget_constraint:
                model_cost = self.model_costs.get(strategy.model, 0.0001)
                estimated_cost = model_cost * strategy.max_tokens
                if estimated_cost > budget_constraint:
                    continue
            
            # Get routing metrics
            if arm_id in self.routing_metrics:
                metrics = self.routing_metrics[arm_id]
                # Score based on cost-quality ratio
                score = metrics.cost_quality_ratio() * (1.0 / (1.0 + metrics.latency_ms / 1000))
            else:
                # Unknown strategy: use exploration
                score = 1.0
            
            candidates[arm_id] = (arm, score)
        
        if not candidates:
            # Fallback to base selection
            return super().select_strategy()
        
        # Select best candidate
        best_arm_id = max(candidates.items(), key=lambda x: x[1][1])[0]
        return self.arms[best_arm_id].strategy
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get RouterBench-style routing statistics."""
        stats = {
            "routing_metrics": {
                arm_id: {
                    "cost_per_token": metrics.cost_per_token,
                    "quality_score": metrics.quality_score,
                    "latency_ms": metrics.latency_ms,
                    "cost_quality_ratio": metrics.cost_quality_ratio(),
                    "efficiency_score": metrics.efficiency_score(),
                }
                for arm_id, metrics in self.routing_metrics.items()
            },
            "best_cost_quality": None,
            "total_routes": len(self.routing_metrics),
        }
        
        if self.routing_metrics:
            best = max(
                self.routing_metrics.items(),
                key=lambda x: x[1].cost_quality_ratio()
            )
            stats["best_cost_quality"] = {
                "arm_id": best[0],
                "ratio": best[1].cost_quality_ratio(),
            }
        
        return stats


# ============================================================================
# Enhanced Platform Integration
# ============================================================================

class EnhancedTokenomicsPlatform:
    """
    Enhanced platform integrating all improvements.
    """
    
    def __init__(self, config: Optional[TokenomicsConfig] = None):
        self.config = config or TokenomicsConfig.from_env()
        
        # Initialize base components
        base_memory = SmartMemoryLayer(
            use_exact_cache=self.config.memory.use_exact_cache,
            use_semantic_cache=self.config.memory.use_semantic_cache,
            vector_store_type=self.config.memory.vector_store,
            embedding_model=self.config.memory.embedding_model,
            similarity_threshold=self.config.memory.similarity_threshold,
            cache_size=self.config.memory.cache_size,
        )
        
        # Enhanced memory
        self.memory = EnhancedMemoryLayer(
            base_memory=base_memory,
            enable_entity_extraction=True,
            enable_preference_learning=True,
            enable_relationship_tracking=True,
        )
        
        # Prompt compressor
        self.compressor = PromptCompressor(
            compression_ratio=self.config.orchestrator.compression_ratio,
        )
        
        # Enhanced bandit
        self.bandit = RouterBenchBandit(
            algorithm=self.config.bandit.algorithm,
            exploration_rate=self.config.bandit.exploration_rate,
            reward_lambda=self.config.bandit.reward_lambda,
        )
        
        # Orchestrator
        self.orchestrator = TokenAwareOrchestrator(
            default_token_budget=self.config.orchestrator.default_token_budget,
            max_context_tokens=self.config.orchestrator.max_context_tokens,
            use_knapsack_optimization=self.config.orchestrator.use_knapsack_optimization,
            compression_ratio=self.config.orchestrator.compression_ratio,
            enable_multi_model_routing=self.config.orchestrator.enable_multi_model_routing,
            provider=self.config.llm.provider,
        )
        
        # LLM provider
        from tokenomics.llm_providers import GeminiProvider, OpenAIProvider, vLLMProvider
        
        llm_cfg = self.config.llm
        if llm_cfg.provider == "gemini":
            self.llm = GeminiProvider(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                project_id=llm_cfg.project_id,
                location=llm_cfg.location,
            )
        elif llm_cfg.provider == "openai":
            self.llm = OpenAIProvider(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
            )
        elif llm_cfg.provider == "vllm":
            self.llm = vLLMProvider(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url,
            )
        else:
            self.llm = OpenAIProvider(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
            )
        
        logger.info("EnhancedTokenomicsPlatform initialized")
    
    def query(
        self,
        query: str,
        token_budget: Optional[int] = None,
        use_cache: bool = True,
        use_bandit: bool = True,
    ) -> Dict:
        """Process query with enhanced features."""
        start_time = time.time()
        
        # 1. Enhanced retrieval
        exact_match, semantic_matches, preference_context = self.memory.retrieve_enhanced(
            query,
            use_preferences=True,
        )
        
        if exact_match:
            logger.info("Exact cache hit", tokens_saved=exact_match.tokens_used)
            return {
                "response": exact_match.response,
                "tokens_used": 0,
                "cache_hit": True,
                "cache_type": "exact",
                "latency_ms": 0,
                "preference_context": preference_context,
            }
        
        # 2. RouterBench strategy selection
        strategy = None
        if use_bandit:
            complexity = self.orchestrator.analyze_complexity(query).value
            strategy = self.bandit.select_strategy_routerbench(
                query_complexity=complexity,
            )
            if not strategy:
                strategy = self.bandit.select_strategy()
        
        # 3. Create plan
        retrieved_context = [m.response for m in semantic_matches[:3]] if semantic_matches else None
        plan = self.orchestrator.plan_query(
            query=query,
            token_budget=token_budget,
            retrieved_context=retrieved_context,
        )
        
        # 4. Apply preference-based personalization
        if preference_context:
            # Modify system prompt based on preferences
            tone = preference_context.get("tone", "neutral")
            format_pref = preference_context.get("format", "paragraph")
            # This would be used to customize the prompt
        
        # 5. Compress prompt with LLM-Lingua techniques
        system_prompt = "You are a helpful AI assistant."
        context_text = plan.compressed_prompt if plan.compressed_prompt else None
        
        compressed_prompt, actual_tokens = self.compressor.compress_prompt(
            system_prompt=system_prompt,
            context=context_text,
            query=query,
            target_tokens=plan.token_budget,
        )
        
        # 6. Generate response
        model = strategy.model if strategy else plan.model
        generation_params = {
            "max_tokens": plan.token_budget // 2,
        }
        if strategy:
            generation_params.update({
                "temperature": strategy.temperature,
                "top_p": strategy.top_p,
            })
        
        # Use model from strategy or plan
        llm_response = self.llm.generate(
            compressed_prompt,
            **generation_params,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 7. Store with enhanced features
        if use_cache:
            self.memory.store_enhanced(
                query=query,
                response=llm_response.text,
                tokens_used=llm_response.tokens_used,
                metadata={
                    "strategy": strategy.arm_id if strategy else None,
                    "model": model,
                    "preferences": preference_context,
                },
            )
        
        # 8. Update RouterBench bandit
        if use_bandit and strategy:
            quality_score = 1.0  # Placeholder - would use actual quality scorer
            reward = self.bandit.compute_reward_routerbench(
                arm_id=strategy.arm_id,
                quality_score=quality_score,
                tokens_used=llm_response.tokens_used,
                latency_ms=latency_ms,
                model=model,
            )
            self.bandit.update(strategy.arm_id, reward)
        
        return {
            "response": llm_response.text,
            "tokens_used": llm_response.tokens_used,
            "cache_hit": False,
            "cache_type": "semantic" if semantic_matches else None,
            "latency_ms": latency_ms,
            "strategy": strategy.arm_id if strategy else None,
            "model": model,
            "preference_context": preference_context,
            "compressed_tokens": actual_tokens,
            "plan": plan,
        }


# ============================================================================
# Test Suite
# ============================================================================

def run_comparison_test():
    """Run comparison test between baseline and enhanced systems."""
    print("\n" + "="*80)
    print("ENHANCED MEMORY SYSTEM TEST")
    print("="*80)
    
    # Test queries
    test_queries = [
        "How to build an agentic system with Autogen?",
        "What is machine learning?",
        "Explain neural networks in simple terms",
        "How to optimize Python code?",
        "What are the best practices for API design?",
    ]
    
    results = {
        "baseline": [],
        "enhanced": [],
        "comparison": {},
    }
    
    # Initialize platforms
    print("\n[1/4] Initializing platforms...")
    
    # Configure to use OpenAI (same as app.py)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
    
    # Set OpenAI API key from environment variable
    # Make sure OPENAI_API_KEY is set in your .env file
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    
    # Create config with OpenAI
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = os.environ["OPENAI_API_KEY"]
    config.memory.use_semantic_cache = False
    config.memory.cache_size = 100
    
    baseline_platform = TokenomicsPlatform(config=config)
    enhanced_platform = EnhancedTokenomicsPlatform(config=config)
    
    # Add strategies
    strategies = [
        Strategy(arm_id="fast", model="gpt-4o-mini", max_tokens=500, temperature=0.7),
        Strategy(arm_id="balanced", model="gpt-4o-mini", max_tokens=1000, temperature=0.5),
        Strategy(arm_id="powerful", model="gpt-4o", max_tokens=2000, temperature=0.3),
    ]
    baseline_platform.bandit.add_strategies(strategies)
    enhanced_platform.bandit.add_strategies(strategies)
    
    print(f"[2/4] Running {len(test_queries)} queries on baseline system...")
    for i, query in enumerate(test_queries, 1):
        print(f"  Query {i}/{len(test_queries)}: {query[:50]}...")
        try:
            result = baseline_platform.query(query, use_cache=True, use_bandit=True)
            results["baseline"].append({
                "query": query,
                "tokens": result["tokens_used"],
                "latency": result["latency_ms"],
                "cache_hit": result["cache_hit"],
                "strategy": result.get("strategy"),
            })
        except Exception as e:
            print(f"    Error: {e}")
            results["baseline"].append({
                "query": query,
                "error": str(e),
            })
    
    print(f"[3/4] Running {len(test_queries)} queries on enhanced system...")
    for i, query in enumerate(test_queries, 1):
        print(f"  Query {i}/{len(test_queries)}: {query[:50]}...")
        try:
            result = enhanced_platform.query(query, use_cache=True, use_bandit=True)
            results["enhanced"].append({
                "query": query,
                "tokens": result["tokens_used"],
                "latency": result["latency_ms"],
                "cache_hit": result["cache_hit"],
                "strategy": result.get("strategy"),
                "preference_context": result.get("preference_context"),
                "compressed_tokens": result.get("compressed_tokens"),
            })
        except Exception as e:
            print(f"    Error: {e}")
            results["enhanced"].append({
                "query": query,
                "error": str(e),
            })
    
    # Compare results
    print("[4/4] Analyzing results...")
    
    baseline_tokens = sum(r.get("tokens", 0) for r in results["baseline"] if "tokens" in r)
    enhanced_tokens = sum(r.get("tokens", 0) for r in results["enhanced"] if "tokens" in r)
    
    baseline_latency = sum(r.get("latency", 0) for r in results["baseline"] if "latency" in r)
    enhanced_latency = sum(r.get("latency", 0) for r in results["enhanced"] if "latency" in r)
    
    baseline_cache_hits = sum(1 for r in results["baseline"] if r.get("cache_hit"))
    enhanced_cache_hits = sum(1 for r in results["enhanced"] if r.get("cache_hit"))
    
    results["comparison"] = {
        "token_savings": baseline_tokens - enhanced_tokens,
        "token_savings_percent": ((baseline_tokens - enhanced_tokens) / baseline_tokens * 100) if baseline_tokens > 0 else 0,
        "latency_reduction": baseline_latency - enhanced_latency,
        "latency_reduction_percent": ((baseline_latency - enhanced_latency) / baseline_latency * 100) if baseline_latency > 0 else 0,
        "cache_hit_rate_baseline": (baseline_cache_hits / len(test_queries)) * 100,
        "cache_hit_rate_enhanced": (enhanced_cache_hits / len(test_queries)) * 100,
    }
    
    # Get enhanced features stats
    user_context = enhanced_platform.memory.get_user_context()
    routing_stats = enhanced_platform.bandit.get_routing_stats()
    
    results["enhanced_features"] = {
        "user_preferences": user_context.get("preferences", {}),
        "entities_learned": user_context.get("entity_count", 0),
        "routing_metrics": routing_stats,
    }
    
    return results


def save_results(results: Dict, filename: str = "enhanced_memory_test_results.json"):
    """Save test results to file."""
    # Convert dataclasses to dicts
    def convert(obj):
        if hasattr(obj, '__dict__'):
            return {k: convert(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    converted = convert(results)
    
    with open(filename, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


def print_summary(results: Dict):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    comp = results["comparison"]
    print(f"\nToken Usage:")
    print(f"  Baseline: {sum(r.get('tokens', 0) for r in results['baseline'] if 'tokens' in r)} tokens")
    print(f"  Enhanced: {sum(r.get('tokens', 0) for r in results['enhanced'] if 'tokens' in r)} tokens")
    print(f"  Savings: {comp['token_savings']} tokens ({comp['token_savings_percent']:.2f}%)")
    
    print(f"\nLatency:")
    print(f"  Baseline: {sum(r.get('latency', 0) for r in results['baseline'] if 'latency' in r):.2f} ms")
    print(f"  Enhanced: {sum(r.get('latency', 0) for r in results['enhanced'] if 'latency' in r):.2f} ms")
    print(f"  Reduction: {comp['latency_reduction']:.2f} ms ({comp['latency_reduction_percent']:.2f}%)")
    
    print(f"\nCache Performance:")
    print(f"  Baseline hit rate: {comp['cache_hit_rate_baseline']:.2f}%")
    print(f"  Enhanced hit rate: {comp['cache_hit_rate_enhanced']:.2f}%")
    
    if "enhanced_features" in results:
        features = results["enhanced_features"]
        print(f"\nEnhanced Features:")
        print(f"  User preferences learned: {len(features.get('user_preferences', {}))}")
        print(f"  Entities extracted: {features.get('entities_learned', 0)}")
        
        routing = features.get("routing_metrics", {})
        if routing.get("best_cost_quality"):
            best = routing["best_cost_quality"]
            print(f"  Best cost-quality ratio: {best['arm_id']} ({best['ratio']:.4f})")


if __name__ == "__main__":
    print("Starting Enhanced Memory System Test...")
    print("This test compares baseline vs enhanced (Mem0 + LLM-Lingua + RouterBench) system")
    
    try:
        results = run_comparison_test()
        save_results(results)
        print_summary(results)
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

