"""Main integration layer for Tokenomics platform."""

from typing import Optional, Dict, Any
import structlog

from .config import TokenomicsConfig
from .memory.memory_layer import SmartMemoryLayer
from .orchestrator.orchestrator import TokenAwareOrchestrator, QueryPlan
from .bandit.bandit import BanditOptimizer, Strategy
from .llm_providers import LLMProvider, GeminiProvider, OpenAIProvider, vLLMProvider
from .judge import QualityJudge

logger = structlog.get_logger()


class TokenomicsPlatform:
    """Main platform integrating all components."""
    
    def __init__(self, config: Optional[TokenomicsConfig] = None):
        """
        Initialize Tokenomics platform.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or TokenomicsConfig.from_env()
        
        # Initialize components with enhanced features (Mem0 + LLM-Lingua)
        # Pass persistent cache path to MemoryCache
        persistent_cache_path = self.config.memory.persistent_cache_path if self.config.memory.persistent_cache_path else None
        
        self.memory = SmartMemoryLayer(
            use_exact_cache=self.config.memory.use_exact_cache,
            use_semantic_cache=self.config.memory.use_semantic_cache,
            vector_store_type=self.config.memory.vector_store,
            embedding_model=self.config.memory.embedding_model,
            similarity_threshold=self.config.memory.similarity_threshold,
            direct_return_threshold=self.config.memory.direct_return_threshold,
            cache_size=self.config.memory.cache_size,
            eviction_policy=self.config.memory.eviction_policy,
            ttl_seconds=self.config.memory.ttl_seconds,
            persistent_storage=persistent_cache_path,
            enable_compression=True,  # LLM-Lingua style compression
            enable_preferences=True,  # Mem0 style preference learning
            enable_llmlingua=self.config.memory.enable_llmlingua,
            llmlingua_model=self.config.memory.llmlingua_model,
            llmlingua_compression_ratio=self.config.memory.llmlingua_compression_ratio,
            compress_query_threshold_tokens=self.config.memory.compress_query_threshold_tokens,
            compress_query_threshold_chars=self.config.memory.compress_query_threshold_chars,
            use_active_retrieval=self.config.memory.use_active_retrieval,
            active_retrieval_max_iterations=self.config.memory.active_retrieval_max_iterations,
            active_retrieval_min_relevance=self.config.memory.active_retrieval_min_relevance,
        )
        
        self.orchestrator = TokenAwareOrchestrator(
            default_token_budget=self.config.orchestrator.default_token_budget,
            max_context_tokens=self.config.orchestrator.max_context_tokens,
            use_knapsack_optimization=self.config.orchestrator.use_knapsack_optimization,
            compression_ratio=self.config.orchestrator.compression_ratio,
            enable_multi_model_routing=self.config.orchestrator.enable_multi_model_routing,
            provider=self.config.llm.provider,
            max_context_ratio=self.config.orchestrator.max_context_ratio,
            min_response_ratio=self.config.orchestrator.min_response_ratio,
        )
        
        self.bandit = BanditOptimizer(
            algorithm=self.config.bandit.algorithm,
            exploration_rate=self.config.bandit.exploration_rate,
            reward_lambda=self.config.bandit.reward_lambda,
            reset_frequency=self.config.bandit.reset_frequency,
            contextual=self.config.bandit.contextual,
            state_file=self.config.bandit.state_file,
            auto_save=self.config.bandit.auto_save,
        )
        
        # Initialize LLM provider
        self.llm_provider = self._create_llm_provider()
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Load bandit state after strategies are added
        if self.config.bandit.state_file:
            try:
                self.bandit.load_state()
            except Exception as e:
                logger.warning("Failed to load bandit state on initialization", error=str(e))
        
        # Initialize quality judge if enabled
        self.quality_judge = None
        if self.config.judge.enabled:
            try:
                self.quality_judge = QualityJudge(self.config.judge)
                logger.info("Quality judge initialized", provider=self.config.judge.provider, model=self.config.judge.model)
            except Exception as e:
                logger.warning("Failed to initialize quality judge", error=str(e))
        
        # Initialize rate limiter (default: 10 requests/second, capacity 20)
        try:
            from .resilience.rate_limiter import RateLimiter
            self.rate_limiter = RateLimiter(rate=10.0, capacity=20.0)
        except Exception as e:
            logger.warning("Failed to initialize rate limiter", error=str(e))
            self.rate_limiter = None
        
        # Initialize state manager
        try:
            from .state.state_manager import StateManager
            state_file = getattr(self.config, 'state_file', 'tokenomics_state.json')
            self.state_manager = StateManager(state_file=state_file)
        except Exception as e:
            logger.warning("Failed to initialize state manager", error=str(e))
            self.state_manager = None
        
        # Initialize metrics collector for observability
        try:
            from .observability.metrics_collector import MetricsCollector
            self.metrics_collector = MetricsCollector()
        except Exception as e:
            logger.warning("Failed to initialize metrics collector", error=str(e))
            self.metrics_collector = None
        
        # Initialize unified data collector and all ML models
        try:
            from .ml.unified_data_collector import UnifiedDataCollector
            from .ml.token_predictor import TokenPredictor
            from .ml.escalation_predictor import EscalationPredictor
            from .ml.complexity_classifier import ComplexityClassifier
            
            # Create unified data collector (single database for all ML models)
            unified_collector = UnifiedDataCollector(db_path="ml_training_data.db")
            
            # Initialize token predictor with unified collector
            self.token_predictor = TokenPredictor(data_collector=unified_collector)
            logger.info("TokenPredictor initialized with unified database")
            
            # Initialize escalation predictor with unified collector
            self.escalation_predictor = EscalationPredictor(
                data_collector=unified_collector
            )
            logger.info("EscalationPredictor initialized with unified database")
            
            # Initialize complexity classifier with unified collector
            # Pass tokenizer from orchestrator for token counting
            orchestrator_tokenizer = self.orchestrator.tokenizer if hasattr(self.orchestrator, 'tokenizer') else None
            self.complexity_classifier = ComplexityClassifier(
                data_collector=unified_collector,
                tokenizer=orchestrator_tokenizer
            )
            logger.info("ComplexityClassifier initialized")
        except Exception as e:
            logger.warning("Failed to initialize ML models with unified database", error=str(e))
            # Fallback: try to initialize escalation predictor with old database
            try:
                from .ml.escalation_predictor import EscalationPredictor
                from .ml.escalation_data_collector import EscalationDataCollector
                escalation_data_collector = EscalationDataCollector(
                    db_path="escalation_prediction_data.db"
                )
                self.escalation_predictor = EscalationPredictor(
                    data_collector=escalation_data_collector
                )
                logger.info("EscalationPredictor initialized (fallback)")
            except Exception as e2:
                logger.warning("Failed to initialize escalation predictor (fallback)", error=str(e2))
                self.escalation_predictor = None
            self.complexity_classifier = None
        
        # Cascading inference settings
        self.cascading_enabled = self.config.cascading.enabled
        self.cascading_quality_threshold = self.config.cascading.quality_threshold
        self.cascading_cheap_model = self.config.cascading.cheap_model
        self.cascading_premium_model = self.config.cascading.premium_model
        self.cascading_use_lightweight = self.config.cascading.use_lightweight_check
        
        # Cascading metrics tracking
        self.cascading_metrics = {
            "total_queries": 0,
            "cheap_model_used": 0,
            "escalations": 0,
            "escalation_rate": 0.0,
            "cost_savings": 0.0,
            # Escalation prediction metrics
            "predicted_escalations": 0,  # Times we predicted escalation
            "skipped_cheap": 0,  # Times we skipped cheap model
            "prediction_accuracy": 0.0,  # How often predictions were correct
            "false_positives": 0,  # Predicted escalation but didn't need it
            "false_negatives": 0,  # Didn't predict but needed escalation
        }
        
        logger.info(
            "TokenomicsPlatform initialized",
            cascading_enabled=self.cascading_enabled,
            cascading_threshold=self.cascading_quality_threshold,
        )
    
    def _find_cheap_arm_id(self) -> Optional[str]:
        """Find arm ID for cheap cascading model."""
        if not self.bandit:
            return None
        
        for arm_id, arm in self.bandit.arms.items():
            if arm.strategy.model == self.cascading_cheap_model:
                return arm_id
        
        return None
    
    def _create_llm_provider(self) -> LLMProvider:
        """Create LLM provider based on config."""
        llm_cfg = self.config.llm
        
        if llm_cfg.provider == "gemini":
            return GeminiProvider(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                project_id=llm_cfg.project_id,
                location=llm_cfg.location,
            )
        elif llm_cfg.provider == "openai":
            return OpenAIProvider(
                model=llm_cfg.model,
                api_key=llm_cfg.api_key,
                base_url=llm_cfg.base_url,
            )
        elif llm_cfg.provider == "vllm":
            return vLLMProvider(
                model=llm_cfg.model,
                base_url=llm_cfg.base_url or "http://localhost:8000",
                api_key=llm_cfg.api_key,
            )
        else:
            raise ValueError(f"Unknown provider: {llm_cfg.provider}")
    
    def _initialize_default_strategies(self):
        """Initialize default bandit strategies for support use cases."""
        # Get the provider from config to determine model names
        provider = self.config.llm.provider
        
        # Strategy settings optimized for support:
        # - Realistic temperature (0.2-0.4) for consistent but natural responses
        # - Token budgets that allow detailed answers when needed
        # - Memory modes that leverage cache and context injection
        if provider == "openai":
            strategies = [
                Strategy(
                    arm_id="cheap",
                    model="gpt-4o-mini",
                    max_tokens=300,
                    temperature=0.2,  # Low temp for simple, consistent answers
                    memory_mode="light",
                    metadata={"query_type": "simple"},
                ),
                Strategy(
                    arm_id="balanced",
                    model="gpt-4o-mini",
                    max_tokens=600,
                    temperature=0.3,  # Standard support temperature
                    memory_mode="rich",  # Uses context injection
                    metadata={"query_type": "medium"},
                ),
                Strategy(
                    arm_id="premium",
                    model="gpt-4o",  # Stronger model for complex queries
                    max_tokens=1000,
                    temperature=0.4,  # Slightly higher for nuanced responses
                    memory_mode="rich",
                    rerank=True,
                    metadata={"query_type": "complex"},
                ),
            ]
        else:  # Default to Gemini
            strategies = [
                Strategy(
                    arm_id="cheap",
                    model="gemini-flash",
                    max_tokens=300,
                    temperature=0.2,
                    memory_mode="light",
                    metadata={"query_type": "simple"},
                ),
                Strategy(
                    arm_id="balanced",
                    model="gemini-flash",
                    max_tokens=600,
                    temperature=0.3,
                    memory_mode="rich",
                    metadata={"query_type": "medium"},
                ),
                Strategy(
                    arm_id="premium",
                    model="gemini-pro",
                    max_tokens=1000,
                    temperature=0.4,
                    memory_mode="rich",
                    rerank=True,
                    metadata={"query_type": "complex"},
                ),
            ]
        
        self.bandit.add_strategies(strategies)
        logger.info("Initialized default strategies", count=len(strategies))
    
    def query(
        self,
        query: str,
        token_budget: Optional[int] = None,
        use_cache: bool = True,
        use_bandit: bool = True,
        system_prompt: Optional[str] = None,
        use_compression: bool = True,
        use_cost_aware_routing: bool = True,
    ) -> Dict:
        """
        Process a query through the Tokenomics platform.
        
        This is the OPTIMIZED path with full Tokenomics intelligence:
        - Memory layer (exact + semantic cache, context injection)
        - LLM Lingua compression on enriched prompts
        - Bandit-based strategy selection (cheap/balanced/premium)
        - User preference adaptation
        
        Args:
            query: User query
            token_budget: Token budget (uses default if None)
            use_cache: Whether to use memory cache
            use_bandit: Whether to use bandit optimizer
            system_prompt: Optional system prompt
            use_compression: Whether to use LLM-Lingua style compression
            use_cost_aware_routing: Whether to use RouterBench cost-quality routing
        
        Returns:
            Dictionary with response and metadata
        """
        # Input validation
        if not query:
            raise ValueError("Query cannot be empty")
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace only")
        
        logger.info("Processing query", query=query[:100])
        
        # Step 1: Enhanced cache retrieval with tiered similarity matching
        # Tiers: exact match > semantic direct return (>0.92) > context (0.80-0.92) > full LLM
        compressed_context = ""
        preference_context = {}
        cache_hit = False
        cache_type = None
        context_similarity = None  # Track similarity for context matches
        cache_entry = None  # Initialize cache_entry
        capsule_tokens = 0  # Track tokens added from compressed context
        
        # Compression metrics tracking
        context_compressed = False
        context_original_tokens = 0
        context_compressed_tokens = 0
        
        # Initialize detailed memory metrics
        memory_metrics = {
            "exact_cache_hits": 0,
            "semantic_direct_hits": 0,
            "semantic_context_hits": 0,
            "semantic_matches_found": 0,
            "top_similarity": None,
            "context_injected": False,
            "context_tokens_added": 0,
            "context_original_tokens": 0,
            "context_compressed_tokens": 0,
            "context_similarity": None,
            "preferences_used": False,
            "preference_tone": None,
            "preference_format": None,
            "preference_confidence": 0.0,
            "preference_interaction_count": 0,
            "memory_operations": {
                "exact_lookup": False,
                "semantic_search": False,
                "context_retrieval": False,
                "preference_retrieval": False,
                "context_compression": False,
            }
        }
        
        if use_cache:
            # Use enhanced retrieval with compression (returns 5 values now)
            try:
                cache_entry, compressed_context, preference_context, match_similarity, mem_ops = self.memory.retrieve_compressed(
                    query=query,
                    context_token_budget=500,  # Limit context to 500 tokens
                    top_k=3,
                )
            except Exception as e:
                logger.error("Cache retrieval failed, continuing without cache", error=str(e))
                cache_entry = None
                compressed_context = ""
                preference_context = {}
                match_similarity = None
                mem_ops = {
                    "exact_lookup": False,
                    "semantic_search": False,
                    "context_retrieval": False,
                    "preference_retrieval": False,
                    "context_compression": False,
                }
            
            # Update memory metrics from memory operations
            memory_metrics["memory_operations"] = mem_ops
            memory_metrics["semantic_matches_found"] = mem_ops.get("semantic_matches_found", 0)
            memory_metrics["top_similarity"] = mem_ops.get("top_similarity")
            
            # Update preference metrics
            if preference_context:
                memory_metrics["preferences_used"] = True
                memory_metrics["preference_tone"] = preference_context.get("tone")
                memory_metrics["preference_format"] = preference_context.get("format")
                if self.memory.user_preferences:
                    memory_metrics["preference_confidence"] = self.memory.user_preferences.confidence
                    memory_metrics["preference_interaction_count"] = self.memory.user_preferences.interaction_count
            
            # Cache entry returned = exact match OR high-similarity semantic match
            if cache_entry:
                # Determine cache type from similarity metadata
                similarity = cache_entry.metadata.get("similarity") or match_similarity
                cache_type = "semantic_direct" if similarity else "exact"
                
                # Update memory metrics
                if cache_type == "exact":
                    memory_metrics["exact_cache_hits"] = 1
                else:
                    memory_metrics["semantic_direct_hits"] = 1
                
                logger.info(
                    "Cache hit - direct return",
                    cache_type=cache_type,
                    similarity=f"{similarity:.3f}" if similarity else "exact",
                    tokens_saved=cache_entry.tokens_used,
                )
                
                # Even for cache hits, select strategy to show what would have been used
                strategy = None
                # Always calculate complexity for diagnostics
                query_complexity = self.orchestrator.analyze_complexity(query).value
                
                if use_bandit:
                    if use_cost_aware_routing:
                        strategy = self.bandit.select_strategy_cost_aware(query_complexity=query_complexity)
                    else:
                        strategy = self.bandit.select_strategy()
                    if strategy:
                        logger.debug("Selected strategy (cache hit)", arm_id=strategy.arm_id)
                
                plan = self.orchestrator.plan_query(
                    query=query,
                    token_budget=token_budget,
                    retrieved_context=None,
                )
                
                if strategy:
                    plan.model = strategy.model
                
                # Calculate memory savings for cache hit
                memory_savings = cache_entry.tokens_used if cache_entry else 0
                
                # Map cache_type to cache_tier
                cache_tier_map = {
                    "exact": "exact",
                    "semantic_direct": "semantic",
                    "context": "capsule",
                    None: "none"
                }
                cache_tier = cache_tier_map.get(cache_type, "none")
                
                # Format user preference
                user_preference_str = None
                if preference_context:
                    tone = preference_context.get("tone", "")
                    format_pref = preference_context.get("format", "")
                    if tone or format_pref:
                        user_preference_str = f"{tone}-{format_pref}" if tone and format_pref else (tone or format_pref)
                
                return {
                    "response": cache_entry.response,
                    "tokens_used": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_hit": True,
                    "cache_type": cache_type,
                    "similarity": similarity,
                    "latency_ms": 0,
                    "strategy": strategy.arm_id if strategy else None,
                    "model": plan.model or (strategy.model if strategy else self.config.llm.model),
                    "plan": plan,
                    "reward": None,
                    "preference_context": preference_context,
                    "component_savings": {
                        "memory_layer": memory_savings,
                        "orchestrator": 0,
                        "bandit": 0,
                        "total_savings": memory_savings,
                    },
                    # Diagnostic fields
                    "query_type": query_complexity,
                    "cache_tier": cache_tier,
                    "capsule_tokens": 0,  # No context added for direct cache hits
                    "strategy_arm": strategy.arm_id if strategy else None,
                    "model_used": plan.model or (strategy.model if strategy else self.config.llm.model),
                    "used_memory": True,
                    "user_preference": user_preference_str,
                    "memory_metrics": memory_metrics,
                }
            
            # Compressed context means medium-similarity matches (similarity_threshold to direct_return_threshold)
            # CRITICAL: Only use context if it will actually save tokens overall
            # Context adds input tokens, so we need to ensure output savings > input cost
            if compressed_context and not cache_entry:  # Only if we didn't already have a direct cache hit
                context_tokens = self.memory.count_tokens(compressed_context)
                # Track context compression (context is always compressed if retrieved)
                context_compressed = True
                # Note: We don't have original context tokens here since compression happens in retrieve_compressed
                # We'll estimate based on the compressed size and compression ratio
                # Assuming ~2.5x compression (0.4 ratio), original would be ~2.5x compressed
                context_compressed_tokens = context_tokens
                context_original_tokens = int(context_tokens / 0.4) if context_tokens > 0 else 0
                
                # Estimate: context might reduce output by 20-30%, but adds context_tokens to input
                # Only use context if similarity is high enough that it's worth it
                # For now, if similarity > 0.70, use context; otherwise skip to avoid token increase
                if match_similarity and match_similarity >= 0.70:
                    cache_hit = True
                    context_similarity = match_similarity  # Store for return
                    cache_type = "context"  # Mark as context-enhanced cache hit
                    capsule_tokens = context_tokens  # Store capsule tokens for diagnostics
                    
                    # Update memory metrics for context injection
                    memory_metrics["semantic_context_hits"] = 1
                    memory_metrics["context_injected"] = True
                    memory_metrics["context_tokens_added"] = context_tokens
                    memory_metrics["context_compressed_tokens"] = context_compressed_tokens
                    memory_metrics["context_original_tokens"] = context_original_tokens
                    memory_metrics["context_similarity"] = match_similarity
                    
                    logger.info("Semantic context match - using cached context", 
                               context_tokens=context_tokens,
                               similarity=f"{match_similarity:.3f}" if match_similarity else "unknown")
                else:
                    # Similarity too low - context would add more tokens than it saves
                    # Skip context, treat as cache miss
                    compressed_context = ""  # Clear context to avoid using it
                    capsule_tokens = 0  # No capsule tokens added
                    context_compressed = False  # Reset compression tracking
                    logger.info("Semantic match but skipping context - similarity too low for token savings",
                               similarity=f"{match_similarity:.3f}" if match_similarity else "unknown",
                               threshold=0.70)
        
        # Step 2: Calculate complexity using ML classifier if available
        strategy = None
        baseline_result = None
        
        # Get query embedding for complexity prediction
        query_embedding = None
        if self.memory and self.memory.use_semantic_cache and self.memory.embedding_model:
            try:
                query_embedding = self.memory.get_embedding(query)
            except Exception:
                pass
        
        # Calculate complexity using ML classifier or heuristic
        if self.complexity_classifier:
            try:
                complexity_str = self.complexity_classifier.predict(query, query_embedding)
                complexity = complexity_str  # simple/medium/complex
                logger.debug("Complexity predicted (ML)", complexity=complexity)
            except Exception as e:
                logger.warning("ML complexity prediction failed, using heuristic", error=str(e))
                complexity = self.orchestrator.analyze_complexity(query).value
        else:
            complexity = self.orchestrator.analyze_complexity(query).value
        
        # NOTE: Baseline is no longer run here. Use compare_with_baseline() for A/B comparisons.
        
        # Step 3: Create query plan with compressed context (before strategy selection to get context quality)
        # Convert compressed context to list format for orchestrator
        retrieved_context = [compressed_context] if compressed_context else None
        
        try:
            plan = self.orchestrator.plan_query(
                query=query,
                token_budget=token_budget,
                retrieved_context=retrieved_context,
            )
        except Exception as e:
            logger.error("Query planning failed", error=str(e))
            raise RuntimeError(f"Failed to create query plan: {e}") from e
        
        # Step 4: Select strategy with RouterBench cost-quality routing (after plan to use context quality)
        if use_bandit:
            try:
                # Now select strategy for optimized path with context quality awareness
                if use_cost_aware_routing:
                    strategy = self.bandit.select_strategy_cost_aware(
                        query_complexity=complexity,
                        context_quality_score=plan.context_quality_score,
                    )
                else:
                    strategy = self.bandit.select_strategy()
                if strategy:
                    logger.info(
                        "Selected strategy",
                        arm_id=strategy.arm_id,
                        model=strategy.model,
                        query_complexity=complexity,
                        context_quality_score=plan.context_quality_score,
                    )
            except Exception as e:
                logger.error("Strategy selection failed, continuing without bandit", error=str(e))
                strategy = None
        
        # Cascading Inference: Determine initial model
        # If cascading enabled, predict escalation likelihood and decide whether to skip cheap model
        original_strategy_model = strategy.model if strategy else None
        initial_model = None
        should_cascade = False
        escalation_likelihood = 0.0  # Track for outcome tracking later
        
        if self.cascading_enabled and strategy:
            # Predict escalation likelihood using ML predictor if enabled
            if (self.config.cascading.use_escalation_prediction and 
                self.escalation_predictor):
                try:
                    query_token_count = self.orchestrator.count_tokens(query)
                    query_embedding = None
                    if self.memory and self.memory.use_semantic_cache and self.memory.embedding_model:
                        try:
                            query_embedding = self.memory.get_embedding(query)
                        except Exception as e:
                            logger.debug("Failed to get embedding for escalation prediction", error=str(e))
                    
                    escalation_likelihood = self.escalation_predictor.predict(
                        query=query,
                        complexity=complexity,
                        context_quality_score=plan.context_quality_score,
                        query_tokens=query_token_count,
                        query_embedding=query_embedding,
                    )
                except Exception as e:
                    logger.warning("Escalation prediction failed, using normal cascading", error=str(e))
                    escalation_likelihood = 0.0
            
            # Decision threshold
            escalation_threshold = self.config.cascading.escalation_prediction_threshold
            
            # Check if strategy selected premium model
            if strategy.model == self.cascading_premium_model:
                # Check if we should skip cheap model based on prediction
                if (self.config.cascading.use_escalation_prediction and 
                    escalation_likelihood >= escalation_threshold):
                    # High likelihood of escalation - skip cheap, go straight to premium
                    initial_model = self.cascading_premium_model
                    should_cascade = False  # No cascading needed
                    self.cascading_metrics["predicted_escalations"] += 1
                    self.cascading_metrics["skipped_cheap"] += 1
                    logger.info(
                        "Escalation predicted, skipping cheap model",
                        likelihood=escalation_likelihood,
                        threshold=escalation_threshold,
                        model=initial_model,
                    )
                else:
                    # Normal cascading: start with cheap, escalate if needed
                    initial_model = self.cascading_cheap_model
                    should_cascade = True
                    logger.info(
                        "Cascading inference: starting with cheap model",
                        original_strategy=strategy.arm_id,
                        original_model=strategy.model,
                        initial_model=initial_model,
                        escalation_likelihood=escalation_likelihood,
                    )
            else:
                # Strategy already selected cheap model, use it
                initial_model = strategy.model
                should_cascade = True
        elif strategy:
            # No cascading, use strategy's model
            initial_model = strategy.model
        else:
            # No strategy, use default
            initial_model = plan.model or self.config.llm.model
        
        # Override model if strategy specifies (or cascading overrides)
        if initial_model:
            plan.model = initial_model
        
        # Step 5: Build prompt (context already compressed by LLMLingua-2 if enabled)
        try:
            prompt = self.orchestrator.build_prompt(plan, system_prompt=system_prompt)
        except Exception as e:
            logger.error("Failed to build prompt", error=str(e))
            raise RuntimeError(f"Failed to build prompt: {e}") from e
        
        # Step 5b: Compress query if it's long (only the query part, not full prompt)
        original_prompt_tokens = self.orchestrator.count_tokens(prompt)
        query_compressed = False
        query_original_tokens = 0
        query_compressed_tokens = 0
        
        if use_compression:
            try:
                # Compress query if it exceeds thresholds
                original_query = plan.query
                query_original_tokens = self.orchestrator.count_tokens(original_query)
                query_original_chars = len(original_query)
                
                compressed_query = self.memory.compress_query_if_needed(original_query)
                query_compressed_tokens = self.orchestrator.count_tokens(compressed_query)
                
                # Check if compression actually occurred (by token count or string difference)
                if (compressed_query != original_query) or (query_compressed_tokens < query_original_tokens):
                    query_compressed = True
                    
                    # Replace query in prompt
                    prompt = prompt.replace(original_query, compressed_query)
                    
                    logger.info(
                        "Query compressed with LLMLingua-2",
                        original_tokens=query_original_tokens,
                        original_chars=query_original_chars,
                        compressed_tokens=query_compressed_tokens,
                        compressed_chars=len(compressed_query),
                        savings=query_original_tokens - query_compressed_tokens,
                    )
                else:
                    logger.debug(
                        "Query compression skipped or no reduction",
                        query_tokens=query_original_tokens,
                        query_chars=query_original_chars,
                    )
            except Exception as e:
                logger.warning("Query compression failed, using original query", error=str(e))
                query_compressed = False
                query_original_tokens = self.orchestrator.count_tokens(plan.query)
                query_compressed_tokens = query_original_tokens
        
        # Step 6: Predict optimal max_tokens (dynamic prediction instead of hardcoded)
        predicted_max_tokens_value = None  # Track for data collection
        if self.token_predictor:
            try:
                # Get query embedding for prediction
                query_embedding = None
                if self.memory and self.memory.use_semantic_cache and self.memory.embedding_model:
                    try:
                        query_embedding = self.memory.get_embedding(query)
                    except Exception as e:
                        logger.debug("Failed to get embedding for prediction", error=str(e))
                
                # Predict max_tokens (use query token count, not full prompt)
                query_token_count = self.orchestrator.count_tokens(query)
                predicted_max_tokens = self.token_predictor.predict(
                    query=query,
                    complexity=complexity,
                    query_tokens=query_token_count,
                    query_embedding=query_embedding,
                )
                
                # Store predicted value for later recording
                predicted_max_tokens_value = predicted_max_tokens
                
                # Use prediction if available, otherwise fallback to strategy
                if strategy:
                    # Override strategy's hardcoded max_tokens with prediction
                    max_response_tokens = predicted_max_tokens
                    logger.info(
                        "Token prediction applied",
                        predicted_tokens=predicted_max_tokens,
                        strategy_default=strategy.max_tokens,
                        complexity=complexity,
                    )
                else:
                    max_response_tokens = predicted_max_tokens
            except Exception as e:
                logger.warning("Token prediction failed, using strategy default", error=str(e))
                # Fallback to strategy or default
                if strategy:
                    max_response_tokens = strategy.max_tokens
                else:
                    max_response_tokens = plan.token_budget // 2
        else:
            # No predictor available, use strategy default
            if strategy:
                max_response_tokens = strategy.max_tokens  # Strategy controls response length
            else:
                max_response_tokens = plan.token_budget // 2
        
        generation_params = {
            "max_tokens": max_response_tokens,
        }
        
        if strategy:
            generation_params.update({
                "temperature": strategy.temperature,
                "top_p": strategy.top_p,
                "frequency_penalty": strategy.frequency_penalty,
                "presence_penalty": strategy.presence_penalty,
                "n": strategy.n,
            })
        
        # Calculate input tokens after compression
        input_tokens = self.orchestrator.count_tokens(prompt)
        
        # Compute baseline_max_tokens for savings calculations (used later)
        baseline_max_tokens = plan.token_budget // 2
        
        # NOTE: Optimized path uses strategy's max_tokens directly
        # Token control comes from aggressive strategy limits, not from baseline capping
        # This ensures production behavior matches benchmark behavior
        
        # Apply rate limiting before LLM call
        if self.rate_limiter:
            wait_time = self.rate_limiter.wait_if_needed(tokens=1.0)
            if wait_time > 0:
                logger.debug("Rate limiter wait", wait_seconds=wait_time)
        
        # Initialize circuit breaker if not exists
        if not hasattr(self, 'circuit_breaker'):
            try:
                from .resilience.circuit_breaker import CircuitBreaker
                self.circuit_breaker = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60.0,
                )
            except Exception as e:
                logger.warning("Failed to initialize circuit breaker", error=str(e))
                self.circuit_breaker = None
        
        # Cascading Inference: Generate with initial model, escalate if needed
        llm_response = None
        escalated = False
        cascade_quality_score = None
        
        try:
            # Use circuit breaker and retry logic for LLM generation
            def _generate(model_name: str):
                # Temporarily switch provider model for this generation
                original_model = self.llm_provider.model
                self.llm_provider.model = model_name
                
                try:
                    if hasattr(self.llm_provider, 'generate_with_retry'):
                        return self.llm_provider.generate_with_retry(
                            prompt,
                            max_retries=3,
                            initial_backoff=1.0,
                            max_backoff=60.0,
                            **generation_params
                        )
                    else:
                        return self.llm_provider.generate(prompt, **generation_params)
                finally:
                    # Restore original model
                    self.llm_provider.model = original_model
            
            # Generate with initial model (cheap if cascading, otherwise strategy's choice)
            if self.circuit_breaker:
                try:
                    llm_response = self.circuit_breaker.call(lambda: _generate(plan.model))
                except RuntimeError as e:
                    # Circuit is open - try to fallback to cache
                    logger.warning("Circuit breaker open, attempting cache fallback", error=str(e))
                    if use_cache and self.memory:
                        try:
                            cache_entry, _, _, _, _ = self.memory.retrieve_compressed(
                                query=query,
                                context_token_budget=500,
                                top_k=1,
                            )
                            if cache_entry:
                                logger.info("Circuit breaker fallback: using cached response")
                                raise RuntimeError("Circuit breaker open and no suitable cache available") from e
                        except Exception:
                            pass
                    raise
            else:
                llm_response = _generate(plan.model)
            
            # Cascading: Check quality and escalate if needed
            if should_cascade and llm_response:
                # Update metrics
                self.cascading_metrics["total_queries"] += 1
                self.cascading_metrics["cheap_model_used"] += 1
                
                # Quick quality check
                if self.cascading_use_lightweight and self.quality_judge:
                    # Use lightweight quality check
                    cascade_quality_score = self.quality_judge.quick_quality_check(query, llm_response.text)
                else:
                    # Fallback: simple heuristic
                    response_length = len(llm_response.text)
                    query_length = len(query)
                    # Simple heuristic: response should be at least 2x query length
                    if response_length >= query_length * 2:
                        cascade_quality_score = 0.9
                    elif response_length >= query_length:
                        cascade_quality_score = 0.7
                    else:
                        cascade_quality_score = 0.5
                
                logger.info(
                    "Cascading quality check",
                    quality_score=cascade_quality_score,
                    threshold=self.cascading_quality_threshold,
                    model=plan.model,
                )
                
                # Escalate if quality insufficient
                if cascade_quality_score < self.cascading_quality_threshold:
                    logger.warning(
                        "Cascading escalation triggered",
                        quality_score=cascade_quality_score,
                        threshold=self.cascading_quality_threshold,
                        escalating_to=self.cascading_premium_model,
                    )
                    
                    # Escalate to premium model
                    plan.model = self.cascading_premium_model
                    self.cascading_metrics["escalations"] += 1
                    escalated = True
                    
                    # Track if we predicted this correctly
                    escalation_threshold = self.config.cascading.escalation_prediction_threshold
                    if (self.config.cascading.use_escalation_prediction and 
                        escalation_likelihood >= escalation_threshold):
                        # We predicted escalation - correct prediction (already counted)
                        pass
                    else:
                        # We didn't predict but needed escalation - false negative
                        self.cascading_metrics["false_negatives"] += 1
                    
                    # Generate with premium model
                    try:
                        if self.circuit_breaker:
                            premium_response = self.circuit_breaker.call(lambda: _generate(self.cascading_premium_model))
                        else:
                            premium_response = _generate(self.cascading_premium_model)
                        
                        # Use premium response (it should be better)
                        llm_response = premium_response
                        logger.info("Cascading escalation: using premium model response")
                    except Exception as e:
                        logger.error("Cascading escalation failed, using cheap model response", error=str(e))
                        # Fallback to cheap model response
                        escalated = False
                else:
                    # No escalation needed
                    escalation_threshold = self.config.cascading.escalation_prediction_threshold
                    if (self.config.cascading.use_escalation_prediction and 
                        escalation_likelihood >= escalation_threshold):
                        # We predicted escalation but didn't need it - false positive
                        self.cascading_metrics["false_positives"] += 1
                
                # Calculate prediction accuracy
                if self.config.cascading.use_escalation_prediction:
                    total_predictions = (
                        self.cascading_metrics["predicted_escalations"] + 
                        self.cascading_metrics["false_negatives"]
                    )
                    if total_predictions > 0:
                        correct_predictions = (
                            self.cascading_metrics["predicted_escalations"] - 
                            self.cascading_metrics["false_positives"]
                        )
                        self.cascading_metrics["prediction_accuracy"] = (
                            correct_predictions / total_predictions * 100
                        )
                
                # Record escalation outcome for ML learning
                if (self.config.cascading.use_escalation_prediction and 
                    self.escalation_predictor and should_cascade):
                    try:
                        escalated_actual = cascade_quality_score < self.cascading_quality_threshold
                        query_token_count = self.orchestrator.count_tokens(query)
                        query_embedding = None
                        if self.memory and self.memory.use_semantic_cache and self.memory.embedding_model:
                            try:
                                query_embedding = self.memory.get_embedding(query)
                            except Exception:
                                pass
                        
                        self.escalation_predictor.record_outcome(
                            query=query,
                            complexity=complexity,
                            context_quality_score=plan.context_quality_score,
                            query_tokens=query_token_count,
                            query_embedding=query_embedding,
                            escalated=escalated_actual,
                            model_used=plan.model if plan else None,
                        )
                    except Exception as e:
                        logger.warning("Failed to record escalation outcome", error=str(e))
                
                # Record complexity prediction for training
                if self.complexity_classifier:
                    try:
                        # Use orchestrator's heuristic as ground truth (for now)
                        # Later, can use human feedback or quality metrics
                        actual_complexity = self.orchestrator._analyze_complexity_heuristic(query).value
                        
                        self.complexity_classifier.record_prediction(
                            query=query,
                            predicted_complexity=complexity,
                            actual_complexity=actual_complexity,  # Ground truth
                            query_embedding=query_embedding,
                            model_used=plan.model if plan else None,
                        )
                    except Exception as e:
                        logger.warning("Failed to record complexity prediction", error=str(e))
                else:
                    logger.info(
                        "Cascading: quality sufficient, using cheap model",
                        quality_score=cascade_quality_score,
                    )
            
        except Exception as e:
            logger.error("LLM generation failed after retries", error=str(e), model=plan.model)
            raise RuntimeError(f"LLM generation failed: {e}") from e
        
        # Update cascading metrics
        if should_cascade:
            self.cascading_metrics["escalation_rate"] = (
                self.cascading_metrics["escalations"] / self.cascading_metrics["total_queries"] * 100
                if self.cascading_metrics["total_queries"] > 0 else 0.0
            )
        
        # Step 7: Rerank if needed (skip if cascading escalated, already have best response)
        final_response = llm_response.text
        if strategy and strategy.rerank and strategy.n > 1 and not escalated:
            responses = self.llm_provider.generate_multiple(prompt, n=strategy.n)
            final_response = max(responses, key=lambda r: len(r.text)).text
        
        # Extract token breakdown from response if available
        # OpenAI returns usage.prompt_tokens and usage.completion_tokens
        if llm_response.metadata:
            # Use actual breakdown from API response
            input_tokens = llm_response.metadata.get("prompt_tokens", input_tokens)
            output_tokens = llm_response.metadata.get("completion_tokens", 0)
        else:
            # Fallback: estimate output tokens
            output_tokens = llm_response.tokens_used - input_tokens if llm_response.tokens_used >= input_tokens else 0
        
        # Initialize final_similarity if not already set (for cache storage)
        if 'final_similarity' not in locals():
            final_similarity = context_similarity if 'context_similarity' in locals() else None
            if cache_entry and not final_similarity:
                final_similarity = cache_entry.metadata.get("similarity") if cache_entry else None
        
        # Step 8: Store in cache
        if use_cache:
            try:
                self.memory.store(
                    query=query,
                    response=final_response,
                    tokens_used=llm_response.tokens_used,
                    metadata={
                        "similarity": final_similarity if 'final_similarity' in locals() else None,
                        "cache_type": final_cache_type if 'final_cache_type' in locals() else None,
                    },
                )
            except Exception as e:
                logger.warning("Failed to store in cache, continuing", error=str(e))
        
        # Step 9: Update bandit with RouterBench-style reward
        reward = None
        if use_bandit and strategy:
            quality_score = 1.0  # Placeholder - would use actual quality scorer
            
            if use_cost_aware_routing:
                # RouterBench reward: considers cost, quality, latency
                reward = self.bandit.compute_reward_routerbench(
                    arm_id=strategy.arm_id,
                    quality_score=quality_score,
                    tokens_used=llm_response.tokens_used,
                    latency_ms=llm_response.latency_ms,
                    model=plan.model or strategy.model,
                )
            else:
                reward = self.bandit.compute_reward(
                    quality_score=quality_score,
                    tokens_used=llm_response.tokens_used,
                    latency_ms=llm_response.latency_ms,
                )
            
            self.bandit.update(strategy.arm_id, reward)
        
        # Calculate component-level savings
        # Memory layer savings: tokens saved from cache hits (0 tokens used = full savings)
        memory_savings = cache_entry.tokens_used if cache_entry else 0
        
        # For context-enhanced cache hits, calculate partial savings
        # CRITICAL: Context adds input tokens, so we need to calculate net savings
        if cache_hit and cache_type == "context":
            # Context-enhanced: we used cached context which added input tokens
            # But it should reduce output tokens
            # Net savings = (output_tokens_saved) - (context_input_tokens_added)
            # We need to estimate what the context input tokens were
            # Since compressed_context was used, we can estimate from the plan
            context_input_estimate = 0
            if plan and plan.retrieved_context:
                context_input_estimate = sum(self.orchestrator.count_tokens(ctx) for ctx in plan.retrieved_context)
            
            # Estimate baseline output (what we'd generate without context)
            estimated_baseline_output = baseline_max_tokens
            # Actual output with context
            actual_output = output_tokens
            # Output savings from having context
            output_savings = max(0, estimated_baseline_output - actual_output)
            # Net savings = output savings - context input cost
            memory_savings = max(0, output_savings - context_input_estimate)
            
            # If context actually increased tokens, don't count it as savings
            if memory_savings < 0:
                logger.warning("Context-enhanced cache increased tokens - not counting as savings",
                             context_input=context_input_estimate,
                             output_savings=output_savings,
                             net_savings=memory_savings)
                memory_savings = 0
        
        # Orchestrator savings: tokens saved from better allocation/compression
        # Compare actual output tokens with baseline expectation
        baseline_expected_output = baseline_max_tokens
        orchestrator_savings = max(0, baseline_expected_output - output_tokens) if not cache_hit or cache_type == "context" else 0
        
        # Bandit savings: tokens saved from selecting optimal strategy
        # Compare strategy's max_tokens limit with baseline
        # If strategy limits output more aggressively, that's a savings
        bandit_savings = 0
        if strategy and not cache_hit:
            # Strategy might limit output tokens, but we already capped it to baseline
            # So bandit savings come from selecting a more efficient strategy
            # This is harder to quantify, so we'll track it separately
            if max_response_tokens < baseline_max_tokens:
                bandit_savings = baseline_max_tokens - max_response_tokens
        
        logger.info(
            "Query completed",
            tokens_used=llm_response.tokens_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=llm_response.latency_ms,
            cache_hit=cache_hit,
            strategy=strategy.arm_id if strategy else None,
            max_tokens=max_response_tokens,
            memory_savings=memory_savings,
            orchestrator_savings=orchestrator_savings,
            bandit_savings=bandit_savings,
        )
        
        # Initialize final_similarity early (before cache storage)
        final_similarity = context_similarity
        if cache_entry and not final_similarity:
            final_similarity = cache_entry.metadata.get("similarity")
        
        # Determine final cache type and similarity
        final_cache_type = cache_type if cache_hit else None
        
        # Use already calculated complexity for diagnostics (calculated earlier in the function)
        query_complexity = complexity
        
        # Map cache_type to cache_tier
        cache_tier_map = {
            "exact": "exact",
            "semantic_direct": "semantic",
            "context": "capsule",
            None: "none"
        }
        cache_tier = cache_tier_map.get(final_cache_type, "none")
        
        # Format user preference
        user_preference_str = None
        if preference_context:
            tone = preference_context.get("tone", "")
            format_pref = preference_context.get("format", "")
            if tone or format_pref:
                user_preference_str = f"{tone}-{format_pref}" if tone and format_pref else (tone or format_pref)
        
        # Calculate compression savings
        context_compression_savings = max(0, context_original_tokens - context_compressed_tokens) if context_compressed else 0
        query_compression_savings = max(0, query_original_tokens - query_compressed_tokens) if query_compressed else 0
        total_compression_savings = context_compression_savings + query_compression_savings
        
        # Add compression savings to orchestrator savings (compression is part of orchestration)
        orchestrator_savings += total_compression_savings
        
        # Build orchestrator metrics
        orchestrator_metrics = {
            "token_budget": plan.token_budget,
            "complexity": query_complexity,
            "model": plan.model or (strategy.model if strategy else self.config.llm.model),
            "allocations": [
                {
                    "component": alloc.component,
                    "tokens": alloc.tokens,
                }
                for alloc in (plan.allocations or [])
            ],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_response_tokens": max_response_tokens,
            "token_efficiency": (output_tokens / input_tokens) if input_tokens > 0 else 0,
        }
        
        # Build bandit metrics
        bandit_metrics = {}
        if strategy:
            bandit_metrics = {
                "strategy": strategy.arm_id,
                "model": strategy.model,
                "max_tokens": strategy.max_tokens,
                "temperature": strategy.temperature,
                "reward": reward,
                "use_cost_aware_routing": use_cost_aware_routing,
            }
            
            # Get RouterBench metrics if available
            if use_cost_aware_routing and strategy.arm_id in self.bandit.arms:
                arm = self.bandit.arms[strategy.arm_id]
                routing_metrics = arm.routing_metrics
                bandit_metrics["routerbench"] = {
                    "total_cost": routing_metrics.total_cost,
                    "total_tokens": routing_metrics.total_tokens,
                    "total_latency_ms": routing_metrics.total_latency_ms,
                    "total_quality": routing_metrics.total_quality,
                    "query_count": routing_metrics.query_count,
                    "avg_cost_per_query": routing_metrics.avg_cost_per_query,
                    "avg_tokens": routing_metrics.avg_tokens,
                    "avg_latency": routing_metrics.avg_latency,
                    "avg_quality": routing_metrics.avg_quality,
                    "cost_quality_ratio": routing_metrics.cost_quality_ratio,
                    "efficiency_score": routing_metrics.efficiency_score,
                }
        
        optimized_result = {
            "response": final_response,
            "tokens_used": llm_response.tokens_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": cache_hit,
            "cache_type": final_cache_type,
            "similarity": final_similarity if 'final_similarity' in locals() else (context_similarity if 'context_similarity' in locals() else None),  # Include similarity for all cache types
            "latency_ms": llm_response.latency_ms,
            "strategy": strategy.arm_id if strategy else None,
            "model": plan.model or (strategy.model if strategy else self.config.llm.model),
            "reward": reward,
            "plan": plan,
            "preference_context": preference_context,
            "max_response_tokens": max_response_tokens,
            "predicted_max_tokens": predicted_max_tokens_value if predicted_max_tokens_value is not None else max_response_tokens,
            # Compression metrics
            "compression_metrics": {
                "context_compressed": context_compressed,
                "context_original_tokens": context_original_tokens,
                "context_compressed_tokens": context_compressed_tokens,
                "context_compression_ratio": context_compressed_tokens / context_original_tokens if context_original_tokens > 0 else 1.0,
                "query_compressed": query_compressed,
                "query_original_tokens": query_original_tokens,
                "query_compressed_tokens": query_compressed_tokens,
                "query_compression_ratio": query_compressed_tokens / query_original_tokens if query_original_tokens > 0 else 1.0,
                "total_compression_savings": total_compression_savings,
            },
            # Component-level savings
            "component_savings": {
                "memory_layer": memory_savings,
                "orchestrator": orchestrator_savings,
                "bandit": bandit_savings,
                "total_savings": memory_savings + orchestrator_savings + bandit_savings,
            },
            # Enhanced metrics
            "orchestrator_metrics": orchestrator_metrics,
            "bandit_metrics": bandit_metrics,
            # Diagnostic fields
            "query_type": query_complexity,
            "cache_tier": cache_tier,
            "capsule_tokens": capsule_tokens,
            "strategy_arm": strategy.arm_id if strategy else None,
            "model_used": plan.model or (strategy.model if strategy else self.config.llm.model),
            "used_memory": cache_hit,
            "user_preference": user_preference_str,
            "memory_metrics": memory_metrics,
        }
        
        # Component savings are calculated in the regular flow (lines 616-678)
        # Baseline comparison and quality judging are now handled in compare_with_baseline()
        
        # Record metrics
        if self.metrics_collector:
            try:
                self.metrics_collector.record_query(
                    success=True,
                    latency_ms=optimized_result.get("latency_ms", 0),
                    tokens_used=optimized_result.get("tokens_used", 0),
                    input_tokens=optimized_result.get("input_tokens", 0),
                    output_tokens=optimized_result.get("output_tokens", 0),
                    cache_hit=optimized_result.get("cache_hit", False),
                )
            except Exception as e:
                logger.warning("Failed to record metrics", error=str(e))
        
        # Record token prediction data for ML training
        if self.token_predictor and not cache_hit:
            try:
                # Get query embedding
                query_embedding = None
                if self.memory and self.memory.use_semantic_cache and self.memory.embedding_model:
                    try:
                        query_embedding = self.memory.get_embedding(query)
                    except Exception:
                        pass
                
                # Record prediction and actual result
                # Get predicted tokens from the result (stored during prediction)
                predicted_tokens = optimized_result.get("predicted_max_tokens", 0)
                # Use query token count (not full prompt tokens) for recording
                query_token_count = self.orchestrator.count_tokens(query)
                self.token_predictor.record_prediction(
                    query=query,
                    complexity=complexity,
                    query_tokens=query_token_count,
                    query_embedding=query_embedding,
                    predicted_tokens=predicted_tokens,
                    actual_output_tokens=output_tokens,
                    model_used=plan.model if plan else None,
                )
            except Exception as e:
                logger.warning("Failed to record token prediction data", error=str(e))
        
        return optimized_result
    
    def _run_baseline_query(
        self,
        query: str,
        token_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Run a baseline query - "what teams do today" (vanilla LLM integration).
        
        This is intentionally simple:
        - Single model (configured default)
        - Prompt = system + raw user query
        - No memory, no semantic cache
        - No LLM Lingua compression
        - No bandit strategy, no context injection
        - Reasonable max_tokens (512) and standard temperature (0.3)
        
        Args:
            query: User query
            token_budget: Token budget (uses default if None)
            system_prompt: Optional system prompt
        
        Returns:
            Baseline result dict
        """
        # Simple prompt - just system + raw query (what teams do today)
        # NO orchestrator, NO compression, NO context injection
        prompt = query
        if system_prompt:
            prompt = f"{system_prompt}\n\n{query}"
        
        # Calculate input tokens
        input_tokens = self.orchestrator.count_tokens(prompt)
        
        # Baseline settings: reasonable defaults for production support use case
        generation_params = {
            "max_tokens": 512,  # Reasonable limit for support responses
            "temperature": 0.3,  # Standard for support (consistent but not robotic)
        }
        
        # Use existing provider (already configured with default model)
        try:
            llm_response = self.llm_provider.generate(prompt, **generation_params)
        except Exception as e:
            logger.error("Baseline LLM generation failed", error=str(e))
            raise RuntimeError(f"Baseline query failed: {e}") from e
        
        # Extract token breakdown
        if llm_response.metadata:
            input_tokens = llm_response.metadata.get("prompt_tokens", input_tokens)
            output_tokens = llm_response.metadata.get("completion_tokens", 0)
        else:
            # Fallback: use tokens_used if available, otherwise count manually
            if llm_response.tokens_used and llm_response.tokens_used >= input_tokens:
                output_tokens = llm_response.tokens_used - input_tokens
            else:
                # Last resort: count tokens manually from response text
                output_tokens = self.orchestrator.count_tokens(llm_response.text)
                logger.debug(
                    "Baseline: counted output tokens manually",
                    counted_tokens=output_tokens,
                    response_length=len(llm_response.text),
                )
        
        # Ensure we have valid token counts
        if input_tokens == 0:
            input_tokens = self.orchestrator.count_tokens(prompt)
            logger.debug("Baseline: counted input tokens manually", counted_tokens=input_tokens)
        
        if output_tokens == 0 and llm_response.text:
            output_tokens = self.orchestrator.count_tokens(llm_response.text)
            logger.debug("Baseline: counted output tokens manually (fallback)", counted_tokens=output_tokens)
        
        # Calculate total tokens
        tokens_used = input_tokens + output_tokens
        if llm_response.tokens_used and llm_response.tokens_used > 0:
            # Prefer API-reported tokens if available
            tokens_used = llm_response.tokens_used
        
        # Verify max_tokens was respected - if not, truncate response
        max_tokens_limit = generation_params.get("max_tokens", 512)
        if output_tokens > max_tokens_limit:
            logger.warning(
                "Baseline generated more tokens than max_tokens limit",
                max_tokens=max_tokens_limit,
                actual_output_tokens=output_tokens,
                response_length=len(llm_response.text),
            )
            # Truncate response to approximate the token limit
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens_limit * 4
            if len(llm_response.text) > max_chars:
                llm_response.text = llm_response.text[:max_chars] + "..."
                # Recalculate output_tokens after truncation
                output_tokens = min(output_tokens, max_tokens_limit)
                logger.info(
                    "Truncated baseline response to respect max_tokens",
                    truncated_to=max_tokens_limit,
                )
        
        # Ensure tokens_used is set correctly (use calculated value)
        if not tokens_used or tokens_used == 0:
            tokens_used = input_tokens + output_tokens
        
        logger.debug(
            "Baseline query completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_used=tokens_used,
            response_length=len(llm_response.text),
            api_tokens_used=getattr(llm_response, 'tokens_used', None),
        )
        
        return {
            "response": llm_response.text,
            "tokens_used": tokens_used,  # Use calculated value, not API value (which might be 0)
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": False,
            "cache_type": "none",
            "similarity": None,
            "latency_ms": llm_response.latency_ms if hasattr(llm_response, 'latency_ms') else 0,
            "strategy": None,
            "model": self.config.llm.model,  # Default model used by baseline
            "reward": None,
            "plan": None,
            "preference_context": None,
            "max_response_tokens": 512,  # Baseline max_tokens
            "component_savings": {
                "memory_layer": 0,
                "orchestrator": 0,
                "bandit": 0,
                "total_savings": 0,
            },
        }
    
    def compare_with_baseline(
        self,
        query: str,
        token_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Run both baseline and optimized paths for A/B comparison.
        
        This method explicitly runs both paths when A/B comparison is needed.
        The optimized path uses full Tokenomics intelligence.
        
        Args:
            query: User query
            token_budget: Token budget (uses default if None)
            system_prompt: Optional system prompt
        
        Returns:
            Dictionary with both baseline and optimized results, plus comparison metrics
        """
        logger.info("Running A/B comparison", query=query[:100])
        
        # Run baseline first
        baseline_result = self._run_baseline_query(
            query=query,
            token_budget=token_budget,
            system_prompt=system_prompt,
        )
        
        # Run optimized path with baseline comparison enabled
        optimized_result = self.query(
            query=query,
            token_budget=token_budget,
            use_cache=True,
            use_bandit=True,
            system_prompt=system_prompt,
            use_compression=True,
            use_cost_aware_routing=True,
        )
        
        # Recalculate component savings with actual baseline comparison
        baseline_tokens = baseline_result["tokens_used"]
        baseline_output = baseline_result["output_tokens"]
        baseline_max_tokens = baseline_result.get("max_response_tokens", 512)
        
        # Get strategy information
        strategy = None
        if optimized_result.get("strategy_arm"):
            # Find the strategy that was used
            for arm_id, arm in self.bandit.arms.items():
                if arm_id == optimized_result["strategy_arm"]:
                    strategy = arm.strategy
                    break
        
        strategy_max_tokens = strategy.max_tokens if strategy else baseline_max_tokens
        optimized_output = optimized_result["output_tokens"]
        
        # Calculate component savings
        if optimized_result.get("cache_hit"):
            # Cache hit: memory layer saved all baseline tokens
            memory_savings = baseline_tokens
            orchestrator_savings = 0
            bandit_savings = 0
        else:
            # No cache hit: calculate actual savings vs baseline
            memory_savings = 0
            
            # Bandit savings: tokens saved from strategy selection
            bandit_savings = max(0, baseline_max_tokens - strategy_max_tokens)
            
            # Actual output reduction from strategy limiting
            actual_output_reduction = max(0, baseline_output - optimized_output)
            bandit_output_savings = min(bandit_savings, actual_output_reduction)
            
            # Orchestrator savings: compression savings + allocation efficiency
            compression_metrics = optimized_result.get("compression_metrics", {})
            compression_savings = compression_metrics.get("total_compression_savings", 0)
            allocation_savings = max(0, actual_output_reduction - bandit_output_savings)
            
            orchestrator_savings = compression_savings + allocation_savings
            bandit_savings = bandit_output_savings
        
        # Update component savings in optimized result
        optimized_result["component_savings"] = {
            "memory_layer": memory_savings,
            "orchestrator": orchestrator_savings,
            "bandit": bandit_savings,
            "total_savings": memory_savings + orchestrator_savings + bandit_savings,
        }
        
        # Store baseline comparison data
        optimized_result["baseline_comparison_result"] = {
            "tokens_used": baseline_result["tokens_used"],
            "input_tokens": baseline_result["input_tokens"],
            "output_tokens": baseline_result["output_tokens"],
            "latency_ms": baseline_result["latency_ms"],
            "model": baseline_result["model"],
            "response": baseline_result["response"],
            "max_response_tokens": baseline_max_tokens,
        }
        
        # If judge is enabled, run quality evaluation
        quality_threshold = 0.85  # Minimum quality score to accept optimized result
        judge_result = None
        
        if self.quality_judge:
            judge_result = self.quality_judge.judge(
                query=query,
                baseline_answer=baseline_result["response"],
                optimized_answer=optimized_result["response"],
            )
            
            if judge_result:
                # Convert judge result to quality score
                if judge_result.winner == "optimized":
                    quality_score = 1.0
                elif judge_result.winner == "equivalent":
                    quality_score = 0.9
                else:  # baseline better
                    quality_score = 0.7
                
                optimized_result["quality_judge"] = {
                    "winner": judge_result.winner,
                    "explanation": judge_result.explanation,
                    "confidence": judge_result.confidence,
                    "quality_score": quality_score,
                }
                
                logger.info(
                    "Quality judged",
                    winner=judge_result.winner,
                    confidence=judge_result.confidence,
                    quality_score=quality_score,
                )
                
                # Quality guarantee: fallback to baseline if quality is too low
                if quality_score < quality_threshold:
                    logger.warning(
                        "Quality below threshold, falling back to baseline",
                        quality_score=quality_score,
                        threshold=quality_threshold,
                    )
                    # Return baseline result with quality warning
                    optimized_result["quality_fallback"] = True
                    optimized_result["quality_warning"] = f"Quality score {quality_score:.2f} below threshold {quality_threshold}, consider using baseline"
                    # Don't update bandit with poor quality result
                else:
                    # Update bandit reward with judge quality if available
                    if strategy:
                        judge_quality_score = quality_score
                        
                        # Update reward with judge quality
                        if hasattr(self.bandit, 'compute_reward_routerbench'):
                            reward = self.bandit.compute_reward_routerbench(
                                arm_id=strategy.arm_id,
                                quality_score=judge_quality_score,
                                tokens_used=optimized_result["tokens_used"],
                                latency_ms=optimized_result["latency_ms"],
                                model=optimized_result["model"],
                            )
                        else:
                            reward = self.bandit.compute_reward(
                                quality_score=judge_quality_score,
                                tokens_used=optimized_result["tokens_used"],
                                latency_ms=optimized_result["latency_ms"],
                            )
                        
                        optimized_result["reward"] = reward
                        self.bandit.update(strategy.arm_id, reward)
        
        return optimized_result
    
    def calculate_real_savings(
        self,
        baseline_result: Dict,
        optimized_result: Dict,
    ) -> Dict[str, Any]:
        """
        Calculate real cost savings ($) with component breakdown.
        
        This provides actual dollar savings, not just token savings,
        with clear attribution to each component.
        
        Args:
            baseline_result: Baseline query result
            optimized_result: Optimized query result
        
        Returns:
            Dictionary with cost savings breakdown
        """
        # Model pricing per 1M tokens (input/output separated)
        MODEL_PRICING = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gemini-flash": {"input": 0.075, "output": 0.30},
            "gemini-pro": {"input": 1.25, "output": 5.00},
        }
        
        # Get model pricing
        baseline_model = baseline_result.get("model", "gpt-4o")
        optimized_model = optimized_result.get("model", "gpt-4o-mini")
        
        # Get token counts
        baseline_tokens = baseline_result.get("tokens_used", 0)
        baseline_input = baseline_result.get("input_tokens", 0)
        baseline_output = baseline_result.get("output_tokens", 0)
        
        optimized_tokens = optimized_result.get("tokens_used", 0)
        optimized_input = optimized_result.get("input_tokens", 0)
        optimized_output = optimized_result.get("output_tokens", 0)
        
        # Get model costs per 1M tokens (input/output separated)
        baseline_pricing = MODEL_PRICING.get(baseline_model, {"input": 2.50, "output": 10.00})
        optimized_pricing = MODEL_PRICING.get(optimized_model, {"input": 0.15, "output": 0.60})
        
        # Calculate actual costs with input/output separation
        baseline_cost = (
            (baseline_input / 1_000_000) * baseline_pricing["input"] +
            (baseline_output / 1_000_000) * baseline_pricing["output"]
        )
        
        optimized_cost = (
            (optimized_input / 1_000_000) * optimized_pricing["input"] +
            (optimized_output / 1_000_000) * optimized_pricing["output"]
        )
        
        total_cost_savings = baseline_cost - optimized_cost
        
        # Component breakdown
        cache_hit = optimized_result.get("cache_hit", False)
        
        # Memory savings: cost saved from cache hits
        if cache_hit:
            memory_savings_cost = baseline_cost  # All baseline cost saved
        else:
            memory_savings_cost = 0
        
        # Bandit savings: cost difference from model selection + max_tokens limits
        baseline_max_tokens = baseline_result.get("max_response_tokens", 512)
        strategy = None
        if optimized_result.get("strategy_arm"):
            for arm_id, arm in self.bandit.arms.items():
                if arm_id == optimized_result["strategy_arm"]:
                    strategy = arm.strategy
                    break
        
        strategy_max_tokens = strategy.max_tokens if strategy else baseline_max_tokens
        
        # Model cost difference (input + output pricing)
        baseline_input_cost_per_token = baseline_pricing["input"] / 1_000_000
        baseline_output_cost_per_token = baseline_pricing["output"] / 1_000_000
        optimized_input_cost_per_token = optimized_pricing["input"] / 1_000_000
        optimized_output_cost_per_token = optimized_pricing["output"] / 1_000_000
        
        # Cost if we used baseline model for same tokens
        baseline_model_cost_for_optimized = (
            optimized_input * baseline_input_cost_per_token +
            optimized_output * baseline_output_cost_per_token
        )
        
        # Actual optimized cost
        actual_optimized_cost = (
            optimized_input * optimized_input_cost_per_token +
            optimized_output * optimized_output_cost_per_token
        )
        
        # Model switch savings
        model_cost_savings = baseline_model_cost_for_optimized - actual_optimized_cost
        
        # Max tokens savings (if strategy limits output)
        max_tokens_savings = 0
        if strategy_max_tokens < baseline_max_tokens:
            # Calculate cost of tokens that would have been generated
            tokens_saved = min(baseline_max_tokens - strategy_max_tokens, baseline_output - optimized_output)
            max_tokens_savings = tokens_saved * optimized_output_cost_per_token
        
        bandit_savings_cost = model_cost_savings + max_tokens_savings
        
        # Orchestrator savings: compression savings + allocation efficiency
        compression_metrics = optimized_result.get("compression_metrics", {})
        compression_savings_tokens = compression_metrics.get("total_compression_savings", 0)
        # Compression saves input tokens (context/query compression)
        compression_savings_cost = compression_savings_tokens * optimized_input_cost_per_token
        
        # Allocation efficiency: better token utilization
        # This is harder to quantify, so we attribute remaining savings to orchestrator
        allocation_savings_cost = max(0, total_cost_savings - memory_savings_cost - bandit_savings_cost - compression_savings_cost)
        
        orchestrator_savings_cost = compression_savings_cost + allocation_savings_cost
        
        return {
            "baseline_cost": round(baseline_cost, 6),
            "optimized_cost": round(optimized_cost, 6),
            "total_savings": round(total_cost_savings, 6),
            "savings_percent": round((total_cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0, 2),
            "component_breakdown": {
                "memory_layer": round(memory_savings_cost, 6),
                "bandit": round(bandit_savings_cost, 6),
                "orchestrator": round(orchestrator_savings_cost, 6),
                "total": round(memory_savings_cost + bandit_savings_cost + orchestrator_savings_cost, 6),
            },
            "model_info": {
                "baseline_model": baseline_model,
                "optimized_model": optimized_model,
                "model_switch_savings": round(model_cost_savings, 6),
            },
        }
    
    def get_stats(self) -> Dict:
        """Get platform statistics."""
        stats = {
            "memory": self.memory.stats(),
            "bandit": self.bandit.stats(),
            "orchestrator": {
                "default_budget": self.orchestrator.default_token_budget,
            },
            "cascading": self.cascading_metrics.copy() if hasattr(self, 'cascading_metrics') else {},
            "token_predictor": self.token_predictor.get_stats() if self.token_predictor else {},
            "escalation_predictor": self.escalation_predictor.get_stats() if self.escalation_predictor else {},
            "complexity_classifier": self.complexity_classifier.get_stats() if self.complexity_classifier else {},
        }
        
        # Update state manager if available
        if self.state_manager:
            try:
                self.state_manager.update_bandit_state(self.bandit.stats())
                self.state_manager.update_cache_state(self.memory.stats())
            except Exception as e:
                logger.warning("Failed to update state manager", error=str(e))
        
        return stats
    
    def save_state(self):
        """Save platform state."""
        if self.state_manager:
            try:
                # Save bandit state
                self.state_manager.update_bandit_state(self.bandit.stats())
                # Save cache state
                self.state_manager.update_cache_state(self.memory.stats())
                logger.info("Platform state saved")
            except Exception as e:
                logger.warning("Failed to save platform state", error=str(e))
    
    def load_state(self):
        """Load platform state."""
        if self.state_manager:
            try:
                state = self.state_manager.get_state()
                # Restore bandit state if available
                if state.get("bandit") and hasattr(self.bandit, 'load_state'):
                    # Bandit has its own state loading mechanism
                    pass
                logger.info("Platform state loaded")
            except Exception as e:
                logger.warning("Failed to load platform state", error=str(e))
    
    def train_token_predictor(self, min_samples: int = 500) -> bool:
        """
        Train token prediction ML model when enough data is available.
        
        Args:
            min_samples: Minimum number of samples required
        
        Returns:
            True if model was trained, False otherwise
        """
        if not self.token_predictor:
            logger.warning("Token predictor not available")
            return False
        
        return self.token_predictor.train_model(min_samples=min_samples)
    
    def train_escalation_predictor(self, min_samples: int = 100) -> bool:
        """
        Train escalation prediction ML model when enough data is available.
        
        Args:
            min_samples: Minimum number of samples required (default: 100)
        
        Returns:
            True if model was trained, False otherwise
        """
        if not self.escalation_predictor:
            logger.warning("Escalation predictor not available")
            return False
        
        return self.escalation_predictor.train_model(min_samples=min_samples)
    
    def train_complexity_classifier(self, min_samples: int = 100) -> bool:
        """
        Train complexity classification ML model when enough data is available.
        
        Args:
            min_samples: Minimum number of samples required (default: 100)
        
        Returns:
            True if model was trained, False otherwise
        """
        if not self.complexity_classifier:
            logger.warning("Complexity classifier not available")
            return False
        
        return self.complexity_classifier.train_model(min_samples=min_samples)

