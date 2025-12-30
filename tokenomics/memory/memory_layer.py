"""Complete memory layer combining exact cache, semantic search, Mem0 preferences, and LLM-Lingua compression."""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import structlog

from .cache import MemoryCache, CacheEntry
from .vector_store import VectorStore, FAISSVectorStore, ChromaVectorStore

logger = structlog.get_logger()


@dataclass
class UserPreferences:
    """Mem0-style user preferences learned from interactions."""
    tone: str = "neutral"  # formal, casual, technical, simple
    format: str = "paragraph"  # list, paragraph, code, concise
    detail_level: str = "medium"  # brief, medium, detailed
    confidence: float = 0.5
    interaction_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, detected_tone: str, detected_format: str):
        """Update preferences with new detection."""
        self.interaction_count += 1
        alpha = min(0.3, 1.0 / self.interaction_count)  # Learning rate
        
        if detected_tone != "neutral":
            if self.tone == detected_tone:
                self.confidence = min(1.0, self.confidence + 0.1)
            else:
                self.confidence = max(0.3, self.confidence - 0.05)
                if self.confidence < 0.4:
                    self.tone = detected_tone
                    self.confidence = 0.5
        
        if detected_format != "paragraph":
            self.format = detected_format
        
        self.last_updated = datetime.now()


class SmartMemoryLayer:
    """
    Smart memory layer with exact and semantic caching.
    
    Enhanced with:
    - Mem0-style user preference learning
    - LLM-Lingua style context compression
    """
    
    def __init__(
        self,
        use_exact_cache: bool = True,
        use_semantic_cache: bool = True,
        vector_store_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        direct_return_threshold: float = 0.85,
        cache_size: int = 1000,
        eviction_policy: str = "lru",
        ttl_seconds: Optional[int] = None,
        persistent_storage: Optional[str] = None,
        enable_compression: bool = True,
        enable_preferences: bool = True,
        enable_llmlingua: bool = True,
        llmlingua_model: str = "llmlingua-2-bert-base-multilingual-cased-meetingbank",
        llmlingua_compression_ratio: float = 0.4,
        compress_query_threshold_tokens: int = 200,
        compress_query_threshold_chars: int = 800,
        use_active_retrieval: bool = False,
        active_retrieval_max_iterations: int = 3,
        active_retrieval_min_relevance: float = 0.65,
    ):
        """
        Initialize smart memory layer.
        
        Args:
            use_exact_cache: Enable exact match caching
            use_semantic_cache: Enable semantic caching
            vector_store_type: "faiss" or "chroma"
            embedding_model: Model name for embeddings
            similarity_threshold: Minimum similarity for semantic context (0.80-0.92)
            direct_return_threshold: Similarity for direct return without LLM call (>0.92)
            cache_size: Max cache entries
            eviction_policy: "lru" or "time-based"
            ttl_seconds: Time-to-live in seconds
            enable_compression: Enable LLM-Lingua style compression
            enable_preferences: Enable Mem0-style preference learning
        """
        self.use_exact_cache = use_exact_cache
        self.use_semantic_cache = use_semantic_cache
        self.enable_compression = enable_compression
        self.enable_preferences = enable_preferences
        self.direct_return_threshold = direct_return_threshold
        self.compress_query_threshold_tokens = compress_query_threshold_tokens
        self.compress_query_threshold_chars = compress_query_threshold_chars
        self.use_active_retrieval = use_active_retrieval
        self.active_retrieval_max_iterations = active_retrieval_max_iterations
        self.active_retrieval_min_relevance = active_retrieval_min_relevance
        
        # Initialize exact cache with optional persistent storage
        if use_exact_cache:
            self.exact_cache = MemoryCache(
                max_size=cache_size,
                eviction_policy=eviction_policy,
                ttl_seconds=ttl_seconds,
                similarity_threshold=similarity_threshold,
                persistent_storage=persistent_storage,
            )
        else:
            self.exact_cache = None
        
        # Initialize embedding model
        if use_semantic_cache:
            try:
                # Disable TensorFlow to avoid DLL issues on Windows
                import os
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model)
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                
                # Initialize vector store
                if vector_store_type == "faiss":
                    self.vector_store: VectorStore = FAISSVectorStore(dimension=embedding_dim)
                elif vector_store_type == "chroma":
                    self.vector_store = ChromaVectorStore()
                else:
                    raise ValueError(f"Unknown vector store type: {vector_store_type}")
            except ImportError as e:
                # Handle TensorFlow DLL issues on Windows gracefully
                logger.warning(
                    "Semantic cache disabled - sentence-transformers failed to load",
                    error=str(e)[:100],
                )
                self.use_semantic_cache = False
                self.vector_store = None
                self.embedding_model = None
            except Exception as e:
                logger.warning(
                    "Semantic cache disabled - initialization failed",
                    error=str(e)[:100],
                )
                self.use_semantic_cache = False
                self.vector_store = None
                self.embedding_model = None
        else:
            self.vector_store = None
            self.embedding_model = None
        
        self.similarity_threshold = similarity_threshold
        
        # Mem0-style user preferences
        self.user_preferences = UserPreferences() if enable_preferences else None
        
        # Initialize tokenizer for compression
        self._tokenizer = None
        if enable_compression:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not available, using character-based estimation")
        
        # Initialize LLMLingua-2 compressor if enabled
        self.llmlingua = None
        if enable_compression and enable_llmlingua:
            try:
                from ..compression.llmlingua_compressor import LLMLinguaCompressor
                self.llmlingua = LLMLinguaCompressor(
                    model_name=llmlingua_model,
                    compression_ratio=llmlingua_compression_ratio,
                )
                if not self.llmlingua.is_available():
                    logger.warning("LLMLingua-2 initialization failed, falling back to simple compression")
                    self.llmlingua = None
            except Exception as e:
                logger.warning(
                    "Failed to initialize LLMLingua-2, falling back to simple compression",
                    error=str(e)[:100],
                )
                self.llmlingua = None
        
        logger.info(
            "SmartMemoryLayer initialized",
            exact_cache=use_exact_cache,
            semantic_cache=use_semantic_cache,
            vector_store=vector_store_type,
            compression=enable_compression,
            llmlingua_enabled=self.llmlingua is not None,
            preferences=enable_preferences,
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            logger.warning("Embedding model not initialized, cannot generate embedding")
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            embedding_list = embedding.tolist()
            logger.debug(
                "Generated embedding",
                text_preview=text[:50],
                embedding_dim=len(embedding_list),
            )
            return embedding_list
        except Exception as e:
            logger.error(
                "Failed to generate embedding",
                error=str(e),
                text_preview=text[:50],
            )
            raise
    
    # =========================================================================
    # LLM-Lingua Style Compression
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4
    
    def compress_context(
        self,
        contexts: List[str],
        target_tokens: int,
    ) -> str:
        """
        Compress context using LLMLingua-2 or fallback to simple compression.
        
        Args:
            contexts: List of context strings to compress
            target_tokens: Target token count
        
        Returns:
            Compressed context string
        """
        if not contexts or not self.enable_compression:
            return " ".join(contexts) if contexts else ""
        
        full_text = " ".join(contexts)
        current_tokens = self.count_tokens(full_text)
        
        # If already within budget, return as-is
        if current_tokens <= target_tokens:
            return full_text
        
        # Use LLMLingua-2 if available
        if self.llmlingua and self.llmlingua.is_available():
            try:
                # Calculate target ratio from target_tokens
                target_ratio = target_tokens / current_tokens if current_tokens > 0 else 0.4
                # Ensure ratio is reasonable (not too aggressive)
                target_ratio = max(0.2, min(0.8, target_ratio))
                
                compressed = self.llmlingua.compress_context(contexts, target_ratio)
                
                # Verify compression worked and is within budget
                compressed_tokens = self.count_tokens(compressed)
                if compressed_tokens <= target_tokens or compressed_tokens < current_tokens:
                    compression_ratio = compressed_tokens / current_tokens if current_tokens > 0 else 1.0
                    logger.debug(
                        "Context compressed with LLMLingua-2",
                        original_tokens=current_tokens,
                        compressed_tokens=compressed_tokens,
                        ratio=f"{compression_ratio:.2%}",
                    )
                    return compressed
                else:
                    logger.warning(
                        "LLMLingua-2 compression didn't meet target, falling back to simple compression",
                        original_tokens=current_tokens,
                        compressed_tokens=compressed_tokens,
                        target_tokens=target_tokens,
                    )
            except Exception as e:
                logger.warning(
                    "LLMLingua-2 compression failed, falling back to simple compression",
                    error=str(e)[:100],
                )
        
        # Fallback to simple compression
        return self._compress_context_simple(contexts, target_tokens)
    
    def _compress_context_simple(
        self,
        contexts: List[str],
        target_tokens: int,
    ) -> str:
        """
        Simple sentence-scoring compression (fallback when LLMLingua-2 is unavailable).
        
        Strategy:
        1. Score sentences by importance (keywords, questions, length)
        2. Select highest-importance sentences within budget
        3. Return compressed context
        """
        full_text = " ".join(contexts)
        current_tokens = self.count_tokens(full_text)
        
        # Split into sentences
        sentences = []
        for ctx in contexts:
            for s in ctx.replace('!', '.').replace('?', '.').split('.'):
                s = s.strip()
                if s and len(s) > 10:
                    sentences.append(s)
        
        if not sentences:
            # Fallback: simple truncation
            if self._tokenizer:
                encoded = self._tokenizer.encode(full_text)
                return self._tokenizer.decode(encoded[:target_tokens])
            return full_text[:target_tokens * 4]
        
        # Score sentences by importance
        important_keywords = [
            "important", "key", "main", "primary", "essential", "critical",
            "must", "should", "need", "require", "first", "step", "example"
        ]
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            lower = sentence.lower()
            
            # Keyword presence
            score += sum(2 for kw in important_keywords if kw in lower)
            
            # Position bonus (earlier sentences often more important)
            score += max(0, 3 - i // 3)
            
            # Length bonus (medium-length sentences)
            if 30 < len(sentence) < 200:
                score += 2
            elif len(sentence) < 30:
                score += 1
            
            # Contains numbers (often specific/important)
            if any(c.isdigit() for c in sentence):
                score += 1
            
            scored_sentences.append((i, sentence, score, self.count_tokens(sentence)))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Select sentences within budget
        selected = []
        token_count = 0
        
        for idx, sentence, score, tokens in scored_sentences:
            if token_count + tokens <= target_tokens:
                selected.append((idx, sentence))
                token_count += tokens
            elif target_tokens - token_count > 20:
                # Truncate last sentence to fit
                remaining = target_tokens - token_count
                if self._tokenizer:
                    encoded = self._tokenizer.encode(sentence)
                    truncated = self._tokenizer.decode(encoded[:remaining - 5])
                else:
                    truncated = sentence[:remaining * 4 - 20]
                selected.append((idx, truncated + "..."))
                break
        
        # Re-sort by original position for coherence
        selected.sort(key=lambda x: x[0])
        
        compressed = ". ".join(s for _, s in selected)
        
        compression_ratio = self.count_tokens(compressed) / current_tokens if current_tokens > 0 else 1.0
        logger.debug(
            "Context compressed (simple fallback)",
            original_tokens=current_tokens,
            compressed_tokens=self.count_tokens(compressed),
            ratio=f"{compression_ratio:.2%}",
        )
        
        return compressed
    
    def compress_query_if_needed(self, query: str) -> str:
        """
        Compress query only if it exceeds thresholds.
        
        Args:
            query: Query string to potentially compress
        
        Returns:
            Compressed query string if thresholds exceeded, otherwise original query
        """
        if not self.enable_compression or not query:
            return query
        
        query_tokens = self.count_tokens(query)
        query_chars = len(query)
        
        # Check if query exceeds thresholds (compress if tokens > threshold OR chars > threshold)
        exceeds_token_threshold = query_tokens > self.compress_query_threshold_tokens
        exceeds_char_threshold = query_chars > self.compress_query_threshold_chars
        
        if not (exceeds_token_threshold or exceeds_char_threshold):
            logger.debug(
                "Query below compression thresholds, skipping",
                query_tokens=query_tokens,
                token_threshold=self.compress_query_threshold_tokens,
                query_chars=query_chars,
                char_threshold=self.compress_query_threshold_chars,
            )
            return query
        
        # Use LLMLingua-2 if available
        logger.debug(
            "Attempting query compression",
            query_tokens=query_tokens,
            query_chars=query_chars,
            token_threshold=self.compress_query_threshold_tokens,
            char_threshold=self.compress_query_threshold_chars,
        )
        
        if self.llmlingua and self.llmlingua.is_available():
            try:
                compressed = self.llmlingua.compress_query(
                    query,
                    max_tokens=self.compress_query_threshold_tokens,
                    max_chars=self.compress_query_threshold_chars,
                )
                
                # Verify compression actually reduced size
                compressed_tokens = self.count_tokens(compressed)
                if compressed_tokens < query_tokens:
                    logger.info(
                        "Query compressed with LLMLingua-2",
                        original_tokens=query_tokens,
                        compressed_tokens=compressed_tokens,
                        original_chars=query_chars,
                        compressed_chars=len(compressed),
                        savings=query_tokens - compressed_tokens,
                    )
                    return compressed
                else:
                    logger.debug(
                        "Query compression didn't reduce size, returning original",
                        original_tokens=query_tokens,
                        compressed_tokens=compressed_tokens,
                    )
                    return query
            except Exception as e:
                logger.warning(
                    "LLMLingua-2 query compression failed, returning original",
                    error=str(e)[:100],
                )
                return query
        
        # If LLMLingua not available, return original (don't use simple compression for queries)
        return query
    
    def compress_prompt(self, prompt: str, target_tokens: Optional[int] = None) -> str:
        """
        LLM-Lingua style compression for the ENTIRE prompt (not just context).
        
        This applies compression to reduce input tokens before sending to the LLM.
        Uses sentence importance scoring to preserve key information.
        
        Args:
            prompt: The full prompt to compress
            target_tokens: Target token count (if None, uses 80% of current)
        
        Returns:
            Compressed prompt
        """
        if not self.enable_compression or not prompt:
            return prompt
        
        current_tokens = self.count_tokens(prompt)
        
        # DON'T compress short prompts - they need to be fully preserved
        # Short queries (< 50 tokens) should pass through unchanged
        if current_tokens < 50:
            return prompt
        
        # If no target specified, aim for 20% reduction
        if target_tokens is None:
            target_tokens = int(current_tokens * 0.8)
        
        # If already within budget, return as-is
        if current_tokens <= target_tokens:
            return prompt
        
        # Split into sentences, preserving structure
        sentences = []
        for s in prompt.replace('!', '.').replace('?', '.').split('.'):
            s = s.strip()
            if s and len(s) > 5:  # Keep shorter fragments for prompts
                sentences.append(s)
        
        if not sentences:
            # Fallback: simple truncation
            if self._tokenizer:
                encoded = self._tokenizer.encode(prompt)
                return self._tokenizer.decode(encoded[:target_tokens])
            return prompt[:target_tokens * 4]
        
        # Score sentences by importance (different weights for prompts vs context)
        important_keywords = [
            "what", "how", "why", "when", "where", "who",  # Question words
            "please", "help", "need", "want", "can",  # Request words
            "issue", "problem", "error", "not working",  # Problem indicators
            "account", "payment", "refund", "cancel", "billing",  # Domain-specific
        ]
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            lower = sentence.lower()
            
            # Keyword presence (high weight for prompts)
            score += sum(3 for kw in important_keywords if kw in lower)
            
            # First and last sentences often most important in queries
            if i == 0:
                score += 5  # First sentence bonus
            if i == len(sentences) - 1:
                score += 3  # Last sentence bonus
            
            # Question marks indicate core query
            if '?' in sentence:
                score += 4
            
            # Contains numbers (specific details)
            if any(c.isdigit() for c in sentence):
                score += 2
            
            # Medium length sentences (not too short, not too long)
            if 20 < len(sentence) < 150:
                score += 2
            
            scored_sentences.append((i, sentence, score, self.count_tokens(sentence)))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # Select sentences within budget
        selected = []
        token_count = 0
        
        for idx, sentence, score, tokens in scored_sentences:
            if token_count + tokens <= target_tokens:
                selected.append((idx, sentence))
                token_count += tokens
        
        # Re-sort by original position for coherence
        selected.sort(key=lambda x: x[0])
        
        compressed = ". ".join(s for _, s in selected)
        if compressed and not compressed.endswith('.'):
            compressed += "."
        
        new_tokens = self.count_tokens(compressed)
        if current_tokens > new_tokens:
            logger.debug(
                "Prompt compressed",
                original_tokens=current_tokens,
                compressed_tokens=new_tokens,
                savings=current_tokens - new_tokens,
            )
        
        return compressed
    
    # =========================================================================
    # Mem0 Style Preference Learning
    # =========================================================================
    
    def detect_preferences(self, query: str) -> Tuple[str, str]:
        """
        Detect user preferences from query.
        
        Returns:
            (detected_tone, detected_format)
        """
        if not self.enable_preferences:
            return "neutral", "paragraph"
        
        query_lower = query.lower()
        
        # Tone detection with priority (order matters - check specific first)
        # Priority 1: Formal (most specific)
        formal_indicators = ["please", "would you", "could you", "kindly", "i would appreciate", "i request"]
        # Priority 2: Technical (specific domain)
        technical_indicators = ["implement", "algorithm", "optimize", "architecture", "api", "function", "code", "runtime", "deployment"]
        # Priority 3: Casual (specific style)
        casual_indicators = ["hey", "what's", "gimme", "yeah", "cool", "awesome", "dude"]
        # Priority 4: Simple (generic - lowest priority)
        simple_indicators = ["explain", "simple", "easy", "basic", "beginner", "eli5", "what is"]
        
        detected_tone = "neutral"
        
        # Check in priority order - first match wins
        if any(ind in query_lower for ind in formal_indicators):
            detected_tone = "formal"
        elif any(ind in query_lower for ind in technical_indicators):
            detected_tone = "technical"
        elif any(ind in query_lower for ind in casual_indicators):
            detected_tone = "casual"
        elif any(ind in query_lower for ind in simple_indicators):
            detected_tone = "simple"
        
        # Format detection
        format_indicators = {
            "list": ["list", "steps", "items", "bullet", "enumerate", "how to"],
            "code": ["code", "example", "implement", "function", "snippet", "script"],
            "concise": ["brief", "short", "quick", "summary", "tldr", "concise"],
        }
        
        detected_format = "paragraph"
        for fmt, indicators in format_indicators.items():
            if any(ind in query_lower for ind in indicators):
                detected_format = fmt
                break
        
        return detected_tone, detected_format
    
    def update_preferences(self, query: str, response: str = ""):
        """Update user preferences based on interaction."""
        if not self.enable_preferences or not self.user_preferences:
            return
        
        detected_tone, detected_format = self.detect_preferences(query)
        self.user_preferences.update(detected_tone, detected_format)
        
        logger.debug(
            "Preferences updated",
            tone=self.user_preferences.tone,
            format=self.user_preferences.format,
            confidence=f"{self.user_preferences.confidence:.2f}",
        )
    
    def get_preference_context(self) -> Dict[str, Any]:
        """Get current user preferences for prompt customization."""
        if not self.enable_preferences or not self.user_preferences:
            return {}
        
        if self.user_preferences.confidence < 0.5:
            return {}  # Not confident enough
        
        return {
            "tone": self.user_preferences.tone,
            "format": self.user_preferences.format,
            "detail_level": self.user_preferences.detail_level,
        }
    
    # =========================================================================
    # Enhanced Retrieval with Compression
    # =========================================================================
    
    def retrieve_compressed(
        self,
        query: str,
        context_token_budget: int = 500,
        top_k: int = 5,
    ) -> Tuple[Optional[CacheEntry], str, Dict[str, Any], Optional[float], Dict[str, Any]]:
        """
        Retrieve from cache with tiered similarity matching.
        
        Tiers:
        - High similarity (>direct_return_threshold): Return cached response directly (0 tokens)
        - Medium similarity (similarity_threshold to direct_return_threshold): Use as context
        - Low similarity (<similarity_threshold): No cache benefit
        
        Returns:
            (cache_entry_or_none, compressed_context, preference_context, similarity_score, memory_metrics)
        """
        # Initialize memory metrics
        memory_metrics = {
            "exact_lookup": False,
            "semantic_search": False,
            "semantic_matches_found": 0,
            "top_similarity": None,
            "context_retrieval": False,
            "preference_retrieval": False,
            "context_compression": False,
        }
        
        # Update preferences from query
        self.update_preferences(query)
        preference_context = self.get_preference_context()
        memory_metrics["preference_retrieval"] = bool(preference_context)
        
        # Standard retrieval or active retrieval
        if self.use_active_retrieval and self.use_semantic_cache and self.vector_store:
            # Use active retrieval for iterative context gathering
            try:
                from .active_retriever import ActiveRetriever
                active_retriever = ActiveRetriever(
                    vector_store=self.vector_store,
                    embedding_model=self.embedding_model,
                    min_relevance=self.active_retrieval_min_relevance,
                    max_iterations=self.active_retrieval_max_iterations,
                )
                
                # Get LLM provider for follow-up generation (if available)
                # Note: ActiveRetriever can work without LLM provider (uses heuristic)
                llm_provider = None
                
                # Iterative retrieval
                active_results = active_retriever.retrieve_iteratively(
                    query=query,
                    top_k=top_k,
                    llm_provider=llm_provider,
                )
                
                # Convert to semantic_matches format
                exact_match = None
                semantic_matches = []
                
                # Check for exact match first
                if self.use_exact_cache:
                    exact_match = self.exact_cache.get_exact(query)
                    if exact_match:
                        memory_metrics["exact_lookup"] = True
                        return exact_match, "", preference_context, None, memory_metrics
                
                # Convert active retrieval results to cache entries
                if self.exact_cache:
                    all_entries = self.exact_cache.get_all_entries()
                    entry_map = {e.query_hash: e for e in all_entries}
                    
                    for entry_id, similarity, metadata in active_results:
                        if entry_id in entry_map:
                            entry = entry_map[entry_id]
                            entry.metadata["similarity"] = similarity
                            semantic_matches.append(entry)
                
                memory_metrics["exact_lookup"] = True
                memory_metrics["semantic_search"] = True
                memory_metrics["active_retrieval_used"] = True
                memory_metrics["active_retrieval_iterations"] = len(active_results) // top_k + 1
                
            except Exception as e:
                logger.warning("Active retrieval failed, falling back to standard retrieval", error=str(e))
                # Fallback to standard retrieval
                exact_match, semantic_matches = self.retrieve(query, top_k)
                memory_metrics["exact_lookup"] = True
                memory_metrics["semantic_search"] = self.use_semantic_cache
        else:
            # Standard retrieval
            exact_match, semantic_matches = self.retrieve(query, top_k)
            memory_metrics["exact_lookup"] = True
            memory_metrics["semantic_search"] = self.use_semantic_cache
        
        # Track semantic matches
        if semantic_matches:
            memory_metrics["semantic_matches_found"] = len(semantic_matches)
            top_similarity = semantic_matches[0].metadata.get("similarity", 0) if semantic_matches else None
            memory_metrics["top_similarity"] = top_similarity
        
        # Tier 1: Exact match - return immediately
        if exact_match:
            logger.info("Exact cache hit", query_hash=exact_match.query_hash[:8])
            return exact_match, "", preference_context, None, memory_metrics
        
        # Tier 2: High similarity semantic match - direct return (like exact cache)
        if semantic_matches:
            top_match = semantic_matches[0]
            similarity = top_match.metadata.get("similarity", 0)
            
            if similarity >= self.direct_return_threshold:
                logger.info(
                    "Semantic direct return",
                    similarity=f"{similarity:.3f}",
                    threshold=self.direct_return_threshold,
                    original_query=top_match.query[:50],
                )
                return top_match, "", preference_context, similarity, memory_metrics
            
            # Tier 3: Medium similarity - use as compressed context
            logger.debug(
                "Semantic context match",
                similarity=f"{similarity:.3f}",
                num_matches=len(semantic_matches),
            )
            contexts = [match.response for match in semantic_matches]
            memory_metrics["context_retrieval"] = True
            compressed_context = self.compress_context(contexts, context_token_budget)
            memory_metrics["context_compression"] = True
            return None, compressed_context, preference_context, similarity, memory_metrics
        
        # Tier 4: No matches - full LLM call needed
        return None, "", preference_context, None, memory_metrics
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[Optional[CacheEntry], List[CacheEntry]]:
        """
        Retrieve from cache: exact match first, then semantic.
        
        Returns:
            (exact_match_entry, list_of_semantic_matches)
        """
        exact_match = None
        semantic_matches = []
        
        # Try exact match first
        if self.use_exact_cache:
            exact_match = self.exact_cache.get_exact(query)
            if exact_match:
                logger.info(
                    "Exact cache hit",
                    query_hash=exact_match.query_hash[:8],
                    query_preview=query[:50],
                )
                return exact_match, []
            else:
                logger.debug("Exact cache miss", query_preview=query[:50])
        
        # Try semantic match
        if self.use_semantic_cache and self.vector_store:
            try:
                # Generate embedding for query
                query_embedding = self.get_embedding(query)
                if query_embedding is None or len(query_embedding) == 0:
                    logger.warning("Failed to generate embedding for query", query_preview=query[:50])
                    return exact_match, []
                
                logger.debug(
                    "Searching semantic cache",
                    query_preview=query[:50],
                    embedding_dim=len(query_embedding),
                    threshold=self.similarity_threshold,
                )
                
                similar_entries = self.vector_store.search(
                    query_embedding,
                    top_k=top_k,
                    threshold=self.similarity_threshold,
                )
                
                logger.debug(
                    "Semantic search completed",
                    num_results=len(similar_entries),
                    query_preview=query[:50],
                )
                
                # Convert vector store results to cache entries
                if self.exact_cache:
                    all_entries = self.exact_cache.get_all_entries()
                    entry_map = {e.query_hash: e for e in all_entries}
                    
                    for entry_id, similarity, metadata in similar_entries:
                        if entry_id in entry_map:
                            entry = entry_map[entry_id]
                            entry.metadata["similarity"] = similarity
                            semantic_matches.append(entry)
                            logger.debug(
                                "Semantic match found",
                                entry_id=entry_id[:8],
                                similarity=f"{similarity:.3f}",
                                threshold=self.similarity_threshold,
                                direct_return_threshold=self.direct_return_threshold,
                            )
                        else:
                            logger.warning(
                                "Semantic match entry not found in exact cache",
                                entry_id=entry_id[:8],
                            )
                
                if semantic_matches:
                    top_similarity = semantic_matches[0].metadata.get("similarity", 0)
                    logger.info(
                        "Semantic cache matches found",
                        num_matches=len(semantic_matches),
                        top_similarity=f"{top_similarity:.3f}",
                        threshold=self.similarity_threshold,
                        direct_return_threshold=self.direct_return_threshold,
                        query_preview=query[:50],
                    )
                else:
                    logger.debug(
                        "No semantic matches above threshold",
                        query_preview=query[:50],
                        threshold=self.similarity_threshold,
                        num_vector_results=len(similar_entries),
                    )
            except Exception as e:
                logger.error(
                    "Error during semantic cache retrieval",
                    error=str(e),
                    query_preview=query[:50],
                )
        
        return exact_match, semantic_matches
    
    def store(
        self,
        query: str,
        response: str,
        tokens_used: int = 0,
        metadata: Optional[dict] = None,
    ) -> CacheEntry:
        """Store query-response pair in cache."""
        query_embedding = None
        
        # Generate embedding if semantic cache is enabled
        if self.use_semantic_cache and self.embedding_model:
            query_embedding = self.get_embedding(query)
        
        # Store in exact cache
        entry = None
        if self.use_exact_cache:
            entry = self.exact_cache.put(
                query=query,
                response=response,
                query_embedding=query_embedding,
                tokens_used=tokens_used,
                metadata=metadata or {},
            )
        
        # Store in vector store for semantic search
        if self.use_semantic_cache and self.vector_store and query_embedding:
            entry_id = entry.query_hash if entry else hashlib.sha256(query.encode()).hexdigest()
            try:
                self.vector_store.add(
                    id=entry_id,
                    embedding=query_embedding,
                    metadata={
                        "query": query[:100],  # Truncate for storage
                        "response": response[:200],
                        "tokens_used": tokens_used,
                        **(metadata or {}),
                    },
                )
                logger.info(
                    "Stored in vector store",
                    entry_id=entry_id[:8],
                    query_preview=query[:50],
                    embedding_dim=len(query_embedding),
                )
            except Exception as e:
                logger.error(
                    "Failed to store in vector store",
                    error=str(e),
                    entry_id=entry_id[:8],
                    query_preview=query[:50],
                )
        
        logger.info("Stored in memory", tokens_saved=tokens_used, query_hash=(entry.query_hash[:8] if entry else "N/A"))
        return entry
    
    def clear(self):
        """Clear all caches."""
        if self.exact_cache:
            self.exact_cache.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("Memory layer cleared")
    
    def stats(self) -> dict:
        """Get statistics."""
        stats = {
            "exact_cache_enabled": self.use_exact_cache,
            "semantic_cache_enabled": self.use_semantic_cache,
            "compression_enabled": self.enable_compression,
            "preferences_enabled": self.enable_preferences,
        }
        
        if self.exact_cache:
            stats.update(self.exact_cache.stats())
        
        if self.user_preferences:
            stats["preferences"] = {
                "tone": self.user_preferences.tone,
                "format": self.user_preferences.format,
                "confidence": self.user_preferences.confidence,
                "interactions": self.user_preferences.interaction_count,
            }
        
        return stats

