"""Active retrieval for iterative context gathering."""

from typing import List, Optional, Tuple
import structlog

logger = structlog.get_logger()


class ActiveRetriever:
    """Iterative retrieval that gathers context until sufficient."""
    
    def __init__(
        self,
        vector_store,
        embedding_model,
        min_relevance: float = 0.65,
        max_iterations: int = 3,
    ):
        """
        Initialize active retriever.
        
        Args:
            vector_store: Vector store for semantic search
            embedding_model: Embedding model for generating query embeddings
            min_relevance: Minimum relevance score for sufficiency
            max_iterations: Maximum number of retrieval iterations
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.min_relevance = min_relevance
        self.max_iterations = max_iterations
        
        logger.info(
            "ActiveRetriever initialized",
            min_relevance=min_relevance,
            max_iterations=max_iterations,
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def is_context_sufficient(
        self,
        query: str,
        contexts: List[Tuple[str, float, dict]],  # (id, similarity, metadata)
        min_relevance: Optional[float] = None,
    ) -> bool:
        """
        Check if retrieved context is sufficient to answer the query.
        
        Args:
            query: Original query
            contexts: List of (entry_id, similarity, metadata) tuples
            min_relevance: Minimum relevance threshold (uses instance default if None)
        
        Returns:
            True if context is sufficient, False otherwise
        """
        if not contexts:
            return False
        
        threshold = min_relevance or self.min_relevance
        
        # Check if top contexts have high enough similarity
        if contexts:
            top_similarity = contexts[0][1] if len(contexts[0]) > 1 else 0.0
            avg_similarity = sum(ctx[1] for ctx in contexts if len(ctx) > 1) / len(contexts)
            
            # Sufficient if top similarity is high OR average is good
            if top_similarity >= threshold or avg_similarity >= (threshold - 0.1):
                logger.debug(
                    "Context sufficient",
                    top_similarity=top_similarity,
                    avg_similarity=avg_similarity,
                    threshold=threshold,
                )
                return True
        
        logger.debug(
            "Context insufficient",
            num_contexts=len(contexts),
            top_similarity=contexts[0][1] if contexts else 0.0,
            threshold=threshold,
        )
        return False
    
    def generate_followup_query(
        self,
        original_query: str,
        existing_contexts: List[Tuple[str, float, dict]],
        llm_provider=None,
    ) -> str:
        """
        Generate a follow-up query to retrieve missing information.
        
        Args:
            original_query: Original user query
            existing_contexts: Already retrieved contexts
            llm_provider: Optional LLM provider for generating follow-up (if None, uses simple heuristic)
        
        Returns:
            Follow-up query string
        """
        if llm_provider:
            # Use LLM to generate refined query
            try:
                context_summaries = [ctx[2].get("query", "")[:100] for ctx in existing_contexts[:3]]
                prompt = f"""Original query: "{original_query}"

Already retrieved information about:
{chr(10).join(f"- {summary}" for summary in context_summaries)}

Generate a refined search query to find additional information that would help answer the original query. Focus on aspects not yet covered.

Refined query:"""
                
                response = llm_provider.generate(
                    prompt,
                    max_tokens=50,
                    temperature=0.7,
                )
                
                followup = response.text.strip().strip('"').strip("'")
                logger.debug("Generated follow-up query", original=original_query[:50], followup=followup[:50])
                return followup
            except Exception as e:
                logger.warning("Failed to generate follow-up query with LLM, using heuristic", error=str(e))
        
        # Fallback: Simple heuristic - add "more information about" or focus on question words
        query_lower = original_query.lower()
        
        # Extract key terms
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        key_terms = [word for word in original_query.split() if word.lower() not in question_words and len(word) > 3]
        
        if key_terms:
            # Use key terms with "more about" or "additional information"
            followup = f"more information about {', '.join(key_terms[:3])}"
        else:
            # Fallback to original query with emphasis
            followup = f"detailed information {original_query}"
        
        logger.debug("Generated follow-up query (heuristic)", original=original_query[:50], followup=followup[:50])
        return followup
    
    def retrieve_iteratively(
        self,
        query: str,
        top_k: int = 3,
        llm_provider=None,
    ) -> List[Tuple[str, float, dict]]:
        """
        Iteratively retrieve contexts until sufficient or max iterations.
        
        Args:
            query: User query
            top_k: Number of results per iteration
            llm_provider: Optional LLM provider for follow-up generation
        
        Returns:
            List of (entry_id, similarity, metadata) tuples
        """
        all_contexts = []
        seen_ids = set()
        current_query = query
        
        for iteration in range(self.max_iterations):
            logger.debug(
                "Active retrieval iteration",
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
                query_preview=current_query[:50],
            )
            
            # Generate embedding for current query
            try:
                query_embedding = self.get_embedding(current_query)
            except Exception as e:
                logger.error("Failed to generate embedding for active retrieval", error=str(e))
                break
            
            # Retrieve top-k
            new_contexts = self.vector_store.search(
                query_embedding,
                top_k=top_k * 2,  # Get more to filter out duplicates
                threshold=self.min_relevance - 0.1,  # Slightly lower threshold for retrieval
            )
            
            # Filter out duplicates and convert to consistent format
            unique_contexts = []
            for ctx in new_contexts:
                # Handle both tuple and dict formats
                if isinstance(ctx, tuple):
                    entry_id = ctx[0]
                    similarity = ctx[1] if len(ctx) > 1 else 0.0
                    metadata = ctx[2] if len(ctx) > 2 else {}
                else:
                    entry_id = ctx.get("id", "")
                    similarity = ctx.get("similarity", 0.0)
                    metadata = ctx.get("metadata", {})
                
                if entry_id and entry_id not in seen_ids:
                    seen_ids.add(entry_id)
                    unique_contexts.append((entry_id, similarity, metadata))
            
            # Add to all contexts
            all_contexts.extend(unique_contexts[:top_k])
            
            # Check sufficiency
            if self.is_context_sufficient(query, all_contexts):
                logger.info(
                    "Active retrieval: context sufficient",
                    iterations=iteration + 1,
                    total_contexts=len(all_contexts),
                )
                break
            
            # Generate follow-up query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = self.generate_followup_query(query, all_contexts, llm_provider)
            else:
                logger.debug("Active retrieval: max iterations reached", total_contexts=len(all_contexts))
        
        # Sort by similarity and return top results
        all_contexts.sort(key=lambda x: x[1] if len(x) > 1 else 0.0, reverse=True)
        
        logger.info(
            "Active retrieval completed",
            iterations=iteration + 1,
            total_contexts=len(all_contexts),
            top_similarity=all_contexts[0][1] if all_contexts else 0.0,
        )
        
        return all_contexts[:top_k * 2]  # Return top results






