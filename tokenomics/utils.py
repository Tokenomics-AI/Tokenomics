"""Utility functions for Tokenomics platform."""

from typing import List, Dict, Optional
import re


def compute_quality_score(
    response: str,
    reference: Optional[str] = None,
    method: str = "length",
) -> float:
    """
    Compute quality score for a response.
    
    Args:
        response: Generated response
        reference: Reference answer (optional)
        method: Scoring method ("length", "bleu", "similarity")
    
    Returns:
        Quality score (0-1)
    """
    if method == "length":
        # Simple heuristic: longer responses are better (up to a point)
        length = len(response)
        # Normalize to 0-1 (assuming 100-1000 chars is good)
        score = min(1.0, max(0.0, (length - 50) / 500))
        return score
    
    elif method == "similarity" and reference:
        # Simple word overlap similarity
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if not reference_words:
            return 0.0
        
        overlap = len(response_words & reference_words)
        score = overlap / len(reference_words)
        return min(1.0, score)
    
    else:
        # Default: assume good quality
        return 0.8


def rerank_responses(
    responses: List[str],
    query: str,
    method: str = "length",
) -> List[tuple[str, float]]:
    """
    Rerank multiple responses.
    
    Args:
        responses: List of response texts
        query: Original query
        method: Reranking method
    
    Returns:
        List of (response, score) tuples, sorted by score
    """
    scored = []
    
    for response in responses:
        if method == "length":
            score = len(response)
        else:
            score = 0.5  # Default
        
        scored.append((response, score))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)


def extract_query_type(query: str) -> str:
    """
    Extract query type for contextual bandits.
    
    Args:
        query: User query
    
    Returns:
        Query type: "simple", "medium", or "complex"
    """
    query_lower = query.lower()
    
    # Simple heuristics
    if len(query) < 50:
        return "simple"
    elif any(keyword in query_lower for keyword in ["explain", "analyze", "compare", "why"]):
        return "complex"
    else:
        return "medium"

