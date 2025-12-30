"""Quality judge for comparing baseline vs optimized answers."""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
import structlog

from ..config import JudgeConfig
from ..llm_providers import LLMProvider, OpenAIProvider, GeminiProvider, vLLMProvider

logger = structlog.get_logger()


@dataclass
class JudgeResult:
    """Result from quality judge evaluation."""
    winner: str  # "baseline", "optimized", or "equivalent"
    explanation: str
    confidence: float  # 0.0 to 1.0


class QualityJudge:
    """Quality judge using LLM to compare answers."""
    
    def __init__(self, config: JudgeConfig):
        """
        Initialize quality judge.
        
        Args:
            config: Judge configuration
        """
        self.config = config
        self.judge_provider = None
        
        if config.enabled:
            self.judge_provider = self._create_judge_provider()
            logger.info("QualityJudge initialized", provider=config.provider, model=config.model)
        else:
            logger.info("QualityJudge disabled")
    
    def _create_judge_provider(self) -> Optional[LLMProvider]:
        """Create LLM provider for judge."""
        try:
            if self.config.provider == "openai":
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                return OpenAIProvider(
                    model=self.config.model,
                    api_key=api_key,
                )
            elif self.config.provider == "gemini":
                api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                return GeminiProvider(
                    model=self.config.model,
                    api_key=api_key,
                )
            elif self.config.provider == "vllm":
                return vLLMProvider(
                    model=self.config.model,
                    base_url=self.config.api_key,  # Use api_key field for base_url
                )
            else:
                logger.warning("Unknown judge provider", provider=self.config.provider)
                return None
        except Exception as e:
            logger.error("Failed to create judge provider", error=str(e))
            return None
    
    def judge(
        self,
        query: str,
        baseline_answer: str,
        optimized_answer: str,
    ) -> Optional[JudgeResult]:
        """
        Judge which answer is better.
        
        Args:
            query: Original user query
            baseline_answer: Baseline answer
            optimized_answer: Optimized answer
        
        Returns:
            JudgeResult with winner, explanation, and confidence, or None if judge disabled/failed
        """
        if not self.config.enabled or not self.judge_provider:
            return None
        
        # Truncate answers if too long
        max_answer_length = 500
        baseline_truncated = baseline_answer[:max_answer_length]
        optimized_truncated = optimized_answer[:max_answer_length]
        
        prompt = f"""Compare these two answers to the question: "{query}"

Baseline answer: "{baseline_truncated}"
Optimized answer: "{optimized_truncated}"

Which is better? Respond with valid JSON only:
{{
  "winner": "baseline" | "optimized" | "equivalent",
  "explanation": "brief explanation (1-2 sentences)",
  "confidence": 0.0-1.0
}}"""

        try:
            response = self.judge_provider.generate(
                prompt,
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent judging
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result_data = json.loads(response_text)
            
            winner = result_data.get("winner", "equivalent").lower()
            explanation = result_data.get("explanation", "No explanation provided")
            confidence = float(result_data.get("confidence", 0.5))
            
            # Normalize winner
            if winner not in ["baseline", "optimized", "equivalent"]:
                if "baseline" in winner.lower():
                    winner = "baseline"
                elif "optimized" in winner.lower():
                    winner = "optimized"
                else:
                    winner = "equivalent"
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            result = JudgeResult(
                winner=winner,
                explanation=explanation,
                confidence=confidence,
            )
            
            logger.info(
                "Quality judged",
                winner=winner,
                confidence=confidence,
                explanation=explanation[:50],
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse judge JSON response", error=str(e), response=response.text[:100])
            # Fallback: try to infer from text
            response_lower = response.text.lower()
            if "baseline" in response_lower and "better" in response_lower:
                winner = "baseline"
            elif "optimized" in response_lower and "better" in response_lower:
                winner = "optimized"
            else:
                winner = "equivalent"
            
            return JudgeResult(
                winner=winner,
                explanation="Parsed from text response",
                confidence=0.5,
            )
        except Exception as e:
            logger.error("Judge evaluation failed", error=str(e))
            return None
    
    def quick_quality_check(self, query: str, response: str) -> float:
        """
        Lightweight quality check using heuristics (fast, no LLM call).
        
        This is used for cascading inference to quickly determine if a response
        is good enough, without the overhead of a full judge comparison.
        
        Args:
            query: Original user query
            response: Generated response
        
        Returns:
            Quality score (0.0-1.0), where 1.0 is high quality
        """
        if not response or len(response.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # 1. Response length check (too short = likely incomplete)
        response_length = len(response)
        query_length = len(query)
        
        # Good responses are typically 2-10x the query length
        if response_length < query_length * 0.5:
            # Response is shorter than query - likely incomplete
            score += 0.2
        elif response_length < query_length:
            # Response is similar length to query - might be too brief
            score += 0.5
        elif response_length < query_length * 10:
            # Response is reasonable length
            score += 0.8
        else:
            # Very long response - might be verbose but complete
            score += 0.9
        
        # 2. Completeness indicators
        response_lower = response.lower()
        
        # Check for common completion patterns
        completeness_indicators = [
            "in conclusion", "to summarize", "in summary",
            "therefore", "thus", "overall", "in short"
        ]
        if any(indicator in response_lower for indicator in completeness_indicators):
            score += 0.1  # Bonus for explicit conclusion
        
        # Check for question answering patterns
        question_words = ["what", "how", "why", "when", "where", "who"]
        query_has_question = any(word in query.lower() for word in question_words)
        
        if query_has_question:
            # Check if response seems to answer the question
            # Simple heuristic: response should be longer and contain relevant terms
            if response_length > query_length * 2:
                score += 0.1
        
        # 3. Relevance check (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response_lower.split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "can", "to", "of", "in",
                     "on", "at", "for", "with", "by", "from", "as", "and", "or",
                     "but", "if", "that", "this", "these", "those"}
        
        query_keywords = query_words - stop_words
        response_keywords = response_words - stop_words
        
        if query_keywords:
            overlap = len(query_keywords & response_keywords) / len(query_keywords)
            score += overlap * 0.3  # Up to 0.3 points for keyword overlap
        
        # 4. Structure check (well-formed responses have structure)
        # Check for paragraphs, lists, or structured content
        has_paragraphs = "\n\n" in response or response.count("\n") >= 2
        has_lists = any(marker in response for marker in ["1.", "2.", "- ", "* ", "• "])
        
        if has_paragraphs or has_lists:
            score += 0.1  # Bonus for structured response
        
        # 5. Error indicators (penalize obvious errors)
        error_indicators = [
            "i'm sorry", "i cannot", "i don't understand", "error",
            "unable to", "cannot process", "invalid"
        ]
        if any(indicator in response_lower for indicator in error_indicators):
            score -= 0.3  # Penalty for error messages
        
        # Normalize to 0.0-1.0
        score = max(0.0, min(1.0, score))
        
        logger.debug(
            "Quick quality check",
            query_preview=query[:50],
            response_length=response_length,
            quality_score=score,
        )
        
        return score

