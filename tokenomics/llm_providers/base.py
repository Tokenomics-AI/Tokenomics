"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass
import time
import random
import structlog

logger = structlog.get_logger()


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    text: str
    tokens_used: int
    model: str
    latency_ms: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize LLM provider.
        
        Args:
            model: Model name
            **kwargs: Provider-specific configuration
        """
        self.model = model
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n: int = 1,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            n: Number of completions
            **kwargs: Additional provider-specific parameters
        
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text with exponential backoff retry logic.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_multiplier: Multiplier for exponential backoff
            **kwargs: Arguments passed to generate()
        
        Returns:
            LLMResponse object
        
        Raises:
            RuntimeError: If all retries fail
        """
        last_exception = None
        backoff = initial_backoff
        
        for attempt in range(max_retries + 1):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate backoff with jitter
                    jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
                    sleep_time = min(backoff + jitter, max_backoff)
                    
                    logger.warning(
                        "LLM generation failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        backoff_seconds=round(sleep_time, 2),
                        error=str(e)[:100],
                    )
                    
                    time.sleep(sleep_time)
                    backoff *= backoff_multiplier
                else:
                    logger.error(
                        "LLM generation failed after all retries",
                        max_retries=max_retries,
                        error=str(e),
                    )
        
        raise RuntimeError(f"LLM generation failed after {max_retries} retries: {last_exception}") from last_exception
    
    def generate_multiple(
        self,
        prompt: str,
        n: int = 3,
        **kwargs,
    ) -> List[LLMResponse]:
        """Generate multiple completions."""
        responses = []
        for _ in range(n):
            response = self.generate(prompt, n=1, **kwargs)
            responses.append(response)
        return responses

