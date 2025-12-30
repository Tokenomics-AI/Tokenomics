"""OpenAI provider."""

import os
from typing import Optional
import time

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key: OpenAI API key
            base_url: Custom base URL (for OpenAI-compatible APIs)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install with: pip install openai")
        
        super().__init__(model, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided or set OPENAI_API_KEY")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the model
                if "gpt-4" in model:
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in model:
                    self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None
    
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
        """Generate text using OpenAI API."""
        start_time = time.time()
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=n,
                **kwargs,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Get first choice
            choice = response.choices[0]
            text = choice.message.content or ""
            
            # Get token usage breakdown
            if response.usage:
                tokens_used = response.usage.total_tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            else:
                tokens_used = 0
                prompt_tokens = 0
                completion_tokens = 0
            
            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                model=self.model,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": choice.finish_reason,
                    "n": n,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise RuntimeError(f"OpenAI API error: {e}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback estimate
        return len(text) // 4

