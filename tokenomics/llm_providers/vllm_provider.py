"""vLLM provider for self-hosted models."""

from typing import Optional
import time
import requests

from .base import LLMProvider, LLMResponse


class vLLMProvider(LLMProvider):
    """vLLM provider for self-hosted models."""
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize vLLM provider.
        
        Args:
            model: Model name/path
            base_url: vLLM server URL
            api_key: API key (if required)
        """
        super().__init__(model, **kwargs)
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        
        # vLLM uses OpenAI-compatible API
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
    
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
        """Generate text using vLLM API."""
        start_time = time.time()
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            **kwargs,
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            data = response.json()
            
            # Get first choice
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            
            # Get token usage
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                model=self.model,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": choice.get("finish_reason"),
                    "n": n,
                },
            )
        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            raise RuntimeError(f"vLLM API error: {e}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (rough estimate for vLLM)."""
        # vLLM doesn't expose tokenizer easily, use rough estimate
        return len(text) // 4

