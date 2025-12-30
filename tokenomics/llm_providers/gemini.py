"""Google Gemini provider via Vertex AI or API key."""

import os
from typing import Optional, Dict
import time

# Try google-generativeai first (API key based)
try:
    import google.generativeai as genai
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False

# Try vertexai (service account based)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Google Gemini provider using API key or Vertex AI."""
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs,
    ):
        """
        Initialize Gemini provider.
        
        Args:
            model: Model name (e.g., "gemini-2.0-flash-exp", "gemini-pro")
            api_key: Google API key (preferred, simpler)
            project_id: GCP project ID (for Vertex AI, alternative to API key)
            location: GCP location (for Vertex AI)
        """
        super().__init__(model, **kwargs)
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.use_vertex_ai = False
        
        # Prefer API key method (simpler)
        if self.api_key and GEMINI_API_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model_name=self.model)
            self.use_vertex_ai = False
        elif self.project_id and VERTEX_AI_AVAILABLE:
            # Fall back to Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            self.client = GenerativeModel(model_name=self.model)
            self.use_vertex_ai = True
        else:
            if not GEMINI_API_AVAILABLE and not VERTEX_AI_AVAILABLE:
                raise ImportError(
                    "Either google-generativeai or google-cloud-aiplatform is required. "
                    "Install with: pip install google-generativeai"
                )
            if not self.api_key and not self.project_id:
                raise ValueError(
                    "Either api_key (GEMINI_API_KEY) or project_id (GOOGLE_CLOUD_PROJECT) must be provided"
                )
    
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
        """Generate text using Gemini."""
        start_time = time.time()
        
        try:
            if self.use_vertex_ai:
                # Vertex AI method
                config = GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p if top_p < 1.0 else None,
                )
                response = self.client.generate_content(
                    prompt,
                    generation_config=config,
                )
                text = response.text if response.text else ""
                candidates_count = len(response.candidates) if hasattr(response, 'candidates') else 1
            else:
                # API key method (google-generativeai)
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
                if top_p < 1.0:
                    generation_config["top_p"] = top_p
                
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                text = response.text if response.text else ""
                candidates_count = len(response.candidates) if hasattr(response, 'candidates') else 1
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens (Gemini doesn't always return token counts)
            # Rough estimate: 1 token ≈ 4 characters
            tokens_used = len(text) // 4 + len(prompt) // 4
            
            return LLMResponse(
                text=text,
                tokens_used=tokens_used,
                model=self.model,
                latency_ms=latency_ms,
                metadata={
                    "candidates": candidates_count,
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise RuntimeError(f"Gemini API error: {e}") from e
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (rough estimate for Gemini)."""
        # Gemini doesn't expose tokenizer, use rough estimate
        return len(text) // 4

