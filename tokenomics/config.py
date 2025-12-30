"""Configuration management for the Tokenomics platform."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(default="gemini", description="LLM provider: gemini, openai, vllm")
    model: str = Field(default="gemini-2.0-flash-exp", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key (if needed)")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    project_id: Optional[str] = Field(default=None, description="GCP project ID for Vertex AI")
    location: Optional[str] = Field(default="us-central1", description="GCP location")


class MemoryConfig(BaseModel):
    """Configuration for memory layer."""
    use_exact_cache: bool = Field(default=True, description="Enable exact match caching")
    use_semantic_cache: bool = Field(default=True, description="Enable semantic caching")
    vector_store: str = Field(default="faiss", description="Vector store: faiss or chroma")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    cache_size: int = Field(default=1000, description="Max cache entries")
    similarity_threshold: float = Field(default=0.70, description="Minimum similarity for semantic context")
    direct_return_threshold: float = Field(default=0.80, description="Similarity threshold for direct cache return (no LLM call)")
    eviction_policy: str = Field(default="lru", description="Eviction policy: lru or time-based")
    ttl_seconds: Optional[int] = Field(default=None, description="Time-to-live in seconds")
    persistent_cache_path: Optional[str] = Field(default="tokenomics_cache.db", description="Path to SQLite database for persistent cache (None = in-memory only)")
    # LLMLingua configuration
    enable_llmlingua: bool = Field(default=True, description="Enable LLMLingua-2 compression")
    llmlingua_model: str = Field(
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        description="LLMLingua-2 model for compression"
    )
    llmlingua_compression_ratio: float = Field(default=0.4, description="Target compression ratio (0.4 = keep 40%)")
    compress_query_threshold_tokens: int = Field(default=150, description="Compress queries longer than this token count")
    compress_query_threshold_chars: int = Field(default=500, description="Compress queries longer than this char count")
    use_active_retrieval: bool = Field(default=False, description="Enable iterative active retrieval")
    active_retrieval_max_iterations: int = Field(default=3, description="Max retrieval iterations")
    active_retrieval_min_relevance: float = Field(default=0.65, description="Minimum relevance for sufficiency")


class OrchestratorConfig(BaseModel):
    """Configuration for token-aware orchestrator."""
    default_token_budget: int = Field(default=4000, description="Default token budget")
    max_context_tokens: int = Field(default=8000, description="Max context window size")
    use_knapsack_optimization: bool = Field(default=True, description="Use knapsack for token allocation")
    compression_ratio: float = Field(default=0.5, description="Compression ratio for low-value content")
    enable_multi_model_routing: bool = Field(default=True, description="Enable multi-model routing")
    max_context_ratio: float = Field(
        default=0.4,
        description="Maximum ratio of budget allocated to context (0.0-1.0)"
    )
    min_response_ratio: float = Field(
        default=0.3,
        description="Minimum ratio of budget reserved for response generation"
    )


class BanditConfig(BaseModel):
    """Configuration for bandit optimizer."""
    algorithm: str = Field(default="ucb", description="Bandit algorithm: ucb, epsilon_greedy, thompson")
    exploration_rate: float = Field(default=0.1, description="Exploration rate for epsilon-greedy")
    reward_lambda: float = Field(default=0.001, description="Lambda for reward = quality - lambda * tokens")
    reset_frequency: Optional[int] = Field(default=None, description="Reset bandit after N queries")
    contextual: bool = Field(default=False, description="Use contextual bandits by query type")
    state_file: Optional[str] = Field(
        default="bandit_state.json", 
        description="Path to bandit state persistence file"
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save state after each update"
    )


class JudgeConfig(BaseModel):
    """Configuration for quality judge."""
    enabled: bool = Field(default=True, description="Enable quality judge for testing")
    provider: str = Field(default="openai", description="LLM provider for judge: openai, gemini")
    model: str = Field(default="gpt-4o", description="Model for quality evaluation")
    api_key: Optional[str] = Field(default=None, description="API key for judge provider")


class CascadingConfig(BaseModel):
    """Configuration for cascading inference."""
    enabled: bool = Field(default=True, description="Enable cascading inference (start with cheap model, escalate if needed)")
    quality_threshold: float = Field(default=0.85, description="Quality threshold for escalation (0.0-1.0)")
    cheap_model: str = Field(default="gpt-4o-mini", description="Cheap model to start with")
    premium_model: str = Field(default="gpt-4o", description="Premium model to escalate to")
    use_lightweight_check: bool = Field(default=True, description="Use lightweight quality check instead of full judge for speed")
    use_escalation_prediction: bool = Field(
        default=True,
        description="Use bandit to predict escalation likelihood and skip cheap model when premium is likely needed"
    )
    escalation_prediction_threshold: float = Field(
        default=0.7,
        description="Escalation probability threshold (0.0-1.0) to skip cheap model. Higher = more conservative (fewer skips)"
    )


class TokenomicsConfig(BaseModel):
    """Main configuration for Tokenomics platform."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    cascading: CascadingConfig = Field(default_factory=CascadingConfig)
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="tokenomics.log", description="Log file path")

    @classmethod
    def from_env(cls) -> "TokenomicsConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "gemini"),
                model=os.getenv("LLM_MODEL", "gemini-2.0-flash-exp"),
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY"),
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            ),
            memory=MemoryConfig(
                use_exact_cache=os.getenv("USE_EXACT_CACHE", "true").lower() == "true",
                use_semantic_cache=os.getenv("USE_SEMANTIC_CACHE", "true").lower() == "true",
                vector_store=os.getenv("VECTOR_STORE", "faiss"),
                enable_llmlingua=os.getenv("ENABLE_LLMLINGUA", "true").lower() == "true",
                use_active_retrieval=os.getenv("USE_ACTIVE_RETRIEVAL", "false").lower() == "true",
            ),
            judge=JudgeConfig(
                enabled=os.getenv("ENABLE_QUALITY_JUDGE", "true").lower() == "true",
                provider=os.getenv("JUDGE_PROVIDER", "openai"),
                model=os.getenv("JUDGE_MODEL", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            cascading=CascadingConfig(
                enabled=os.getenv("CASCADING_ENABLED", "true").lower() == "true",
                quality_threshold=float(os.getenv("CASCADING_QUALITY_THRESHOLD", "0.85")),
            ),
        )

