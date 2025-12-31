"""Factory for creating TokenomicsPlatform with dependency injection."""

from typing import Optional, Dict, Any
from ..config import TokenomicsConfig
from ..memory.memory_layer import SmartMemoryLayer
from ..orchestrator.orchestrator import TokenAwareOrchestrator
from ..bandit.bandit import BanditOptimizer
from ..llm_providers import LLMProvider
from ..judge import QualityJudge
import structlog

logger = structlog.get_logger()


class PlatformFactory:
    """Factory for creating platform instances with injected dependencies."""
    
    @staticmethod
    def create(
        config: Optional[TokenomicsConfig] = None,
        memory_layer: Optional[SmartMemoryLayer] = None,
        orchestrator: Optional[TokenAwareOrchestrator] = None,
        bandit: Optional[BanditOptimizer] = None,
        llm_provider: Optional[LLMProvider] = None,
        quality_judge: Optional[QualityJudge] = None,
    ):
        """
        Create TokenomicsPlatform with optional dependency injection.
        
        If a dependency is not provided, it will be created from config.
        This allows for testing with mock dependencies.
        
        Args:
            config: Platform configuration
            memory_layer: Optional memory layer instance
            orchestrator: Optional orchestrator instance
            bandit: Optional bandit optimizer instance
            llm_provider: Optional LLM provider instance
            quality_judge: Optional quality judge instance
        
        Returns:
            TokenomicsPlatform instance
        """
        from ..core import TokenomicsPlatform
        
        # Create a custom platform class that accepts dependencies
        class CustomPlatform(TokenomicsPlatform):
            def __init__(self, *args, **kwargs):
                # Store injected dependencies
                self._injected_memory = memory_layer
                self._injected_orchestrator = orchestrator
                self._injected_bandit = bandit
                self._injected_llm_provider = llm_provider
                self._injected_quality_judge = quality_judge
                
                # Call parent init
                super().__init__(*args, **kwargs)
                
                # Override with injected dependencies
                if memory_layer:
                    self.memory = memory_layer
                if orchestrator:
                    self.orchestrator = orchestrator
                if bandit:
                    self.bandit = bandit
                if llm_provider:
                    self.llm_provider = llm_provider
                if quality_judge:
                    self.quality_judge = quality_judge
        
        return CustomPlatform(config=config)







