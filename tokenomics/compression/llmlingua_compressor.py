"""LLMLingua-2 compression wrapper for context and query compression."""

from typing import Optional, List, Dict, Any
import structlog

logger = structlog.get_logger()


class LLMLinguaCompressor:
    """
    Wrapper for LLMLingua-2 prompt compression.
    
    LLMLingua-2 uses a small encoder model to identify and remove
    less important tokens while preserving semantic meaning.
    
    Features:
    - Context compression for reducing context window usage
    - Query compression for long queries
    - Configurable compression ratios
    - Graceful fallback when unavailable
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        compression_ratio: float = 0.4,
        device: str = "cpu",
    ):
        """
        Initialize LLMLingua-2 compressor.
        
        Args:
            model_name: HuggingFace model for compression
            compression_ratio: Target compression ratio (0.4 = keep 40% of tokens)
            device: Device to run on (cpu/cuda)
        """
        self.model_name = model_name
        self.compression_ratio = compression_ratio
        self.device = device
        self._compressor = None
        self._initialized = False
        self._init_error: Optional[str] = None
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize LLMLingua-2 compressor."""
        try:
            # Set environment variable to suppress TensorFlow warnings
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
            
            # Import LLMLingua
            try:
                from llmlingua import PromptCompressor
            except ImportError:
                try:
                    from llmlingua2 import PromptCompressor
                except ImportError:
                    raise ImportError(
                        "llmlingua package not installed. Install with: pip install llmlingua"
                    )
            
            # Initialize with appropriate parameters
            # Use device_map="cpu" for compatibility
            init_kwargs = {
                "model_name": self.model_name,
                "use_llmlingua2": True,
                "device_map": "cpu",
            }
            
            # Try initialization with device_map
            try:
                self._compressor = PromptCompressor(**init_kwargs)
            except (TypeError, ValueError, Exception) as e1:
                # Fallback: Try without device_map
                try:
                    init_kwargs.pop("device_map", None)
                    self._compressor = PromptCompressor(**init_kwargs)
                except Exception as e2:
                    # Final fallback: Just model_name with use_llmlingua2
                    try:
                        self._compressor = PromptCompressor(
                            model_name=self.model_name,
                            use_llmlingua2=True,
                        )
                    except Exception as e3:
                        # Last resort: default initialization
                        self._compressor = PromptCompressor()
            
            self._initialized = True
            logger.info(
                "LLMLingua-2 initialized successfully",
                model=self.model_name,
                compression_ratio=self.compression_ratio,
            )
        except ImportError as e:
            self._init_error = f"llmlingua package not installed: {str(e)}"
            logger.warning(
                "LLMLingua-2 not available - package not installed",
                error=self._init_error,
            )
        except Exception as e:
            self._init_error = f"Failed to initialize LLMLingua-2: {str(e)}"
            logger.warning(
                "Failed to initialize LLMLingua-2",
                error=self._init_error,
                model=self.model_name,
            )
    
    def is_available(self) -> bool:
        """Check if LLMLingua-2 is available and initialized."""
        return self._initialized and self._compressor is not None
    
    def get_init_error(self) -> Optional[str]:
        """Get initialization error if any."""
        return self._init_error
    
    def compress_context(
        self,
        contexts: List[str],
        target_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compress context strings using LLMLingua-2.
        
        Args:
            contexts: List of context strings to compress
            target_ratio: Target compression ratio (overrides default)
            
        Returns:
            Dictionary with:
                - compressed_text: The compressed context
                - original_tokens: Original token count
                - compressed_tokens: Compressed token count
                - compression_ratio: Actual compression ratio achieved
                - success: Whether compression was successful
        """
        if not self.is_available():
            logger.warning("LLMLingua-2 not initialized, returning original context")
            combined = " ".join(contexts)
            return {
                "compressed_text": combined,
                "original_tokens": len(combined.split()),
                "compressed_tokens": len(combined.split()),
                "compression_ratio": 1.0,
                "success": False,
                "error": self._init_error or "Not initialized",
            }
        
        if not contexts:
            return {
                "compressed_text": "",
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 1.0,
                "success": True,
            }
        
        ratio = target_ratio or self.compression_ratio
        combined_context = "\n\n".join(contexts)
        
        try:
            # Call LLMLingua-2 compression
            result = self._compressor.compress_prompt(
                context=[combined_context],
                rate=ratio,
                force_tokens=["\n", ".", "!", "?", ","],  # Preserve punctuation
                drop_consecutive=True,
            )
            
            compressed_text = result.get("compressed_prompt", combined_context)
            original_tokens = result.get("origin_tokens", len(combined_context.split()))
            compressed_tokens = result.get("compressed_tokens", len(compressed_text.split()))
            
            actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            logger.debug(
                "Context compressed with LLMLingua-2",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                ratio=f"{actual_ratio:.2%}",
            )
            
            return {
                "compressed_text": compressed_text,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": actual_ratio,
                "success": True,
            }
            
        except Exception as e:
            logger.warning(
                "LLMLingua-2 compression failed",
                error=str(e),
            )
            combined = " ".join(contexts)
            return {
                "compressed_text": combined,
                "original_tokens": len(combined.split()),
                "compressed_tokens": len(combined.split()),
                "compression_ratio": 1.0,
                "success": False,
                "error": str(e),
            }
    
    def compress_query(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Compress a query string using LLMLingua-2.
        
        Args:
            query: Query string to compress
            max_tokens: Maximum tokens in output
            max_chars: Maximum characters in output
            
        Returns:
            Compressed query string
        """
        if not self.is_available():
            logger.warning("LLMLingua-2 not initialized, returning original query")
            return query
        
        if not query:
            return query
        
        try:
            # Calculate target ratio based on max_tokens/max_chars
            query_tokens = len(query.split())
            query_chars = len(query)
            
            target_ratio = self.compression_ratio
            if max_tokens and query_tokens > max_tokens:
                target_ratio = min(target_ratio, max_tokens / query_tokens)
            if max_chars and query_chars > max_chars:
                char_ratio = max_chars / query_chars
                target_ratio = min(target_ratio, char_ratio)
            
            # Compress with calculated ratio
            result = self._compressor.compress_prompt(
                context=[query],
                rate=target_ratio,
                force_tokens=["\n", ".", "!", "?"],
            )
            
            compressed = result.get("compressed_prompt", query)
            
            logger.debug(
                "Query compressed with LLMLingua-2",
                original_tokens=len(query.split()),
                compressed_tokens=len(compressed.split()),
            )
            
            return compressed
            
        except Exception as e:
            logger.warning(
                "LLMLingua-2 query compression failed",
                error=str(e),
            )
            return query
    
    def compress_prompt(
        self,
        instruction: str,
        context: Optional[str] = None,
        question: Optional[str] = None,
        target_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compress a full prompt with instruction, context, and question.
        
        Args:
            instruction: System instruction
            context: Context/background information
            question: User question
            target_ratio: Target compression ratio
            
        Returns:
            Dictionary with compressed prompt and metrics
        """
        if not self.is_available():
            # Build uncompressed prompt
            parts = [instruction]
            if context:
                parts.append(context)
            if question:
                parts.append(question)
            combined = "\n\n".join(parts)
            
            return {
                "compressed_prompt": combined,
                "original_tokens": len(combined.split()),
                "compressed_tokens": len(combined.split()),
                "compression_ratio": 1.0,
                "success": False,
                "error": self._init_error or "Not initialized",
            }
        
        ratio = target_ratio or self.compression_ratio
        
        try:
            # Build prompt components
            result = self._compressor.compress_prompt(
                instruction=instruction,
                context=[context] if context else None,
                question=question,
                rate=ratio,
                force_tokens=["\n", ".", "!", "?", ","],
            )
            
            compressed_prompt = result.get("compressed_prompt", "")
            original_tokens = result.get("origin_tokens", 0)
            compressed_tokens = result.get("compressed_tokens", 0)
            
            actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            return {
                "compressed_prompt": compressed_prompt,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": actual_ratio,
                "success": True,
            }
            
        except Exception as e:
            # Build uncompressed prompt as fallback
            parts = [instruction]
            if context:
                parts.append(context)
            if question:
                parts.append(question)
            combined = "\n\n".join(parts)
            
            return {
                "compressed_prompt": combined,
                "original_tokens": len(combined.split()),
                "compressed_tokens": len(combined.split()),
                "compression_ratio": 1.0,
                "success": False,
                "error": str(e),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compressor statistics."""
        return {
            "initialized": self._initialized,
            "model": self.model_name,
            "compression_ratio": self.compression_ratio,
            "device": self.device,
            "error": self._init_error,
        }





