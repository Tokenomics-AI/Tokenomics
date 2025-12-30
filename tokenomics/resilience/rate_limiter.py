"""Rate limiter using token bucket algorithm."""

import time
from typing import Optional
from threading import Lock
import structlog

logger = structlog.get_logger()


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: float = 20.0,  # bucket capacity
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()
        
        logger.info(
            "RateLimiter initialized",
            rate=rate,
            capacity=capacity,
        )
    
    def _add_tokens(self):
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * self.rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now
    
    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if tokens were acquired, False if rate limit exceeded
        """
        with self.lock:
            self._add_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                logger.warning(
                    "Rate limit exceeded",
                    requested=tokens,
                    available=self.tokens,
                    rate=self.rate,
                )
                return False
    
    def wait_if_needed(self, tokens: float = 1.0) -> float:
        """
        Wait if needed to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Time waited in seconds
        """
        wait_time = 0.0
        
        with self.lock:
            self._add_tokens()
            
            if self.tokens < tokens:
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                logger.debug(
                    "Rate limiter wait",
                    wait_seconds=wait_time,
                    tokens_needed=tokens_needed,
                )
        
        if wait_time > 0:
            time.sleep(wait_time)
            # Update tokens after waiting
            with self.lock:
                self._add_tokens()
                self.tokens -= tokens
        
        return wait_time
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self.lock:
            self._add_tokens()
            return {
                "tokens_available": self.tokens,
                "capacity": self.capacity,
                "rate": self.rate,
                "utilization": (self.capacity - self.tokens) / self.capacity * 100 if self.capacity > 0 else 0,
            }






