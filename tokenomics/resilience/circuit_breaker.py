"""Circuit breaker pattern for LLM API calls."""

import time
from enum import Enum
from typing import Optional, Callable, Any
from threading import Lock
import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery (half-open)
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()
        
        logger.info(
            "CircuitBreaker initialized",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            RuntimeError: If circuit is open
            Exception: Original exception from function
        """
        with self.lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise RuntimeError(
                        f"Circuit breaker is OPEN. "
                        f"Last failure: {self.last_failure_time}. "
                        f"Wait {self.recovery_timeout}s before retry."
                    )
        
        # Try to execute function
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("Circuit breaker recovered, closing circuit")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                elif self.failure_count > 0:
                    # Reset on success
                    self.failure_count = 0
            
            return result
        
        except self.expected_exception as e:
            # Failure - increment count
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        "Circuit breaker opened",
                        failure_count=self.failure_count,
                        threshold=self.failure_threshold,
                    )
                else:
                    logger.warning(
                        "Circuit breaker failure",
                        failure_count=self.failure_count,
                        threshold=self.failure_threshold,
                    )
            
            raise
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker manually reset")







