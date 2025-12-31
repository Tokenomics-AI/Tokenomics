"""Unified state management for Tokenomics platform."""

import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from threading import Lock
from datetime import datetime
import structlog

logger = structlog.get_logger()


class StateManager:
    """Centralized state management for platform components."""
    
    def __init__(self, state_file: Optional[str] = "tokenomics_state.json"):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file (None = in-memory only)
        """
        self.state_file = Path(state_file) if state_file else None
        self.lock = Lock()
        self.state: Dict[str, Any] = {
            "bandit": {},
            "cache": {},
            "metrics": {},
            "last_saved": None,
        }
        
        # Load existing state if file exists
        if self.state_file and self.state_file.exists():
            self.load()
        else:
            logger.info("StateManager initialized with empty state")
    
    def save(self):
        """Save state to file."""
        if not self.state_file:
            return
        
        with self.lock:
            self.state["last_saved"] = datetime.now().isoformat()
            
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)
                logger.debug("State saved", file=str(self.state_file))
            except Exception as e:
                logger.error("Failed to save state", error=str(e))
    
    def load(self):
        """Load state from file."""
        if not self.state_file or not self.state_file.exists():
            return
        
        with self.lock:
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info("State loaded", file=str(self.state_file))
            except Exception as e:
                logger.error("Failed to load state", error=str(e))
                self.state = {
                    "bandit": {},
                    "cache": {},
                    "metrics": {},
                    "last_saved": None,
                }
    
    def update_bandit_state(self, bandit_state: Dict[str, Any]):
        """Update bandit state."""
        with self.lock:
            self.state["bandit"] = bandit_state
            self.save()
    
    def update_cache_state(self, cache_state: Dict[str, Any]):
        """Update cache state."""
        with self.lock:
            self.state["cache"] = cache_state
            self.save()
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics."""
        with self.lock:
            self.state["metrics"] = metrics
            self.save()
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state snapshot."""
        with self.lock:
            return self.state.copy()
    
    def restore_state(self, state: Dict[str, Any]):
        """Restore state from snapshot."""
        with self.lock:
            self.state = state
            self.save()
        logger.info("State restored from snapshot")







