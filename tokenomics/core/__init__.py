"""Core Tokenomics platform components."""

# Import from parent directory's core.py file directly
# The actual TokenomicsPlatform is in ../core.py, not in this directory
import sys
import importlib.util
from pathlib import Path

# Get the parent directory (tokenomics/)
parent_dir = Path(__file__).parent.parent

# Load core.py as a module with proper package context
core_file = parent_dir / "core.py"
spec = importlib.util.spec_from_file_location("tokenomics.core_module", core_file, submodule_search_locations=[str(parent_dir)])
core_module = importlib.util.module_from_spec(spec)

# Set __package__ to allow relative imports
core_module.__package__ = "tokenomics"
core_module.__name__ = "tokenomics.core_module"

# Execute the module
spec.loader.exec_module(core_module)

# Extract TokenomicsPlatform
TokenomicsPlatform = core_module.TokenomicsPlatform

from .platform_factory import PlatformFactory

__all__ = ["TokenomicsPlatform", "PlatformFactory"]







