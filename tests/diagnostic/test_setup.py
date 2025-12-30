# test_setup.py
import os
import sys
from pathlib import Path

# Add project root to path (go up two levels from tests/diagnostic/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(project_root / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

print("Testing Tokenomics Platform Setup...")
print("=" * 50)

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your-openai-api-key":
    print("⚠ WARNING: OPENAI_API_KEY not set in .env file")
    print("  Please edit .env and add your actual OpenAI API key")
else:
    print(f"✓ API key found (length: {len(api_key)})")

# Initialize config
try:
    config = TokenomicsConfig.from_env()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o-mini"
    config.llm.api_key = api_key if api_key and api_key != "your-openai-api-key" else "sk-test"
    print("✓ Configuration loaded")
except Exception as e:
    print(f"❌ Config error: {e}")
    exit(1)

# Initialize platform
try:
    platform = TokenomicsPlatform(config=config)
    print("✓ Platform initialized")
except Exception as e:
    print(f"❌ Platform init error: {e}")
    exit(1)

# Test a simple query (only if API key is valid)
if api_key and api_key != "your-openai-api-key" and api_key.startswith("sk-"):
    print("\nTesting with a simple query...")
    try:
        result = platform.query("What is Python?")
        print(f"✓ Query successful")
        print(f"  Response length: {len(result.get('response', ''))}")
        print(f"  Tokens used: {result.get('tokens_used', 'N/A')}")
    except Exception as e:
        print(f"⚠ Query test failed: {e}")
        print("  (This is okay if API key is invalid or quota exceeded)")

print("\n" + "=" * 50)
print("Setup verification complete!")


