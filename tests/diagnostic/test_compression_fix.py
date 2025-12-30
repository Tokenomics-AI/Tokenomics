"""Quick test to verify query compression is working."""

import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig

print("=" * 80)
print("TESTING QUERY COMPRESSION FIX")
print("=" * 80)

# Initialize platform
config = TokenomicsConfig.from_env()
config.llm.provider = "openai"
config.llm.model = "gpt-4o-mini"
config.llm.api_key = os.getenv("OPENAI_API_KEY")

print(f"\nCompression Thresholds:")
print(f"  Tokens: {config.memory.compress_query_threshold_tokens}")
print(f"  Chars: {config.memory.compress_query_threshold_chars}")

platform = TokenomicsPlatform(config=config)

# Test with a long query that should definitely trigger compression
long_query = """I need a comprehensive and detailed explanation of how modern machine learning algorithms work in production systems. Please cover the following topics in depth: 1) Neural network architectures including CNNs, RNNs, LSTMs, and Transformers, 2) The backpropagation algorithm and how gradients flow through the network, 3) Optimization techniques like SGD, Adam, and learning rate scheduling, 4) Regularization methods including dropout, batch normalization, and L2 regularization, 5) How these models are deployed in production with considerations for latency and throughput, 6) Best practices for model monitoring and retraining in production environments. Please provide specific examples and code snippets where appropriate."""

print(f"\nTest Query:")
print(f"  Length: {len(long_query)} characters")
print(f"  Tokens (estimated): ~{len(long_query) // 4}")

# Check thresholds
token_threshold = config.memory.compress_query_threshold_tokens
char_threshold = config.memory.compress_query_threshold_chars

print(f"\nThreshold Check:")
print(f"  Exceeds char threshold ({char_threshold}): {len(long_query) > char_threshold}")
print(f"  Should compress: {len(long_query) > char_threshold or (len(long_query) // 4) > token_threshold}")

# Test compression directly
print(f"\nTesting compression directly...")
compressed = platform.memory.compress_query_if_needed(long_query)
print(f"  Original length: {len(long_query)}")
print(f"  Compressed length: {len(compressed)}")
print(f"  Compression occurred: {compressed != long_query}")

# Now test through full query
print(f"\nTesting through full platform query...")
result = platform.query(
    query=long_query,
    use_cache=False,  # Disable cache to see compression
    use_bandit=False,  # Disable bandit for simpler test
    use_compression=True,
)

compression_metrics = result.get("compression_metrics", {})
print(f"\nCompression Metrics:")
print(f"  Query compressed: {compression_metrics.get('query_compressed', False)}")
print(f"  Original tokens: {compression_metrics.get('query_original_tokens', 0)}")
print(f"  Compressed tokens: {compression_metrics.get('query_compressed_tokens', 0)}")
print(f"  Compression ratio: {compression_metrics.get('query_compression_ratio', 1.0):.2%}")

if compression_metrics.get('query_compressed', False):
    print("\n✅ COMPRESSION WORKING!")
else:
    print("\n❌ Compression not triggered")
    print(f"   Query length: {len(long_query)} chars")
    print(f"   Thresholds: {token_threshold} tokens, {char_threshold} chars")

print("\n" + "=" * 80)

