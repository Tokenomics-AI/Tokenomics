#!/bin/bash
# WSL Setup Script for Tokenomics with LLMLingua

set -e

echo "=========================================="
echo "Setting up Tokenomics in WSL Ubuntu"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -qq

# Install pip if not available
echo "Installing pip and venv..."
sudo apt-get install -y python3-pip python3-venv python3-dev -qq

# Create virtual environment
echo "Creating virtual environment..."
cd /mnt/d/Nayan/Tokenomics/Prototype

# Remove old venv if exists
rm -rf .venv-wsl

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install numpy (compatible with Python 3.12)
echo "Installing numpy..."
pip install "numpy>=1.26.0,<2.0"

# Install scipy
echo "Installing scipy..."
pip install "scipy>=1.11.0"

# Install PyTorch CPU
echo "Installing PyTorch (this may take a few minutes)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers and accelerate
echo "Installing transformers and accelerate..."
pip install transformers accelerate huggingface-hub

# Install LLMLingua
echo "Installing LLMLingua..."
pip install llmlingua

# Install sentence-transformers
echo "Installing sentence-transformers..."
pip install sentence-transformers

# Install other requirements
echo "Installing remaining dependencies..."
pip install faiss-cpu tiktoken openai google-generativeai
pip install pydantic python-dotenv flask structlog tenacity requests cachetools scikit-learn

echo ""
echo "=========================================="
echo "Verifying LLMLingua..."
echo "=========================================="

python3 << 'EOF'
print("Testing LLMLingua initialization...")
try:
    from llmlingua import PromptCompressor
    print("Importing PromptCompressor...")
    c = PromptCompressor(
        model_name='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank',
        use_llmlingua2=True,
        device_map='cpu'
    )
    print("SUCCESS: LLMLingua initialized!")
    
    # Test compression
    result = c.compress_prompt(
        context=['This is a test context for compression verification. The quick brown fox jumps over the lazy dog. Machine learning is transforming industries.'],
        rate=0.5
    )
    compressed = result.get("compressed_prompt", "")
    print(f"Compression test: Original ~150 chars -> {len(compressed)} chars")
    print("LLMLingua is WORKING!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the diagnostic test:"
echo "  wsl -d Ubuntu"
echo "  cd /mnt/d/Nayan/Tokenomics/Prototype"
echo "  source .venv-wsl/bin/activate"
echo "  python test_comprehensive_showcase.py"
