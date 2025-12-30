"""Quick test script to verify Gemini API key works."""

import os
import sys

# Set API key directly (replace with your actual key from Google Cloud Console)
API_KEY = "AIzaSyCvSI80PtKuVejnkIiiNxjjN6PyRRngB1E"  # Update this with the key from the console

# Try with corrected format (if it starts with Alza, try AIza)
if API_KEY.startswith("Alza"):
    print("WARNING: Key starts with 'Alza' - Google keys usually start with 'AIza'")
    print("   Please verify you copied the complete key from Google Cloud Console")
    print()

try:
    import google.generativeai as genai
    
    print(f"Testing API key: {API_KEY[:10]}...")
    print()
    
    # Configure
    genai.configure(api_key=API_KEY)
    
    # Try to list models (this will fail if key is invalid)
    print("Testing API connection...")
    models = list(genai.list_models())
    
    print("SUCCESS: API key is valid!")
    print(f"SUCCESS: Found {len(models)} available models")
    print()
    print("You can now run:")
    print("  python examples/basic_usage.py")
    print("  python examples/benchmark.py")
    
except Exception as e:
    print("ERROR: API key test failed!")
    print(f"Error: {e}")
    print()
    print("Please:")
    print("1. Go to: https://console.cloud.google.com/apis/credentials")
    print("2. Find your 'Tokenomics' API key")
    print("3. Click 'Copy key' button")
    print("4. Update the API_KEY variable in this script")
    print("5. Make sure the key starts with 'AIza' (capital I)")
    print("6. Ensure the Generative Language API is enabled")

