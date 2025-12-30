# API Key Configuration Note

## Current Status
The API key in your `.env` file starts with "Alza" but Google API keys typically start with "AIza" (capital I, not lowercase L).

## To Fix

1. **Copy the complete API key** from Google Cloud Console:
   - Go to: https://console.cloud.google.com/apis/credentials
   - Find your "Tokenomics" API key
   - Click "Copy key" button
   - The key should start with "AIza" and be 39 characters long

2. **Update the .env file** with the correct key:
   ```env
   GEMINI_API_KEY=AIza...your-complete-key-here...
   ```

3. **Or update it directly in examples/basic_usage.py**:
   ```python
   os.environ["GEMINI_API_KEY"] = "AIza...your-complete-key-here..."
   ```

## Verify Key Format
- ✅ Should start with: `AIza` (capital I)
- ✅ Should be 39 characters long
- ✅ Should contain alphanumeric characters

Once you have the correct key, the platform will work!

