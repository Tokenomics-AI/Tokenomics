"""Test the Flask API status endpoint."""

import requests
import time
import sys

def test_api():
    print("Testing Flask API status endpoint...")
    print("-" * 60)
    
    try:
        # Wait a moment in case server is starting
        time.sleep(1)
        
        response = requests.get('http://localhost:5000/api/status', timeout=5)
        print(f"Status Code: {response.status_code}")
        
        data = response.json()
        print(f"Status: {data.get('status', 'unknown')}")
        
        if data.get('status') == 'ok':
            print("[OK] API is working correctly!")
            print(f"  Platform stats: {data.get('platform_stats', {})}")
        elif data.get('status') == 'error':
            print(f"[ERROR] API returned error: {data.get('message', 'Unknown error')}")
            print(f"  Error type: {data.get('error_type', 'unknown')}")
        else:
            print(f"Response: {data}")
            
    except requests.exceptions.ConnectionError:
        print("[INFO] Flask app is not running")
        print("  Start it with: python app.py")
        print("  Then access: http://localhost:5000/dashboard")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_api()

