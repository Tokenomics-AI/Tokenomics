"""Complete test of dashboard functionality."""

import requests
import json
import time

def test_dashboard():
    print("=" * 60)
    print("Dashboard Complete Test")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Check if server is running
    print("\n[1] Testing server status...")
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Server is running")
            print(f"  Status: {data.get('status', 'unknown')}")
            if data.get('status') == 'ok':
                print(f"  Platform stats: {data.get('platform_stats', {})}")
            elif data.get('status') == 'error':
                print(f"  [ERROR] {data.get('message', 'Unknown error')}")
        else:
            print(f"  [ERROR] Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] Cannot connect to server: {e}")
        print("  Make sure Flask app is running: python app.py")
        return False
    
    # Test 2: Test dashboard page loads
    print("\n[2] Testing dashboard page...")
    try:
        response = requests.get(f"{base_url}/dashboard", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "query-mode-group" in content:
                print("  [OK] Dashboard page loads with new controls")
            else:
                print("  [WARNING] Dashboard page loads but new controls not found")
            if "api-status-indicator" in content:
                print("  [OK] API status indicator found")
        else:
            print(f"  [ERROR] Status code: {response.status_code}")
    except Exception as e:
        print(f"  [ERROR] Cannot load dashboard: {e}")
    
    # Test 3: Test API run endpoint with same query mode
    print("\n[3] Testing API run endpoint (same query mode)...")
    try:
        test_data = {
            'queries': ['What is machine learning?'],
            'mode': 'tokenomics',
            'num_queries': 1
        }
        response = requests.post(f"{base_url}/api/run", json=test_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("  [OK] API run endpoint works")
            print(f"  Run ID: {data.get('id', 'N/A')}")
            print(f"  Results: {len(data.get('results', []))} query(s)")
            if data.get('results'):
                result = data['results'][0]
                print(f"  Tokens used: {result.get('tokens_used', 0)}")
                print(f"  Cache hit: {result.get('cache_hit', False)}")
        else:
            print(f"  [ERROR] Status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] API run failed: {e}")
    
    # Test 4: Test with multiple queries (similar mode simulation)
    print("\n[4] Testing API run endpoint (multiple similar queries)...")
    try:
        test_data = {
            'queries': [
                'What is machine learning?',
                'Explain machine learning to me',
                'Tell me about machine learning'
            ],
            'mode': 'tokenomics',
            'num_queries': 3
        }
        response = requests.post(f"{base_url}/api/run", json=test_data, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print("  [OK] Multiple queries processed")
            print(f"  Results: {len(data.get('results', []))} queries")
            cache_hits = sum(1 for r in data.get('results', []) if r.get('cache_hit'))
            print(f"  Cache hits: {cache_hits}/{len(data.get('results', []))}")
            total_tokens = sum(r.get('tokens_used', 0) for r in data.get('results', []))
            print(f"  Total tokens: {total_tokens}")
        else:
            print(f"  [ERROR] Status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] Multiple queries test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nIf all tests passed, the backend is working correctly.")
    print("If the dashboard UI is not working, check browser console for JavaScript errors.")
    print("\nTo test the UI:")
    print("  1. Open http://localhost:5000/dashboard in your browser")
    print("  2. Open browser Developer Tools (F12)")
    print("  3. Check Console tab for any JavaScript errors")
    print("  4. Try running an experiment and check Network tab for API calls")

if __name__ == "__main__":
    test_dashboard()

