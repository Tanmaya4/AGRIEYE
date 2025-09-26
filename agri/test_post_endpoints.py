#!/usr/bin/env python3
"""
Test script for Agricultural API POST endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, name):
    """Test a POST endpoint and display results"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"📍 Endpoint: POST {endpoint}")
    print('='*60)
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS - Status: {response.status_code}")
            result = response.json()
            
            # Display key information
            print(f"📊 Analysis Type: {result.get('analysis_type', 'N/A')}")
            print(f"🕒 Timestamp: {result.get('timestamp', 'N/A')}")
            
            # Display summary if available
            if 'summary' in result:
                print("\n📋 SUMMARY:")
                summary = result['summary']
                for key, value in summary.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Display recommendations if available (for enhanced analysis)
            if 'recommendations' in result:
                print("\n🚨 RECOMMENDATIONS:")
                recs = result['recommendations']
                for category, actions in recs.items():
                    if actions:  # Only show if there are actions
                        print(f"  {category.upper()}:")
                        for action in actions:
                            print(f"    • {action}")
            
            print(f"\n✅ {name} completed successfully!")
            
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Is the API server running on localhost:5000?")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    print("🌱 Agricultural Analysis API - POST Endpoint Testing")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"🕒 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test the three main POST endpoints
    endpoints_to_test = [
        ("/analyze/real", "Real Data Analysis"),
        ("/analyze/hyperspectral", "Hyperspectral Analysis"), 
        ("/analyze/enhanced", "Enhanced Analysis (Complete)")
    ]
    
    for endpoint, name in endpoints_to_test:
        test_endpoint(endpoint, name)
        time.sleep(1)  # Small delay between tests
    
    print(f"\n{'='*60}")
    print("🎉 All POST endpoint tests completed!")
    print('='*60)

if __name__ == "__main__":
    main()