"""
API Examples - Agricultural Analysis Endpoints
Demonstrates how to call all analysis endpoints
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_api_endpoint(endpoint_name, method, url, description):
    """Test an API endpoint and display results"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {endpoint_name}")
    print(f"📋 Description: {description}")
    print(f"🌐 URL: {method} {url}")
    print('='*60)
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS (Status: {response.status_code})")
            
            # Display key information from response
            if 'status' in result:
                print(f"📊 API Status: {result['status']}")
            
            if 'summary' in result:
                print(f"\n📈 KEY METRICS:")
                summary = result['summary']
                for key, value in summary.items():
                    print(f"  - {key.replace('_', ' ').title()}: {value}")
            
            if 'recommendations' in result and result['recommendations']:
                print(f"\n🚨 CRITICAL ACTIONS:")
                for action in result['recommendations'].get('critical', []):
                    print(f"  • {action}")
                    
            # Show data source info if available
            if 'data_sources' in result:
                print(f"\n📁 DATA SOURCES:")
                for source, info in result['data_sources'].items():
                    print(f"  - {source.title()}: {info}")
                    
        else:
            print(f"❌ ERROR (Status: {response.status_code})")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Make sure the API server is running!")
        print("Start server with: python agricultural_api.py")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    """Run all API endpoint examples"""
    print("🌱 AGRICULTURAL ANALYSIS API - ENDPOINT EXAMPLES")
    print("=" * 80)
    
    # 1. Health Check
    test_api_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/health",
        "Check API health and data file availability"
    )
    
    time.sleep(1)
    
    # 2. Data Info
    test_api_endpoint(
        "Data Information",
        "GET", 
        f"{BASE_URL}/data/info",
        "Get information about available data sources"
    )
    
    time.sleep(1)
    
    # 3. Real Data Analysis
    test_api_endpoint(
        "Real Data Analysis",
        "POST",
        f"{BASE_URL}/analyze/real",
        "Analyze weather, soil, and sensor data for crop stress and soil conditions"
    )
    
    time.sleep(2)
    
    # 4. Hyperspectral Analysis  
    test_api_endpoint(
        "Hyperspectral Analysis",
        "POST",
        f"{BASE_URL}/analyze/hyperspectral", 
        "Analyze AVIRIS hyperspectral imagery for vegetation indices and crop health"
    )
    
    time.sleep(2)
    
    # 5. Enhanced Analysis (Recommended)
    test_api_endpoint(
        "Enhanced Analysis (RECOMMENDED)",
        "POST",
        f"{BASE_URL}/analyze/enhanced",
        "Complete analysis: Real data + Hyperspectral + Cross-validation + Recommendations"
    )
    
    time.sleep(1)
    
    # 6. Reports List
    test_api_endpoint(
        "Reports List",
        "GET",
        f"{BASE_URL}/reports",
        "List all available analysis reports"
    )
    
    print(f"\n{'='*80}")
    print("✅ ALL API EXAMPLES COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()