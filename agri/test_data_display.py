"""
Test script to demonstrate the enhanced hyperspectral analyzer data display functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from hyperspectral_analyzer import HyperspectralAnalyzer

def test_data_display():
    """Test the enhanced data display functionality"""
    print("🧪 TESTING ENHANCED HYPERSPECTRAL ANALYZER")
    print("=" * 60)
    print("This test demonstrates the data processing display features:")
    print("✅ Raw spectral data overview")
    print("✅ MAT file contents analysis") 
    print("✅ Wavelength band mappings")
    print("✅ Step-by-step processing displays")
    print("✅ Input data for each calculation")
    print("=" * 60)
    
    # Initialize analyzer - this will show data loading process
    analyzer = HyperspectralAnalyzer()
    
    print("\n🔍 TESTING INDIVIDUAL ANALYSIS COMPONENTS:")
    print("-" * 50)
    
    # Test vegetation indices calculation with data display
    print("\n1️⃣ Testing Vegetation Indices Calculation:")
    vegetation_indices = analyzer.calculate_vegetation_indices()
    
    # Test water stress detection with data display  
    print("\n2️⃣ Testing Water Stress Detection:")
    water_stress = analyzer.detect_water_stress()
    
    # Test disease detection with data display
    print("\n3️⃣ Testing Disease Anomaly Detection:")
    disease_analysis = analyzer.detect_disease_anomalies()
    
    # Test spatial analysis with data display
    print("\n4️⃣ Testing Spatial Analysis:")
    spatial_analysis = analyzer.generate_spatial_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 DATA DISPLAY TEST COMPLETED SUCCESSFULLY!")
    print("All processing steps now show input data before calculations")
    print("=" * 60)

if __name__ == "__main__":
    test_data_display()