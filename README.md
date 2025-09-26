# Agricultural Analysis System

## Project Structure

### Core Analysis Files
- `real_agricultural_analysis.py` - Your working real data analysis system
- `hyperspectral_analyzer.py` - Advanced hyperspectral image analysis
- `enhanced_agricultural_system.py` - Integrated analysis system
- 
### Front End
- https://github.com/xanderbilla/sih-hack-the-box

### Data Files  
- `aviris150.mat` - Hyperspectral image data (150x150x186 bands)
- `weather_subset.csv` - Weather data (100 records)
- `soil_subset.csv` - Soil pH data (100 records) 
- `sensor_data.csv` - Sensor readings (20 records)
- `soil.csv` - Additional soil data

### Environment
- `agri_env/` - Python virtual environment
- `requirements.txt` - Dependencies

### Reports
- `enhanced_agricultural_report_20250926_013626.json` - Latest comprehensive analysis

## Usage

### Run Individual Analysis
```bash
# Real data analysis only
python real_agricultural_analysis.py

# Hyperspectral analysis only  
python hyperspectral_analyzer.py

# Complete integrated analysis (recommended)
python enhanced_agricultural_system.py
```

## Analysis Capabilities

### Real Data Analysis
- Crop stress assessment from sensor/weather data
- Soil condition analysis from pH and nutrient data
- Weather pattern analysis for disease risk
- Cross-validation with multiple data sources

### Hyperspectral Analysis
- 6 vegetation indices (NDVI, EVI, SAVI, GNDVI, NDRE, CI_Green)
- Water stress detection using absorption bands
- Disease anomaly detection through spectral signatures
- Spatial analysis across 22,500 pixels (150x150)
- Wavelength range: 400-2500nm with 186 spectral bands

### Integrated System
- Cross-validates findings between sensor and spectral data
- Provides actionable recommendations by priority
- Generates unified comprehensive reports
- Multi-modal data fusion approach

## Key Results from Your Data
- **Critical crop stress detected (90/100)**
- **99.7% of field area is stressed** 
- **Severe water deficit confirmed by both methods**
- **Critical nutrient deficiency (N: 14.3, P: 8.7)**
- **Emergency irrigation recommended**

## Dependencies
All required packages are in the virtual environment (`agri_env/`).
