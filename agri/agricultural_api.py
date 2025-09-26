"""
Agricultural Analysis Flask API
Provides REST endpoints for real agricultural data analysis and hyperspectral analysis
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
import traceback

# Import your analysis modules
from real_agricultural_analysis import RealAgriculturalAnalysis
from hyperspectral_analyzer import HyperspectralAnalyzer
from enhanced_agricultural_system import EnhancedAgriculturalSystem

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with available routes"""
    return jsonify({
        'message': 'Agricultural Analysis API',
        'version': '1.0.0',
        'description': 'REST API for real agricultural data analysis and hyperspectral analysis',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /analyze/real': 'Real agricultural data analysis',
            'POST /analyze/hyperspectral': 'Hyperspectral analysis only',
            'POST /analyze/hyperspectral/comprehensive': 'Comprehensive hyperspectral analysis with all outputs',
            'POST /analyze/enhanced': 'Complete enhanced analysis (real + hyperspectral)',
            'GET /reports': 'List available reports',
            'GET /reports/<filename>': 'Download specific report',
            'GET /data/info': 'Data source information'
        },
        'data_sources': {
            'weather': 'weather_subset.csv',
            'soil': 'soil_subset.csv', 
            'sensor': 'sensor_data.csv',
            'hyperspectral': 'aviris150.mat'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if data files exist
        data_files = {
            'weather_subset.csv': os.path.exists('weather_subset.csv'),
            'soil_subset.csv': os.path.exists('soil_subset.csv'),
            'sensor_data.csv': os.path.exists('sensor_data.csv'),
            'aviris150.mat': os.path.exists('aviris150.mat')
        }
        
        # Test basic imports
        test_real = RealAgriculturalAnalysis()
        test_hyperspectral = HyperspectralAnalyzer()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'data_files': data_files,
            'modules_loaded': True,
            'missing_files': [f for f, exists in data_files.items() if not exists]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/analyze/real', methods=['POST'])
def analyze_real_data():
    """
    Real agricultural data analysis endpoint
    Analyzes weather, soil, and sensor data
    """
    try:
        print("Starting real agricultural data analysis...")
        
        # Initialize analyzer
        analyzer = RealAgriculturalAnalysis()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        response = {
            'status': 'success',
            'analysis_type': 'real_data_analysis',
            'timestamp': datetime.now().isoformat(),
            'data': report,
            'summary': {
                'stress_level': report.get('crop_stress', {}).get('stress_level', 'Unknown'),
                'stress_index': report.get('crop_stress', {}).get('stress_index', 0),
                'soil_grade': report.get('soil_condition', {}).get('soil_grade', 'Unknown'),
                'fertility_score': report.get('soil_condition', {}).get('fertility_score', 0)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'analysis_type': 'real_data_analysis',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/analyze/hyperspectral', methods=['POST'])
def analyze_hyperspectral():
    """
    Hyperspectral analysis endpoint
    Analyzes AVIRIS hyperspectral imagery
    """
    try:
        print("Starting hyperspectral analysis...")
        
        # Initialize hyperspectral analyzer
        analyzer = HyperspectralAnalyzer()
        
        # Generate comprehensive hyperspectral analysis
        report = analyzer.comprehensive_hyperspectral_analysis()
        
        response = {
            'status': 'success',
            'analysis_type': 'hyperspectral_analysis',
            'timestamp': datetime.now().isoformat(),
            'data': report,
            'summary': {
                'crop_health_status': report.get('crop_health', {}).get('overall_status', 'Unknown'),
                'health_score': report.get('crop_health', {}).get('overall_score', 0),
                'ndvi': report.get('vegetation_indices', {}).get('ndvi', 0),
                'water_stress_level': report.get('water_stress', {}).get('stress_level', 'Unknown'),
                'disease_risk': report.get('disease_analysis', {}).get('risk_level', 'Unknown')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'analysis_type': 'hyperspectral_analysis',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/analyze/hyperspectral/comprehensive', methods=['GET', 'POST'])
def analyze_hyperspectral_comprehensive():
    """
    Comprehensive hyperspectral analysis endpoint
    Combines all hyperspectral analysis outputs including:
    - Raw hyperspectral data info
    - Vegetation indices calculations
    - Crop health assessment
    - Water stress analysis  
    - Disease anomaly detection
    - Spatial analysis
    - Spectral signatures
    - Band-wise analysis
    - Statistical summaries
    """
    try:
        print("Starting comprehensive hyperspectral analysis...")
        
        # Initialize hyperspectral analyzer
        analyzer = HyperspectralAnalyzer()
        
        # Get all hyperspectral analysis components
        full_analysis = analyzer.comprehensive_hyperspectral_analysis()
        
        # Additional detailed analysis
        spectral_data = analyzer.spectral_data
        wavelengths = analyzer.wavelengths
        
        # Calculate additional spectral statistics
        spectral_stats = {
            'data_dimensions': {
                'height': spectral_data.shape[0],
                'width': spectral_data.shape[1], 
                'bands': spectral_data.shape[2],
                'total_pixels': spectral_data.shape[0] * spectral_data.shape[1],
                'total_data_points': spectral_data.size
            },
            'wavelength_info': {
                'min_wavelength': float(wavelengths[0]) if wavelengths is not None else 400.0,
                'max_wavelength': float(wavelengths[-1]) if wavelengths is not None else 2500.0,
                'spectral_resolution': float(wavelengths[1] - wavelengths[0]) if wavelengths is not None and len(wavelengths) > 1 else 11.29,
                'wavelength_bands': len(wavelengths) if wavelengths is not None else 186
            },
            'reflectance_statistics': {
                'min_reflectance': float(np.min(spectral_data)),
                'max_reflectance': float(np.max(spectral_data)),
                'mean_reflectance': float(np.mean(spectral_data)),
                'std_reflectance': float(np.std(spectral_data)),
                'median_reflectance': float(np.median(spectral_data))
            }
        }
        
        # Sample spectral signatures from different regions
        height, width, bands = spectral_data.shape
        sample_signatures = {
            'center_pixel': spectral_data[height//2, width//2, :].tolist(),
            'corner_pixels': {
                'top_left': spectral_data[0, 0, :].tolist(),
                'top_right': spectral_data[0, -1, :].tolist(),
                'bottom_left': spectral_data[-1, 0, :].tolist(),
                'bottom_right': spectral_data[-1, -1, :].tolist()
            },
            'average_signature': np.mean(spectral_data, axis=(0, 1)).tolist(),
            'wavelengths': wavelengths.tolist() if wavelengths is not None else list(np.linspace(400, 2500, bands))
        }
        
        # Band-wise analysis
        band_analysis = {}
        for i in range(0, bands, bands//10):  # Sample every 10th of bands
            band_data = spectral_data[:, :, i]
            band_analysis[f'band_{i}'] = {
                'wavelength': float(wavelengths[i]) if wavelengths is not None else 400 + (i * 11.29),
                'min_value': float(np.min(band_data)),
                'max_value': float(np.max(band_data)),
                'mean_value': float(np.mean(band_data)),
                'std_value': float(np.std(band_data)),
                'unique_values': int(len(np.unique(band_data)))
            }
        
        # Vegetation indices expanded analysis
        vegetation_indices = full_analysis.get('vegetation_indices', {})
        expanded_vegetation = {}
        
        # Define thresholds for interpretation
        thresholds = {
            'ndvi': {'poor': 0.2, 'moderate': 0.4, 'good': 0.7},
            'evi': {'poor': 0.2, 'moderate': 0.4, 'good': 0.6},
            'savi': {'poor': 0.15, 'moderate': 0.3, 'good': 0.5},
            'gndvi': {'poor': 0.2, 'moderate': 0.4, 'good': 0.6}
        }
        
        for index_name, value in vegetation_indices.items():
            if isinstance(value, (int, float)):
                # Simple interpretation based on thresholds
                thresh = thresholds.get(index_name, {'poor': 0.2, 'moderate': 0.4, 'good': 0.6})
                if value < thresh['poor']:
                    interpretation = 'Poor vegetation health'
                    status = 'Critical'
                elif value < thresh['moderate']:
                    interpretation = 'Moderate vegetation health'
                    status = 'Warning'
                elif value < thresh['good']:
                    interpretation = 'Good vegetation health'
                    status = 'Acceptable'
                else:
                    interpretation = 'Excellent vegetation health'
                    status = 'Optimal'
                    
                expanded_vegetation[index_name] = {
                    'value': float(value),
                    'interpretation': interpretation,
                    'threshold_status': status,
                    'confidence_level': 'High' if abs(value) > 0.1 else 'Medium'
                }
        
        # Compile comprehensive response
        comprehensive_response = {
            'status': 'success',
            'analysis_type': 'comprehensive_hyperspectral_analysis',
            'timestamp': datetime.now().isoformat(),
            'data_source': {
                'file': 'aviris150.mat',
                'type': 'AVIRIS Hyperspectral Imagery',
                'format': 'MATLAB .mat file',
                'sensor_type': 'Airborne Visible/Infrared Imaging Spectrometer'
            },
            'spectral_statistics': spectral_stats,
            'core_analysis': full_analysis,
            'detailed_vegetation_indices': expanded_vegetation,
            'spectral_signatures': sample_signatures,
            'band_analysis': band_analysis,
            'agricultural_insights': {
                'crop_condition_summary': f"Overall health: {full_analysis.get('crop_health', {}).get('health_status', 'Unknown')}",
                'management_recommendations': ["Monitor water stress levels", "Consider precision agriculture", "Regular hyperspectral monitoring"],
                'priority_areas': f"{np.sum(spectral_data < np.mean(spectral_data))} pixels below average",
                'seasonal_indicators': "Analysis based on current spectral signatures"
            },
            'technical_metadata': {
                'processing_time': datetime.now().isoformat(),
                'algorithm_versions': {
                    'ndvi': '1.0',
                    'evi': '2.0', 
                    'savi': '1.0',
                    'water_stress': '1.5',
                    'disease_detection': '1.2'
                },
                'quality_flags': {'data_quality': 'Good', 'missing_pixels': 0, 'anomalous_pixels': int(np.sum(spectral_data < 0))},
                'data_completeness': f"{100.0 - (np.sum(spectral_data == 0) / spectral_data.size * 100):.2f}% complete"
            }
        }
        
        # Add summary for quick access
        comprehensive_response['executive_summary'] = {
            'overall_health_status': full_analysis.get('crop_health', {}).get('health_status', 'Unknown'),
            'health_score': full_analysis.get('crop_health', {}).get('overall_score', 0),
            'primary_concerns': [concern for concern in ['Water stress', 'Low vegetation', 'Disease risk'] if full_analysis.get('water_stress', {}).get('stress_level') == 'High' or full_analysis.get('crop_health', {}).get('health_status') == 'Critical'],
            'key_metrics': {
                'ndvi': vegetation_indices.get('ndvi', 0),
                'evi': vegetation_indices.get('evi', 0),
                'water_stress': full_analysis.get('water_stress', {}).get('stress_index', 0),
                'disease_risk': full_analysis.get('disease_detection', {}).get('risk_score', 0)
            },
            'spatial_coverage': {
                'total_area_analyzed': height * width,
                'healthy_pixels': full_analysis.get('spatial_analysis', {}).get('spatial_zones', {}).get('healthy_area_percent', 0),
                'stressed_pixels': full_analysis.get('spatial_analysis', {}).get('spatial_zones', {}).get('stressed_area_percent', 0)
            }
        }
        
        return jsonify(comprehensive_response)
        
    except Exception as e:
        print(f"Error in comprehensive hyperspectral analysis: {e}")
        return jsonify({
            'status': 'error',
            'analysis_type': 'comprehensive_hyperspectral_analysis',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/analyze/enhanced', methods=['POST'])
def analyze_enhanced():
    """
    Enhanced agricultural analysis endpoint
    Combines real data analysis with hyperspectral analysis
    Provides cross-validation and integrated insights
    """
    try:
        print("Starting enhanced agricultural analysis...")
        
        # Initialize enhanced system
        enhanced_system = EnhancedAgriculturalSystem()
        
        # Generate unified comprehensive report
        report = enhanced_system.generate_unified_report()
        
        # Extract key metrics for summary
        real_analysis = report.get('real_data_analysis', {})
        hyperspectral_analysis = report.get('hyperspectral_analysis', {})
        insights = report.get('integrated_insights', {})
        
        response = {
            'status': 'success',
            'analysis_type': 'enhanced_analysis',
            'timestamp': datetime.now().isoformat(),
            'data': report,
            'summary': {
                # Real data metrics
                'real_stress_index': real_analysis.get('crop_stress', {}).get('stress_index', 0),
                'soil_grade': real_analysis.get('soil_condition', {}).get('soil_grade', 'Unknown'),
                'fertility_score': real_analysis.get('soil_condition', {}).get('fertility_score', 0),
                
                # Hyperspectral metrics
                'vegetation_health_score': hyperspectral_analysis.get('crop_health', {}).get('overall_score', 0),
                'ndvi_value': hyperspectral_analysis.get('vegetation_indices', {}).get('ndvi', 0),
                'water_stress_level': hyperspectral_analysis.get('water_stress', {}).get('stress_level', 'Unknown'),
                
                # Integrated insights
                'overall_health_grade': insights.get('integrated_field_health', {}).get('health_grade', 'Unknown'),
                'integrated_health_score': insights.get('integrated_field_health', {}).get('overall_health_score', 0),
                'stress_correlation': insights.get('stress_analysis_correlation', {}).get('agreement_level', 'Unknown'),
                'spatial_priority': insights.get('field_spatial_assessment', {}).get('spatial_priority', 'Unknown')
            },
            'recommendations': report.get('action_recommendations', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'analysis_type': 'enhanced_analysis',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/reports', methods=['GET'])
def list_reports():
    """
    List all available analysis reports
    """
    try:
        # Find all JSON report files
        report_files = []
        for filename in os.listdir('.'):
            if filename.endswith('.json') and any(keyword in filename for keyword in 
                ['agricultural', 'hyperspectral', 'enhanced']):
                
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(filename),
                    'created': datetime.fromtimestamp(os.path.getctime(filename)).isoformat(),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filename)).isoformat()
                }
                
                # Try to extract analysis type from filename
                if 'real_agricultural' in filename:
                    file_info['type'] = 'real_data_analysis'
                elif 'hyperspectral' in filename:
                    file_info['type'] = 'hyperspectral_analysis'
                elif 'enhanced' in filename:
                    file_info['type'] = 'enhanced_analysis'
                else:
                    file_info['type'] = 'unknown'
                
                report_files.append(file_info)
        
        # Sort by creation time (newest first)
        report_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_reports': len(report_files),
            'reports': report_files
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/reports/<filename>', methods=['GET'])
def get_report(filename):
    """
    Download or view a specific report file
    """
    try:
        # Validate filename to prevent directory traversal
        if not filename.endswith('.json') or '/' in filename or '\\' in filename:
            return jsonify({
                'status': 'error',
                'message': 'Invalid filename'
            }), 400
            
        # Check if file exists
        if not os.path.exists(filename):
            return jsonify({
                'status': 'error',
                'message': 'Report not found'
            }), 404
            
        # Check if client wants to download or view
        download = request.args.get('download', 'false').lower() == 'true'
        
        if download:
            # Send file as download
            return send_file(filename, as_attachment=True, download_name=filename)
        else:
            # Return JSON content
            with open(filename, 'r') as f:
                report_data = json.load(f)
            
            return jsonify({
                'status': 'success',
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'data': report_data
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/data/info', methods=['GET'])
def data_info():
    """
    Get information about available data sources
    """
    try:
        data_info = {}
        
        # Check weather data
        if os.path.exists('weather_subset.csv'):
            import pandas as pd
            weather_df = pd.read_csv('weather_subset.csv')
            data_info['weather'] = {
                'file': 'weather_subset.csv',
                'records': len(weather_df),
                'columns': list(weather_df.columns),
                'date_range': {
                    'avg_temp_range': [float(weather_df['avg_temp'].min()), float(weather_df['avg_temp'].max())],
                    'precipitation_total': float(weather_df['precipitation'].sum())
                }
            }
        
        # Check soil data
        if os.path.exists('soil_subset.csv'):
            import pandas as pd
            soil_df = pd.read_csv('soil_subset.csv')
            data_info['soil'] = {
                'file': 'soil_subset.csv',
                'records': len(soil_df),
                'columns': list(soil_df.columns),
                'ph_range': [float(soil_df['ph'].min()), float(soil_df['ph'].max())] if 'ph' in soil_df.columns else None
            }
        
        # Check sensor data
        if os.path.exists('sensor_data.csv'):
            import pandas as pd
            sensor_df = pd.read_csv('sensor_data.csv')
            data_info['sensor'] = {
                'file': 'sensor_data.csv',
                'records': len(sensor_df),
                'columns': list(sensor_df.columns)
            }
        
        # Check hyperspectral data
        if os.path.exists('aviris150.mat'):
            import scipy.io
            mat_data = scipy.io.loadmat('aviris150.mat')
            for key, value in mat_data.items():
                if not key.startswith('__') and hasattr(value, 'shape'):
                    data_info['hyperspectral'] = {
                        'file': 'aviris150.mat',
                        'shape': list(value.shape),
                        'data_type': str(value.dtype),
                        'size_mb': round(os.path.getsize('aviris150.mat') / (1024*1024), 2)
                    }
                    break
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'data_sources': data_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("="*60)
    print("AGRICULTURAL ANALYSIS API SERVER")
    print("="*60)
    print("Starting Flask API server...")
    print("Available endpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /analyze/real  - Real data analysis")
    print("  POST /analyze/hyperspectral - Hyperspectral analysis")
    print("  POST /analyze/enhanced - Enhanced analysis")
    print("  GET  /reports       - List reports")
    print("  GET  /reports/<file> - Get specific report")
    print("  GET  /data/info     - Data source information")
    print("="*60)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode
    )