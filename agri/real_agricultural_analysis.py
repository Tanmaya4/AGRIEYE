import pandas as pd
import numpy as np
from datetime import datetime
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import json

class RealAgriculturalAnalysis:
    """Real agricultural analysis based on actual data values"""
    
    def __init__(self):
        self.weather_data = None
        self.soil_data = None
        self.sensor_data = None
        self.spectral_data = None
        
    def load_real_data(self):
        """Load actual data files"""
        print("Loading real data files...")
        
        # Load weather data
        try:
            self.weather_data = pd.read_csv('weather_subset.csv')
            print(f"Loaded weather data: {self.weather_data.shape}")
            print(f"Weather columns: {list(self.weather_data.columns)}")
        except Exception as e:
            print(f"Error loading weather data: {e}")
            
        # Load soil data
        try:
            self.soil_data = pd.read_csv('soil_subset.csv')
            print(f"Loaded soil data: {self.soil_data.shape}")
            print(f"Soil columns: {list(self.soil_data.columns)}")
        except Exception as e:
            print(f"Error loading soil data: {e}")
            
        # Load sensor data
        try:
            self.sensor_data = pd.read_csv('sensor_data.csv')
            print(f"Loaded sensor data: {self.sensor_data.shape}")
            print(f"Sensor columns: {list(self.sensor_data.columns)}")
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            
        # Load spectral data
        try:
            mat_data = scipy.io.loadmat('aviris150.mat')
            # Find the actual data array in the mat file
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    self.spectral_data = value
                    print(f"Loaded spectral data: {self.spectral_data.shape}")
                    break
        except Exception as e:
            print(f"Error loading spectral data: {e}")
            
    def analyze_crop_stress_real(self):
        """Calculate REAL crop stress based on actual data values"""
        stress_factors = []
        analysis_details = {}
        
        if self.weather_data is not None:
            # Actual temperature analysis
            avg_temp = self.weather_data['avg_temp'].mean()
            max_temp = self.weather_data['max_temp'].max()
            min_temp = self.weather_data['min_temp'].min()
            
            analysis_details['temperature'] = {
                'average': float(avg_temp),
                'maximum': float(max_temp),
                'minimum': float(min_temp)
            }
            
            # Temperature stress calculation
            temp_stress = 0
            if max_temp > 35:
                temp_stress = 25
            elif max_temp > 30:
                temp_stress = 15
            elif min_temp < 10:
                temp_stress = 20
            
            stress_factors.append(('temperature_stress', temp_stress))
            
            # Precipitation analysis
            total_precip = self.weather_data['precipitation'].sum()
            avg_precip = self.weather_data['precipitation'].mean()
            
            analysis_details['precipitation'] = {
                'total': float(total_precip),
                'average': float(avg_precip)
            }
            
            # Drought stress
            drought_stress = 0
            if avg_precip < 1.0:
                drought_stress = 30
            elif avg_precip < 2.0:
                drought_stress = 15
                
            stress_factors.append(('drought_stress', drought_stress))
        
        if self.sensor_data is not None:
            # Actual moisture analysis
            avg_moisture = self.sensor_data['moisture'].mean()
            min_moisture = self.sensor_data['moisture'].min()
            
            analysis_details['soil_moisture'] = {
                'average': float(avg_moisture),
                'minimum': float(min_moisture)
            }
            
            # Moisture stress
            moisture_stress = 0
            if avg_moisture < 30:
                moisture_stress = 25
            elif avg_moisture < 50:
                moisture_stress = 10
                
            stress_factors.append(('moisture_stress', moisture_stress))
            
            # Nutrient analysis
            avg_nitrogen = self.sensor_data['nitrogen'].mean()
            avg_phosphorus = self.sensor_data['phosphorus'].mean()
            
            analysis_details['nutrients'] = {
                'nitrogen_avg': float(avg_nitrogen),
                'phosphorus_avg': float(avg_phosphorus)
            }
            
            # Nutrient deficiency stress
            nutrient_stress = 0
            if avg_nitrogen < 20:
                nutrient_stress += 15
            if avg_phosphorus < 15:
                nutrient_stress += 10
                
            stress_factors.append(('nutrient_stress', nutrient_stress))
        
        # Calculate total stress index
        total_stress = sum([stress[1] for stress in stress_factors])
        total_stress = min(100, total_stress)  # Cap at 100
        
        # Determine stress level
        if total_stress < 20:
            stress_level = "Low"
        elif total_stress < 40:
            stress_level = "Medium"
        elif total_stress < 70:
            stress_level = "High"
        else:
            stress_level = "Critical"
            
        return {
            'stress_index': total_stress,
            'stress_level': stress_level,
            'stress_factors': dict(stress_factors),
            'analysis_details': analysis_details,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_soil_condition_real(self):
        """Analyze REAL soil condition based on actual pH and nutrient data"""
        analysis = {}
        
        if self.soil_data is not None and 'ph' in self.soil_data.columns:
            # Actual pH analysis
            avg_ph = self.soil_data['ph'].mean()
            min_ph = self.soil_data['ph'].min()
            max_ph = self.soil_data['ph'].max()
            ph_std = self.soil_data['ph'].std()
            
            analysis['ph_analysis'] = {
                'average': float(avg_ph),
                'minimum': float(min_ph),
                'maximum': float(max_ph),
                'standard_deviation': float(ph_std)
            }
            
            # pH quality assessment
            if 6.0 <= avg_ph <= 7.5:
                ph_quality = "Excellent"
                ph_score = 90
            elif 5.5 <= avg_ph <= 8.0:
                ph_quality = "Good"
                ph_score = 75
            elif 5.0 <= avg_ph <= 8.5:
                ph_quality = "Fair"
                ph_score = 60
            else:
                ph_quality = "Poor"
                ph_score = 40
                
            analysis['ph_assessment'] = {
                'quality': ph_quality,
                'score': ph_score
            }
        
        if self.sensor_data is not None:
            # Nutrient analysis from sensor data
            nutrients = {
                'nitrogen': self.sensor_data['nitrogen'].mean(),
                'phosphorus': self.sensor_data['phosphorus'].mean(),
                'moisture': self.sensor_data['moisture'].mean()
            }
            
            analysis['nutrients'] = {k: float(v) for k, v in nutrients.items()}
            
            # Nutrient quality scores
            nitrogen_score = min(100, (nutrients['nitrogen'] / 50) * 100)
            phosphorus_score = min(100, (nutrients['phosphorus'] / 30) * 100)
            moisture_score = min(100, (nutrients['moisture'] / 80) * 100)
            
            analysis['nutrient_scores'] = {
                'nitrogen_score': float(nitrogen_score),
                'phosphorus_score': float(phosphorus_score),
                'moisture_score': float(moisture_score)
            }
            
            # Overall fertility score
            fertility_score = (nitrogen_score + phosphorus_score + moisture_score) / 3
            if 'ph_assessment' in analysis:
                fertility_score = (fertility_score + analysis['ph_assessment']['score']) / 2
                
            analysis['fertility_score'] = float(fertility_score)
            
            # Grade assignment
            if fertility_score >= 85:
                grade = "A"
            elif fertility_score >= 70:
                grade = "B"
            elif fertility_score >= 55:
                grade = "C"
            elif fertility_score >= 40:
                grade = "D"
            else:
                grade = "F"
                
            analysis['soil_grade'] = grade
        
        analysis['timestamp'] = datetime.now().isoformat()
        return analysis
    
    def analyze_weather_patterns_real(self):
        """Analyze actual weather patterns for disease risk"""
        if self.weather_data is None:
            return None
            
        analysis = {}
        
        # Temperature analysis
        temp_stats = {
            'avg_temperature': float(self.weather_data['avg_temp'].mean()),
            'max_temperature': float(self.weather_data['max_temp'].max()),
            'min_temperature': float(self.weather_data['min_temp'].min()),
            'temp_range': float(self.weather_data['max_temp'].max() - self.weather_data['min_temp'].min())
        }
        
        # Humidity and precipitation analysis
        precip_stats = {
            'total_precipitation': float(self.weather_data['precipitation'].sum()),
            'avg_precipitation': float(self.weather_data['precipitation'].mean()),
            'max_precipitation': float(self.weather_data['precipitation'].max()),
            'days_with_rain': int((self.weather_data['precipitation'] > 0).sum())
        }
        
        # Wind analysis
        wind_stats = {
            'avg_wind_speed': float(self.weather_data['wind_speed'].mean()),
            'max_wind_speed': float(self.weather_data['wind_speed'].max())
        }
        
        # Disease risk assessment based on real conditions
        disease_risk_factors = []
        
        # High humidity + moderate temp = fungal risk
        if temp_stats['avg_temperature'] > 20 and precip_stats['days_with_rain'] > 5:
            disease_risk_factors.append(('fungal_infection_risk', 0.7))
            
        # Hot dry conditions = stress-related diseases
        if temp_stats['avg_temperature'] > 30 and precip_stats['avg_precipitation'] < 1:
            disease_risk_factors.append(('drought_stress_diseases', 0.6))
            
        # Temperature extremes = general plant stress
        if temp_stats['temp_range'] > 25:
            disease_risk_factors.append(('temperature_stress', 0.5))
        
        analysis = {
            'temperature_stats': temp_stats,
            'precipitation_stats': precip_stats,
            'wind_stats': wind_stats,
            'disease_risk_factors': dict(disease_risk_factors),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report based on REAL data analysis"""
        print("\n" + "="*60)
        print("REAL AGRICULTURAL DATA ANALYSIS REPORT")
        print("="*60)
        
        # Load data first
        self.load_real_data()
        
        report = {
            'report_type': 'Real Data Analysis',
            'generated_at': datetime.now().isoformat(),
            'data_sources': {}
        }
        
        # Document data sources
        if self.weather_data is not None:
            report['data_sources']['weather'] = f"{len(self.weather_data)} records"
        if self.soil_data is not None:
            report['data_sources']['soil'] = f"{len(self.soil_data)} records"
        if self.sensor_data is not None:
            report['data_sources']['sensor'] = f"{len(self.sensor_data)} records"
        if self.spectral_data is not None:
            report['data_sources']['spectral'] = f"Array shape: {self.spectral_data.shape}"
            
        print(f"\nData Sources Loaded:")
        for source, info in report['data_sources'].items():
            print(f"  - {source.capitalize()}: {info}")
        
        # Real crop stress analysis
        print(f"\n{'='*40}")
        print("1. CROP STRESS ANALYSIS (Real Data)")
        print('='*40)
        stress_analysis = self.analyze_crop_stress_real()
        report['crop_stress'] = stress_analysis
        
        print(f"Stress Index: {stress_analysis['stress_index']}/100 ({stress_analysis['stress_level']})")
        print(f"Contributing Factors:")
        for factor, value in stress_analysis['stress_factors'].items():
            print(f"  - {factor.replace('_', ' ').title()}: {value}")
            
        # Real soil analysis
        print(f"\n{'='*40}")
        print("2. SOIL CONDITION ANALYSIS (Real Data)")
        print('='*40)
        soil_analysis = self.analyze_soil_condition_real()
        report['soil_condition'] = soil_analysis
        
        if 'ph_analysis' in soil_analysis:
            ph_data = soil_analysis['ph_analysis']
            print(f"pH Analysis:")
            print(f"  - Average pH: {ph_data['average']:.2f}")
            print(f"  - pH Range: {ph_data['minimum']:.2f} - {ph_data['maximum']:.2f}")
            print(f"  - pH Quality: {soil_analysis['ph_assessment']['quality']}")
            
        if 'nutrients' in soil_analysis:
            nutrients = soil_analysis['nutrients']
            print(f"Nutrient Levels:")
            print(f"  - Nitrogen: {nutrients['nitrogen']:.1f}")
            print(f"  - Phosphorus: {nutrients['phosphorus']:.1f}")
            print(f"  - Moisture: {nutrients['moisture']:.1f}%")
            print(f"Overall Soil Grade: {soil_analysis.get('soil_grade', 'N/A')}")
            print(f"Fertility Score: {soil_analysis.get('fertility_score', 0):.1f}/100")
            
        # Real weather analysis
        print(f"\n{'='*40}")
        print("3. WEATHER PATTERN ANALYSIS (Real Data)")
        print('='*40)
        weather_analysis = self.analyze_weather_patterns_real()
        if weather_analysis:
            report['weather_analysis'] = weather_analysis
            
            temp_stats = weather_analysis['temperature_stats']
            precip_stats = weather_analysis['precipitation_stats']
            
            print(f"Temperature:")
            print(f"  - Average: {temp_stats['avg_temperature']:.1f}°C")
            print(f"  - Range: {temp_stats['min_temperature']:.1f}°C to {temp_stats['max_temperature']:.1f}°C")
            
            print(f"Precipitation:")
            print(f"  - Total: {precip_stats['total_precipitation']:.1f}mm")
            print(f"  - Days with rain: {precip_stats['days_with_rain']}")
            
            print(f"Disease Risk Factors:")
            for factor, risk in weather_analysis['disease_risk_factors'].items():
                print(f"  - {factor.replace('_', ' ').title()}: {risk:.1%}")
        
        # Save real report
        report_filename = f"real_agricultural_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n{'='*60}")
        print(f"REAL DATA ANALYSIS COMPLETE")
        print(f"Report saved to: {report_filename}")
        print("="*60)
        
        return report

if __name__ == "__main__":
    analyzer = RealAgriculturalAnalysis()
    analyzer.generate_comprehensive_report()