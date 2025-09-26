"""
Enhanced Agricultural Analysis System
Integrates hyperspectral analysis with existing real agricultural analysis
without modifying the working real_agricultural_analysis.py code
Now includes: 30-second monitoring, temp file storage, and area chart visualization
"""

import json
import time
import tempfile
import os
from datetime import datetime
from collections import deque
from real_agricultural_analysis import RealAgriculturalAnalysis
from hyperspectral_analyzer import HyperspectralAnalyzer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class EnhancedAgriculturalSystem:
    """Comprehensive agricultural analysis combining real data analysis with hyperspectral insights"""
    
    def __init__(self, max_history=100):
        self.real_analyzer = RealAgriculturalAnalysis()
        self.hyperspectral_analyzer = HyperspectralAnalyzer()
        self.max_history = max_history
        
        # Data storage for time-series monitoring
        self.monitoring_data = {
            'timestamps': deque(maxlen=max_history),
            'stress_index': deque(maxlen=max_history),
            'health_score': deque(maxlen=max_history),
            'ndvi_values': deque(maxlen=max_history),
            'fertility_scores': deque(maxlen=max_history),
            'integrated_scores': deque(maxlen=max_history)
        }
        
        # Create temp directory for storing results
        self.temp_dir = tempfile.mkdtemp(prefix='agri_monitoring_')
        print(f"📁 Temp directory created: {self.temp_dir}")
        
    def generate_unified_report(self):
        """Generate comprehensive report combining both analysis systems"""
        print("\n" + "="*80)
        print("ENHANCED AGRICULTURAL MONITORING SYSTEM")
        print("Combining Real Data Analysis + Advanced Hyperspectral Analysis")
        print("="*80)
        
        unified_report = {
            'system_type': 'Enhanced Agricultural Monitoring',
            'generated_at': datetime.now().isoformat(),
            'analysis_components': ['real_data_analysis', 'hyperspectral_analysis'],
            'data_integration': 'Multi-modal fusion approach'
        }
        
        # Get real agricultural analysis (your working code)
        print("\n🌱 RUNNING REAL DATA ANALYSIS...")
        print("-" * 50)
        real_analysis = self.real_analyzer.generate_comprehensive_report()
        unified_report['real_data_analysis'] = real_analysis
        
        # Get hyperspectral analysis
        print(f"\n🛰️  RUNNING HYPERSPECTRAL ANALYSIS...")
        print("-" * 50)
        hyperspectral_analysis = self.hyperspectral_analyzer.comprehensive_hyperspectral_analysis()
        unified_report['hyperspectral_analysis'] = hyperspectral_analysis
        
        # Cross-validation and insights
        print(f"\n🔄 CROSS-VALIDATION & INTEGRATED INSIGHTS")
        print("=" * 60)
        
        insights = self._generate_integrated_insights(real_analysis, hyperspectral_analysis)
        unified_report['integrated_insights'] = insights
        
        # Print integrated insights
        for category, insight_data in insights.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            if isinstance(insight_data, dict):
                for key, value in insight_data.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            print(f"  - {key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            print(f"  - {key.replace('_', ' ').title()}: {value}")
                    else:
                        print(f"  - {key.replace('_', ' ').title()}: {value}")
                        
        # Action recommendations
        print(f"\n📋 ACTIONABLE RECOMMENDATIONS")
        print("=" * 60)
        recommendations = self._generate_action_recommendations(real_analysis, hyperspectral_analysis)
        unified_report['action_recommendations'] = recommendations
        
        for priority, actions in recommendations.items():
            print(f"\n{priority.upper()} PRIORITY:")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {action}")
        
        # Save unified report
        report_filename = f"enhanced_agricultural_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(unified_report, f, indent=2)
            
        print(f"\n{'='*80}")
        print("ENHANCED AGRICULTURAL ANALYSIS COMPLETE")
        print(f"Unified Report saved to: {report_filename}")
        print("="*80)
        
        return unified_report
    
    def _generate_integrated_insights(self, real_analysis, hyperspectral_analysis):
        """Generate integrated insights by combining both analysis results"""
        insights = {}
        
        try:
            # Cross-validate crop stress
            real_stress = real_analysis.get('crop_stress', {}).get('stress_index', 0)
            hyperspectral_health = hyperspectral_analysis.get('crop_health', {}).get('overall_score', 50)
            
            # Convert hyperspectral health to stress (inverse relationship)
            hyperspectral_stress = 100 - hyperspectral_health
            
            stress_correlation = abs(real_stress - hyperspectral_stress) / 100
            stress_agreement = 1 - stress_correlation
            
            insights['stress_analysis_correlation'] = {
                'real_data_stress': real_stress,
                'hyperspectral_stress': hyperspectral_stress,
                'correlation_strength': stress_agreement,
                'agreement_level': 'High' if stress_agreement > 0.7 else 'Moderate' if stress_agreement > 0.4 else 'Low'
            }
            
            # Water stress cross-validation
            real_moisture = real_analysis.get('soil_condition', {}).get('nutrients', {}).get('moisture', 0)
            hyperspectral_water_stress = hyperspectral_analysis.get('water_stress', {}).get('water_stress_index', 0.5)
            
            # Convert moisture to stress index (inverse)
            real_water_stress = max(0, (80 - real_moisture) / 80)
            
            water_correlation = 1 - abs(real_water_stress - hyperspectral_water_stress)
            
            insights['water_stress_validation'] = {
                'soil_moisture_based_stress': real_water_stress,
                'spectral_water_stress': hyperspectral_water_stress,
                'validation_score': water_correlation,
                'consistency': 'Consistent' if water_correlation > 0.6 else 'Inconsistent'
            }
            
            # Disease risk assessment
            real_weather_risk = max([
                risk for risk in real_analysis.get('weather_analysis', {}).get('disease_risk_factors', {}).values()
            ] + [0])
            
            hyperspectral_disease_risk = hyperspectral_analysis.get('disease_analysis', {}).get('disease_risk_score', 0)
            
            combined_disease_risk = (real_weather_risk + hyperspectral_disease_risk) / 2
            
            insights['disease_risk_assessment'] = {
                'weather_based_risk': real_weather_risk,
                'spectral_anomaly_risk': hyperspectral_disease_risk,
                'combined_risk_score': combined_disease_risk,
                'risk_level': 'Critical' if combined_disease_risk > 0.7 else 'High' if combined_disease_risk > 0.5 else 'Moderate' if combined_disease_risk > 0.3 else 'Low'
            }
            
            # Spatial coverage analysis
            spatial_data = hyperspectral_analysis.get('spatial_analysis', {})
            if spatial_data:
                stressed_areas = spatial_data.get('spatial_zones', {}).get('stressed_area_percent', 0)
                field_uniformity = spatial_data.get('uniformity_index', 1)
                
                insights['field_spatial_assessment'] = {
                    'stressed_area_coverage': stressed_areas,
                    'field_uniformity_index': field_uniformity,
                    'spatial_priority': 'High' if stressed_areas > 30 else 'Medium' if stressed_areas > 15 else 'Low'
                }
            
            # Overall field health integration
            vegetation_health = hyperspectral_analysis.get('crop_health', {}).get('overall_score', 50)
            soil_health = real_analysis.get('soil_condition', {}).get('fertility_score', 50)
            
            # Weighted overall health (considering all factors)
            stress_penalty = real_stress * 0.3
            disease_penalty = combined_disease_risk * 20
            water_penalty = max(real_water_stress, hyperspectral_water_stress) * 15
            
            integrated_health_score = vegetation_health * 0.4 + soil_health * 0.3 - stress_penalty - disease_penalty - water_penalty
            integrated_health_score = max(0, min(100, integrated_health_score))
            
            insights['integrated_field_health'] = {
                'overall_health_score': integrated_health_score,
                'vegetation_component': vegetation_health,
                'soil_component': soil_health,
                'stress_impact': stress_penalty,
                'disease_impact': disease_penalty,
                'water_impact': water_penalty,
                'health_grade': 'A' if integrated_health_score >= 85 else 'B' if integrated_health_score >= 70 else 'C' if integrated_health_score >= 55 else 'D' if integrated_health_score >= 40 else 'F'
            }
            
        except Exception as e:
            print(f"Error generating integrated insights: {e}")
            insights['error'] = str(e)
            
        return insights
    
    def _generate_action_recommendations(self, real_analysis, hyperspectral_analysis):
        """Generate prioritized action recommendations based on integrated analysis"""
        recommendations = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        try:
            # Critical actions (immediate intervention required)
            real_stress = real_analysis.get('crop_stress', {}).get('stress_index', 0)
            if real_stress > 80:
                recommendations['critical'].append("IMMEDIATE: Implement emergency irrigation - critical drought stress detected")
                
            disease_risk = hyperspectral_analysis.get('disease_analysis', {}).get('disease_risk_score', 0)
            if disease_risk > 0.7:
                recommendations['critical'].append("IMMEDIATE: Apply preventive fungicide treatment - high disease risk detected")
                
            water_stress = hyperspectral_analysis.get('water_stress', {}).get('water_stress_index', 0)
            if water_stress > 0.8:
                recommendations['critical'].append("IMMEDIATE: Increase irrigation frequency - severe water deficit detected")
                
            # High priority actions
            avg_temp = real_analysis.get('crop_stress', {}).get('analysis_details', {}).get('temperature', {}).get('average', 0)
            if avg_temp > 35:
                recommendations['high'].append("Install shade nets or cooling systems - excessive heat stress")
                
            nitrogen = real_analysis.get('soil_condition', {}).get('nutrients', {}).get('nitrogen', 0)
            phosphorus = real_analysis.get('soil_condition', {}).get('nutrients', {}).get('phosphorus', 0)
            if nitrogen < 15 or phosphorus < 10:
                recommendations['high'].append("Apply balanced NPK fertilizer - critical nutrient deficiency")
                
            stressed_areas = hyperspectral_analysis.get('spatial_analysis', {}).get('spatial_zones', {}).get('stressed_area_percent', 0)
            if stressed_areas > 25:
                recommendations['high'].append("Implement precision agriculture for stressed zones - significant spatial variation")
                
            # Medium priority actions
            soil_grade = real_analysis.get('soil_condition', {}).get('soil_grade', 'C')
            if soil_grade in ['D', 'F']:
                recommendations['medium'].append("Soil amendment program - improve soil fertility with organic matter")
                
            vegetation_health = hyperspectral_analysis.get('crop_health', {}).get('overall_score', 50)
            if 40 < vegetation_health < 70:
                recommendations['medium'].append("Enhanced crop monitoring - moderate vegetation health requires attention")
                
            uniformity = hyperspectral_analysis.get('spatial_analysis', {}).get('uniformity_index', 1)
            if uniformity < 0.7:
                recommendations['medium'].append("Variable rate application - field uniformity improvement needed")
                
            # Low priority actions (maintenance/optimization)
            if disease_risk > 0.3:
                recommendations['low'].append("Preventive crop protection measures - moderate disease risk")
                
            if real_stress > 40:
                recommendations['low'].append("Install soil moisture sensors for better irrigation management")
                
            ph_quality = real_analysis.get('soil_condition', {}).get('ph_assessment', {}).get('quality', '')
            if ph_quality == 'Good':
                recommendations['low'].append("Monitor soil pH levels - maintain current good conditions")
                
            recommendations['low'].append("Regular hyperspectral monitoring for early problem detection")
            
            # Ensure each category has content
            if not recommendations['critical']:
                recommendations['critical'].append("No critical issues detected - continue regular monitoring")
            if not recommendations['high']:
                recommendations['high'].append("Maintain current management practices")
            if not recommendations['medium']:
                recommendations['medium'].append("Consider preventive measures for long-term health")
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations['critical'].append("Error in analysis - manual inspection required")
            
        return recommendations


    def save_temp_report(self, report_data, iteration):
        """Save report to temp file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agri_report_{timestamp}_iter{iteration:03d}.json"
        filepath = os.path.join(self.temp_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"💾 Report saved: {filename}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None
    
    def update_monitoring_data(self, report_data):
        """Update time-series monitoring data"""
        now = datetime.now()
        self.monitoring_data['timestamps'].append(now)
        
        # Extract key metrics
        try:
            real_data = report_data.get('real_data_analysis', {})
            hyperspectral_data = report_data.get('hyperspectral_analysis', {})
            insights = report_data.get('integrated_insights', {})
            
            # Stress index from real analysis
            crop_stress = real_data.get('crop_stress', {})
            stress_index = crop_stress.get('stress_index', 0)
            self.monitoring_data['stress_index'].append(stress_index)
            
            # Health score from hyperspectral
            health_score = hyperspectral_data.get('crop_health_assessment', {}).get('health_score', 0)
            self.monitoring_data['health_score'].append(health_score)
            
            # NDVI values
            vegetation_indices = hyperspectral_data.get('vegetation_indices', {})
            ndvi = vegetation_indices.get('ndvi', 0)
            self.monitoring_data['ndvi_values'].append(ndvi * 100)  # Convert to percentage
            
            # Fertility scores
            soil_condition = real_data.get('soil_condition', {})
            fertility = soil_condition.get('fertility_score', 0)
            self.monitoring_data['fertility_scores'].append(fertility)
            
            # Integrated health score
            integrated_score = insights.get('integrated_health_score', 0)
            self.monitoring_data['integrated_scores'].append(integrated_score)
            
        except Exception as e:
            print(f"⚠️  Error updating monitoring data: {e}")
    
    def generate_area_chart(self, iteration):
        """Generate area chart with agricultural metrics over time"""
        if len(self.monitoring_data['timestamps']) < 2:
            print("📊 Not enough data points for chart generation")
            return None
            
        try:
            # Set up the plot
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('🌱 Agricultural Monitoring Dashboard - Real-time Analysis', 
                        fontsize=16, fontweight='bold')
            
            timestamps = list(self.monitoring_data['timestamps'])
            
            # Top subplot - Health and Stress Metrics
            stress_data = list(self.monitoring_data['stress_index'])
            health_data = list(self.monitoring_data['health_score'])
            integrated_data = list(self.monitoring_data['integrated_scores'])
            
            ax1.fill_between(timestamps, stress_data, alpha=0.7, color='red', label='Stress Index')
            ax1.fill_between(timestamps, health_data, alpha=0.6, color='green', label='Health Score')
            ax1.fill_between(timestamps, integrated_data, alpha=0.5, color='blue', label='Integrated Score')
            
            ax1.set_title('Health & Stress Monitoring', fontweight='bold')
            ax1.set_ylabel('Score (0-100)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Bottom subplot - NDVI and Fertility
            ndvi_data = list(self.monitoring_data['ndvi_values'])
            fertility_data = list(self.monitoring_data['fertility_scores'])
            
            ax2.fill_between(timestamps, ndvi_data, alpha=0.7, color='darkgreen', label='NDVI (%)')
            ax2.fill_between(timestamps, fertility_data, alpha=0.6, color='brown', label='Soil Fertility')
            
            ax2.set_title('Vegetation & Soil Quality', fontweight='bold')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Score')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart without displaying
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"agri_chart_{timestamp}_iter{iteration:03d}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close without displaying
            
            print(f"📈 Area chart saved: {chart_filename}")
            return chart_path
            
        except Exception as e:
            print(f"❌ Error generating area chart: {e}")
            return None
    
    def generate_hyperspectral_visualization(self, iteration):
        """Generate hyperspectral image visualization"""
        try:
            # Get hyperspectral data from analyzer
            hyperspectral_data = self.hyperspectral_analyzer.spectral_data
            
            if hyperspectral_data is None:
                print("⚠️ No hyperspectral data available for visualization")
                return None
                
            # Create RGB composite and vegetation index visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'🛰️ Hyperspectral Analysis - Iteration {iteration}', fontsize=16, fontweight='bold')
            
            height, width, bands = hyperspectral_data.shape
            
            # 1. RGB Composite (using bands approximating R, G, B)
            rgb_bands = [int(bands * 0.7), int(bands * 0.5), int(bands * 0.3)]  # Approximate RGB
            rgb_composite = hyperspectral_data[:, :, rgb_bands]
            rgb_composite = (rgb_composite - rgb_composite.min()) / (rgb_composite.max() - rgb_composite.min())
            ax1.imshow(rgb_composite)
            ax1.set_title('RGB Composite', fontweight='bold')
            ax1.axis('off')
            
            # 2. NDVI Visualization
            red_band = hyperspectral_data[:, :, int(bands * 0.7)]  # ~700nm (red)
            nir_band = hyperspectral_data[:, :, int(bands * 0.8)]  # ~800nm (NIR)
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
            
            ndvi_plot = ax2.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            ax2.set_title('NDVI Map', fontweight='bold')
            ax2.axis('off')
            plt.colorbar(ndvi_plot, ax=ax2, fraction=0.046, pad=0.04)
            
            # 3. Spectral Signature Sample
            center_y, center_x = height // 2, width // 2
            sample_spectrum = hyperspectral_data[center_y-5:center_y+5, center_x-5:center_x+5, :].mean(axis=(0, 1))
            wavelengths = np.linspace(400, 2500, bands)  # Assuming 400-2500nm range
            
            ax3.plot(wavelengths, sample_spectrum, 'b-', linewidth=1.5)
            ax3.set_title('Center Pixel Spectral Signature', fontweight='bold')
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Reflectance')
            ax3.grid(True, alpha=0.3)
            
            # 4. Vegetation Health Map
            # Calculate simple vegetation health based on multiple indices
            green_band = hyperspectral_data[:, :, int(bands * 0.5)]  # ~550nm (green)
            evi_factor = 2.5 * (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * green_band + 1)
            health_map = np.clip((ndvi + evi_factor) / 2, -1, 1)
            
            health_plot = ax4.imshow(health_map, cmap='RdYlGn', vmin=-1, vmax=1)
            ax4.set_title('Vegetation Health Map', fontweight='bold')
            ax4.axis('off')
            plt.colorbar(health_plot, ax=ax4, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save hyperspectral visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hyper_filename = f"hyperspectral_viz_{timestamp}_iter{iteration:03d}.png"
            hyper_path = os.path.join(self.temp_dir, hyper_filename)
            plt.savefig(hyper_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🛰️ Hyperspectral visualization saved: {hyper_filename}")
            return hyper_path
            
        except Exception as e:
            print(f"❌ Error generating hyperspectral visualization: {e}")
            return None
    
    def export_frontend_data(self, iteration):
        """Export graph data in frontend-compatible format (JSON)"""
        if not self.monitoring_data['timestamps']:
            return None
            
        try:
            # Convert timestamps to ISO format for frontend
            timestamps = [ts.isoformat() for ts in self.monitoring_data['timestamps']]
            
            frontend_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "iteration": iteration,
                    "data_points": len(timestamps),
                    "interval_seconds": 10,
                    "chart_type": "area"
                },
                "time_series": {
                    "timestamps": timestamps,
                    "datasets": [
                        {
                            "name": "Stress Index",
                            "color": "#FF6B6B",
                            "data": list(self.monitoring_data['stress_index']),
                            "unit": "score",
                            "range": [0, 100],
                            "description": "Overall crop stress level (higher = more stressed)"
                        },
                        {
                            "name": "Health Score", 
                            "color": "#4ECDC4",
                            "data": list(self.monitoring_data['health_score']),
                            "unit": "score",
                            "range": [0, 100],
                            "description": "Hyperspectral-derived vegetation health"
                        },
                        {
                            "name": "NDVI Percentage",
                            "color": "#45B7D1", 
                            "data": list(self.monitoring_data['ndvi_values']),
                            "unit": "percentage",
                            "range": [0, 100],
                            "description": "Normalized Difference Vegetation Index"
                        },
                        {
                            "name": "Soil Fertility",
                            "color": "#F7DC6F",
                            "data": list(self.monitoring_data['fertility_scores']),
                            "unit": "score", 
                            "range": [0, 100],
                            "description": "Soil nutrient and pH-based fertility assessment"
                        },
                        {
                            "name": "Integrated Score",
                            "color": "#BB8FCE",
                            "data": list(self.monitoring_data['integrated_scores']),
                            "unit": "score",
                            "range": [0, 100], 
                            "description": "Combined health assessment from all data sources"
                        }
                    ]
                },
                "summary_stats": {
                    "current_values": {
                        "stress_index": self.monitoring_data['stress_index'][-1] if self.monitoring_data['stress_index'] else 0,
                        "health_score": self.monitoring_data['health_score'][-1] if self.monitoring_data['health_score'] else 0,
                        "ndvi_percentage": self.monitoring_data['ndvi_values'][-1] if self.monitoring_data['ndvi_values'] else 0,
                        "fertility_score": self.monitoring_data['fertility_scores'][-1] if self.monitoring_data['fertility_scores'] else 0,
                        "integrated_score": self.monitoring_data['integrated_scores'][-1] if self.monitoring_data['integrated_scores'] else 0
                    },
                    "trends": self._calculate_trends() if len(self.monitoring_data['timestamps']) > 1 else {}
                },
                "chart_config": {
                    "type": "area",
                    "stacked": False,
                    "smooth": True,
                    "fill_opacity": 0.3,
                    "stroke_width": 2,
                    "point_radius": 4,
                    "grid": True,
                    "legend": True,
                    "tooltip": True,
                    "zoom": True,
                    "responsive": True
                }
            }
            
            # Generate and save visualizations
            chart_path = None
            hyperspectral_path = None
            
            if len(self.monitoring_data['timestamps']) > 1:
                chart_path = self.generate_area_chart(iteration)
                hyperspectral_path = self.generate_hyperspectral_visualization(iteration)
            
            # Add visualization paths to frontend data
            frontend_data["visualizations"] = {
                "area_chart": {
                    "path": chart_path.replace(self.temp_dir + os.sep, "") if chart_path else None,
                    "full_path": chart_path,
                    "description": "Time-series area chart of agricultural metrics"
                },
                "hyperspectral_image": {
                    "path": hyperspectral_path.replace(self.temp_dir + os.sep, "") if hyperspectral_path else None,
                    "full_path": hyperspectral_path,
                    "description": "Hyperspectral analysis visualization with RGB, NDVI, and health maps"
                }
            }
            
            # Save frontend data to temp file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frontend_filename = f"frontend_data_{timestamp}_iter{iteration:03d}.json"
            frontend_filepath = os.path.join(self.temp_dir, frontend_filename)
            
            with open(frontend_filepath, 'w') as f:
                json.dump(frontend_data, f, indent=2, default=str)
                
            print(f"📊 Frontend data exported: {frontend_filename}")
            return frontend_data
            
        except Exception as e:
            print(f"❌ Error exporting frontend data: {e}")
            return None
    
    def _calculate_trends(self):
        """Calculate trend indicators for each metric"""
        trends = {}
        
        try:
            metrics = {
                'stress_index': list(self.monitoring_data['stress_index']),
                'health_score': list(self.monitoring_data['health_score']), 
                'ndvi_values': list(self.monitoring_data['ndvi_values']),
                'fertility_scores': list(self.monitoring_data['fertility_scores']),
                'integrated_scores': list(self.monitoring_data['integrated_scores'])
            }
            
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    recent_avg = sum(values[-3:]) / min(3, len(values))  # Last 3 points average
                    earlier_avg = sum(values[:-3] or values) / max(1, len(values[:-3] or values))
                    
                    change = recent_avg - earlier_avg
                    trend_direction = "improving" if change > 1 else "declining" if change < -1 else "stable"
                    
                    trends[metric_name] = {
                        "direction": trend_direction,
                        "change_value": round(change, 2),
                        "recent_average": round(recent_avg, 2),
                        "earlier_average": round(earlier_avg, 2)
                    }
                    
        except Exception as e:
            print(f"⚠️ Error calculating trends: {e}")
            
        return trends
    
    def continuous_monitoring(self, interval_seconds=10, max_iterations=None):
        """Run continuous monitoring with specified interval"""
        print(f"\n🚀 Starting Continuous Agricultural Monitoring")
        print(f"⏰ Interval: {interval_seconds} seconds")
        print(f"📁 Temp Directory: {self.temp_dir}")
        print(f"🔄 Max Iterations: {max_iterations or 'Unlimited'}")
        print("\n" + "="*80)
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 60)
                
                # Run comprehensive analysis
                report_data = self.generate_unified_report()
                
                # Update monitoring data
                self.update_monitoring_data(report_data)
                
                # Save to temp file
                self.save_temp_report(report_data, iteration)
                
                # Export frontend-compatible data (includes chart and hyperspectral generation)
                self.export_frontend_data(iteration)
                
                # Display current status
                self._display_current_status()
                
                # Wait for next iteration (except on last iteration)
                if max_iterations is None or iteration < max_iterations:
                    print(f"\n⏳ Waiting {interval_seconds} seconds for next analysis...")
                    time.sleep(interval_seconds)
                    
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user after {iteration} iterations")
        except Exception as e:
            print(f"\n\n❌ Error in continuous monitoring: {e}")
        finally:
            print(f"\n📊 Final Summary:")
            print(f"   - Total iterations: {iteration}")
            print(f"   - Data points collected: {len(self.monitoring_data['timestamps'])}")
            print(f"   - Temp files location: {self.temp_dir}")
            print(f"\n✅ Monitoring session completed!")
    
    def _display_current_status(self):
        """Display current monitoring status"""
        if not self.monitoring_data['timestamps']:
            return
            
        latest_idx = -1
        print(f"\n📊 CURRENT STATUS:")
        print(f"   🚨 Stress Index: {self.monitoring_data['stress_index'][latest_idx]:.1f}/100")
        print(f"   💚 Health Score: {self.monitoring_data['health_score'][latest_idx]:.1f}/100")
        print(f"   🌿 NDVI: {self.monitoring_data['ndvi_values'][latest_idx]:.2f}%")
        print(f"   🏞️  Fertility: {self.monitoring_data['fertility_scores'][latest_idx]:.1f}/100")
        print(f"   🔗 Integrated: {self.monitoring_data['integrated_scores'][latest_idx]:.1f}/100")


if __name__ == "__main__":
    # Run enhanced agricultural analysis system with continuous monitoring
    print("🌱 Enhanced Agricultural Monitoring System")
    print("🕒 10-Second Interval Monitoring with Area Charts & Frontend Data Export")
    
    enhanced_system = EnhancedAgriculturalSystem(max_history=50)
    
    # Run continuous monitoring (10 seconds interval)
    # You can specify max_iterations to limit the number of runs
    enhanced_system.continuous_monitoring(interval_seconds=10, max_iterations=10)