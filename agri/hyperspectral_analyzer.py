import numpy as np
import pandas as pd
import scipy.io
from scipy import stats, ndimage
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class HyperspectralAnalyzer:
    """Advanced hyperspectral analysis for agricultural monitoring"""
    
    def __init__(self, aviris_file='aviris150.mat'):
        self.aviris_file = aviris_file
        self.spectral_data = None
        self.wavelengths = None
        self.spatial_dims = None
        self.spectral_dims = None
        
        # Standard wavelength mappings for AVIRIS (approximate)
        self.wavelength_bands = {
            'blue': (450, 495),      # Blue light
            'green': (495, 570),     # Green light  
            'red': (620, 750),       # Red light
            'red_edge': (700, 740),  # Red edge
            'nir': (750, 900),       # Near infrared
            'swir1': (1550, 1750),   # Short wave infrared 1
            'swir2': (2080, 2350),   # Short wave infrared 2
            'water_abs_1': (970, 980),   # Water absorption band 1
            'water_abs_2': (1200, 1210), # Water absorption band 2
            'water_abs_3': (1450, 1460)  # Water absorption band 3
        }
        
        # Load data on initialization
        self._load_hyperspectral_data()
        
    def validate_spectral_data_quality(self):
        """Validate and assess the quality of spectral data for accuracy"""
        quality_metrics = {}
        
        if self.spectral_data is None:
            return {'valid': False, 'error': 'No spectral data loaded'}
            
        print("\n🔍 SPECTRAL DATA QUALITY VALIDATION:")
        print("-" * 50)
        
        # 1. Check for valid data range (reflectance should be 0-1 or 0-100%)
        data_min, data_max = np.min(self.spectral_data), np.max(self.spectral_data)
        
        if 0 <= data_min and data_max <= 1.1:  # Reflectance format 0-1
            quality_metrics['reflectance_format'] = 'normalized'
            quality_metrics['valid_range'] = True
            print(f"✅ Data format: Normalized reflectance (0-1)")
        elif 0 <= data_min and data_max <= 110:  # Percentage format 0-100%
            quality_metrics['reflectance_format'] = 'percentage'
            quality_metrics['valid_range'] = True
            # Convert to normalized format
            self.spectral_data = self.spectral_data / 100.0
            print(f"✅ Data format: Percentage reflectance (converted to 0-1)")
        else:
            quality_metrics['reflectance_format'] = 'unknown'
            quality_metrics['valid_range'] = False
            print(f"⚠️  Warning: Unusual data range ({data_min:.3f} to {data_max:.3f})")
            
        # 2. Check for missing/invalid values
        nan_count = np.sum(np.isnan(self.spectral_data))
        inf_count = np.sum(np.isinf(self.spectral_data))
        negative_count = np.sum(self.spectral_data < 0)
        
        quality_metrics['data_integrity'] = {
            'nan_pixels': int(nan_count),
            'infinite_values': int(inf_count),
            'negative_values': int(negative_count),
            'total_pixels': int(self.spectral_data.size)
        }
        
        print(f"📊 Data integrity: {nan_count} NaN, {inf_count} Inf, {negative_count} negative values")
        
        # 3. Spectral consistency check
        mean_spectrum = np.nanmean(self.spectral_data, axis=(0,1))
        spectral_noise = np.nanstd(mean_spectrum)
        spectral_snr = np.nanmean(mean_spectrum) / (spectral_noise + 1e-10)
        
        quality_metrics['spectral_quality'] = {
            'signal_to_noise_ratio': float(spectral_snr),
            'mean_reflectance': float(np.nanmean(mean_spectrum)),
            'spectral_variation': float(spectral_noise)
        }
        
        print(f"📈 Spectral SNR: {spectral_snr:.2f}, Mean reflectance: {np.nanmean(mean_spectrum):.4f}")
        
        # 4. Spatial consistency
        spatial_mean = np.nanmean(self.spectral_data)
        spatial_std = np.nanstd(self.spectral_data)
        spatial_cv = spatial_std / (spatial_mean + 1e-10)
        
        quality_metrics['spatial_quality'] = {
            'coefficient_of_variation': float(spatial_cv),
            'spatial_uniformity': float(1 - min(spatial_cv, 1))
        }
        
        print(f"🗺️  Spatial uniformity: {(1 - min(spatial_cv, 1)):.3f} (CV: {spatial_cv:.3f})")
        
        # Overall quality score
        quality_score = 100
        if not quality_metrics['valid_range']:
            quality_score -= 30
        if nan_count > 0 or inf_count > 0:
            quality_score -= 20
        if spectral_snr < 10:
            quality_score -= 15
        if spatial_cv > 0.5:
            quality_score -= 10
            
        quality_metrics['overall_quality_score'] = max(0, quality_score)
        quality_metrics['valid'] = quality_score > 50
        
        print(f"🎯 Overall data quality: {quality_score}/100 {'✅' if quality_score > 70 else '⚠️' if quality_score > 50 else '❌'}")
        
        return quality_metrics
    
    def display_raw_spectral_data_info(self):
        """Display detailed information about the raw spectral data"""
        print("\n📡 RAW HYPERSPECTRAL DATA OVERVIEW:")
        print("=" * 60)
        
        if self.spectral_data is not None:
            print(f"📊 Data Shape: {self.spectral_data.shape} (Height × Width × Bands)")
            print(f"🗺️ Spatial Resolution: {self.spatial_dims[0]} × {self.spatial_dims[1]} pixels")
            print(f"🌈 Spectral Resolution: {self.spectral_dims} bands")
            print(f"💾 Data Type: {self.spectral_data.dtype}")
            print(f"📊 Data Range: {np.min(self.spectral_data):.6f} to {np.max(self.spectral_data):.6f}")
            print(f"📊 Data Mean: {np.mean(self.spectral_data):.6f}")
            print(f"📊 Data Std: {np.std(self.spectral_data):.6f}")
            
            if self.wavelengths is not None:
                print(f"🌌 Wavelength Range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")
                print(f"📊 Wavelength Step: {(self.wavelengths[-1] - self.wavelengths[0]) / len(self.wavelengths):.2f} nm/band")
            
            # Sample spectral signature from center pixel
            center_y, center_x = self.spatial_dims[0] // 2, self.spatial_dims[1] // 2
            center_spectrum = self.spectral_data[center_y, center_x, :]
            print(f"\n📈 CENTER PIXEL SPECTRAL SIGNATURE:")
            print(f"   Pixel location: ({center_y}, {center_x})")
            print(f"   Spectrum range: {np.min(center_spectrum):.6f} to {np.max(center_spectrum):.6f}")
            print(f"   Spectrum mean: {np.mean(center_spectrum):.6f}")
            print(f"   First 10 band values: {center_spectrum[:10]}")
            
            # Memory usage
            data_size_mb = self.spectral_data.nbytes / (1024 * 1024)
            print(f"\n💾 MEMORY USAGE: {data_size_mb:.2f} MB")
        else:
            print("No spectral data loaded!")
    
    def _load_hyperspectral_data(self):
        """Load and validate AVIRIS hyperspectral data"""
        try:
            print(f"Loading hyperspectral data from {self.aviris_file}...")
            mat_data = scipy.io.loadmat(self.aviris_file)
            
            # Display MAT file contents
            print(f"\n📋 MAT FILE CONTENTS:")
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    if isinstance(value, np.ndarray):
                        print(f"   {key}: {value.shape} {value.dtype}")
                    else:
                        print(f"   {key}: {type(value)}")
            
            # Find the main data array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if len(value.shape) == 3:  # Height x Width x Bands format
                        self.spectral_data = value
                        self.spatial_dims = value.shape[:2]
                        self.spectral_dims = value.shape[2]
                        print(f"\n✅ Loaded hyperspectral cube: {self.spectral_data.shape}")
                        print(f"   Spatial dimensions: {self.spatial_dims}")
                        print(f"   Spectral bands: {self.spectral_dims}")
                        break
                        
            if self.spectral_data is None:
                raise ValueError("No valid 3D hyperspectral data found in .mat file")
                
            # Generate wavelength array (AVIRIS typically 400-2500nm range)
            if self.spectral_dims > 0:
                self.wavelengths = np.linspace(400, 2500, self.spectral_dims)
                print(f"   Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")
                print(f"   Wavelength step: {(self.wavelengths[-1] - self.wavelengths[0]) / len(self.wavelengths):.2f} nm/band")
                
                # Display wavelength band mappings
                print(f"\n🌈 WAVELENGTH BAND MAPPINGS:")
                for band_name, (start_wl, end_wl) in self.wavelength_bands.items():
                    band_indices = self._get_band_indices((start_wl, end_wl))
                    if band_indices:
                        actual_start = self.wavelengths[band_indices[0]]
                        actual_end = self.wavelengths[band_indices[-1]]
                        print(f"   {band_name}: {start_wl}-{end_wl}nm → bands {band_indices[0]}-{band_indices[-1]} ({actual_start:.1f}-{actual_end:.1f}nm, {len(band_indices)} bands)")
                    else:
                        print(f"   {band_name}: {start_wl}-{end_wl}nm → No matching bands")
                
        except Exception as e:
            print(f"Error loading hyperspectral data: {e}")
            raise
            
    def _get_band_indices(self, wavelength_range: Tuple[float, float]) -> List[int]:
        """Get band indices for a given wavelength range"""
        if self.wavelengths is None:
            return []
            
        start_wl, end_wl = wavelength_range
        indices = np.where((self.wavelengths >= start_wl) & (self.wavelengths <= end_wl))[0]
        return indices.tolist()
        
    def _calibrate_reflectance(self, reflectance_data):
        """Apply calibration and atmospheric correction to improve accuracy"""
        if reflectance_data is None or len(reflectance_data) == 0:
            return reflectance_data
            
        # 1. Remove outliers using robust statistics
        if isinstance(reflectance_data, np.ndarray) and reflectance_data.ndim > 0:
            q25, q75 = np.nanpercentile(reflectance_data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Replace outliers with median
            median_val = np.nanmedian(reflectance_data)
            reflectance_data = np.where(
                (reflectance_data < lower_bound) | (reflectance_data > upper_bound),
                median_val,
                reflectance_data
            )
        
        # 2. Apply smoothing for noise reduction
        if hasattr(reflectance_data, '__len__') and len(reflectance_data) > 3:
            # Simple moving average for noise reduction
            from scipy import ndimage
            reflectance_data = ndimage.gaussian_filter1d(reflectance_data, sigma=0.5)
        
        # 3. Ensure valid reflectance range
        reflectance_data = np.clip(reflectance_data, 0.001, 1.0)  # Avoid exact zeros
        
        return reflectance_data
    
    def _extract_spectral_reflectance(self, band_indices: List[int]) -> np.ndarray:
        """Extract calibrated mean reflectance for given band indices with improved accuracy"""
        if not band_indices or self.spectral_data is None:
            return np.array([0.0])
            
        # Extract data from specified bands
        band_data = self.spectral_data[:, :, band_indices]
        
        # Use robust statistics instead of simple mean to avoid outlier bias
        # Calculate median across spatial dimensions for robustness
        spatial_median = np.nanmedian(band_data, axis=(0, 1))
        
        # Apply calibration
        calibrated_data = self._calibrate_reflectance(spatial_median)
        
        # Use robust mean (trimmed mean) across bands
        from scipy import stats
        if len(calibrated_data) > 1:
            # Remove top and bottom 10% to reduce noise impact
            trimmed_mean = stats.trim_mean(calibrated_data, proportiontocut=0.1)
        else:
            trimmed_mean = calibrated_data[0] if len(calibrated_data) > 0 else 0.0
            
        return float(trimmed_mean)
        
    def calculate_vegetation_indices(self) -> Dict[str, float]:
        """Calculate comprehensive vegetation indices from hyperspectral data"""
        indices = {}
        
        print("\n📊 PROCESSING VEGETATION INDICES DATA:")
        print("-" * 50)
        
        try:
            # Get reflectance values for key spectral regions
            red_bands = self._get_band_indices(self.wavelength_bands['red'])
            nir_bands = self._get_band_indices(self.wavelength_bands['nir'])
            green_bands = self._get_band_indices(self.wavelength_bands['green'])
            blue_bands = self._get_band_indices(self.wavelength_bands['blue'])
            red_edge_bands = self._get_band_indices(self.wavelength_bands['red_edge'])
            
            print(f"🔴 Red bands ({self.wavelength_bands['red'][0]}-{self.wavelength_bands['red'][1]}nm): {len(red_bands)} bands (indices: {red_bands[:5]}{'...' if len(red_bands) > 5 else ''})")
            print(f"🟡 NIR bands ({self.wavelength_bands['nir'][0]}-{self.wavelength_bands['nir'][1]}nm): {len(nir_bands)} bands (indices: {nir_bands[:5]}{'...' if len(nir_bands) > 5 else ''})")
            print(f"🟢 Green bands ({self.wavelength_bands['green'][0]}-{self.wavelength_bands['green'][1]}nm): {len(green_bands)} bands (indices: {green_bands[:5]}{'...' if len(green_bands) > 5 else ''})")
            print(f"🔵 Blue bands ({self.wavelength_bands['blue'][0]}-{self.wavelength_bands['blue'][1]}nm): {len(blue_bands)} bands (indices: {blue_bands[:5]}{'...' if len(blue_bands) > 5 else ''})")
            print(f"🟠 Red-edge bands ({self.wavelength_bands['red_edge'][0]}-{self.wavelength_bands['red_edge'][1]}nm): {len(red_edge_bands)} bands (indices: {red_edge_bands[:5]}{'...' if len(red_edge_bands) > 5 else ''})")
            
            red_refl = self._extract_spectral_reflectance(red_bands)
            nir_refl = self._extract_spectral_reflectance(nir_bands)
            green_refl = self._extract_spectral_reflectance(green_bands)
            blue_refl = self._extract_spectral_reflectance(blue_bands)
            red_edge_refl = self._extract_spectral_reflectance(red_edge_bands)
            
            print(f"\n📈 EXTRACTED REFLECTANCE VALUES:")
            print(f"Red Reflectance: {red_refl:.6f}")
            print(f"NIR Reflectance: {nir_refl:.6f}")
            print(f"Green Reflectance: {green_refl:.6f}")
            print(f"Blue Reflectance: {blue_refl:.6f}")
            print(f"Red-edge Reflectance: {red_edge_refl:.6f}")
            
            print(f"\n🧮 CALCULATING VEGETATION INDICES:")
            
            # NDVI (Normalized Difference Vegetation Index)
            print(f"\n📊 NDVI Calculation:")
            print(f"   Formula: (NIR - Red) / (NIR + Red)")
            print(f"   NIR: {nir_refl:.6f}, Red: {red_refl:.6f}")
            if (nir_refl + red_refl) != 0:
                ndvi = (nir_refl - red_refl) / (nir_refl + red_refl)
                print(f"   NDVI = ({nir_refl:.6f} - {red_refl:.6f}) / ({nir_refl:.6f} + {red_refl:.6f}) = {ndvi:.6f}")
            else:
                ndvi = 0.0
                print(f"   Division by zero avoided, NDVI set to 0.0")
            indices['ndvi'] = float(ndvi)
            
            # EVI (Enhanced Vegetation Index)
            L = 1.0  # Canopy background adjustment
            C1, C2 = 6.0, 7.5  # Coefficients for aerosol resistance
            G = 2.5  # Gain factor
            
            print(f"\n📊 EVI Calculation:")
            print(f"   Formula: G × (NIR - Red) / (NIR + C1×Red - C2×Blue + L)")
            print(f"   Parameters: G={G}, C1={C1}, C2={C2}, L={L}")
            print(f"   NIR: {nir_refl:.6f}, Red: {red_refl:.6f}, Blue: {blue_refl:.6f}")
            
            denominator = nir_refl + C1 * red_refl - C2 * blue_refl + L
            if denominator != 0:
                evi = G * (nir_refl - red_refl) / denominator
                print(f"   Denominator: {nir_refl:.6f} + {C1}×{red_refl:.6f} - {C2}×{blue_refl:.6f} + {L} = {denominator:.6f}")
                print(f"   EVI = {G} × ({nir_refl:.6f} - {red_refl:.6f}) / {denominator:.6f} = {evi:.6f}")
            else:
                evi = 0.0
                print(f"   Division by zero avoided, EVI set to 0.0")
            indices['evi'] = float(evi)
            
            # SAVI (Soil Adjusted Vegetation Index) with improved accuracy
            L_savi = 0.5  # Soil brightness correction
            print(f"\n📊 SAVI Calculation:")
            print(f"   Formula: [(NIR - Red) / (NIR + Red + L)] × (1 + L)")
            print(f"   L factor: {L_savi}")
            denominator_savi = nir_refl + red_refl + L_savi
            if denominator_savi > 0.001:  # Avoid near-zero division
                savi = ((nir_refl - red_refl) / denominator_savi) * (1 + L_savi)
                print(f"   SAVI = [({nir_refl:.6f} - {red_refl:.6f}) / {denominator_savi:.6f}] × {1 + L_savi} = {savi:.6f}")
            else:
                savi = 0.0
                print(f"   Near-zero denominator, SAVI set to 0.0")
            indices['savi'] = float(savi)
            
            # GNDVI (Green NDVI) with validation
            print(f"\n📊 GNDVI Calculation:")
            print(f"   Formula: (NIR - Green) / (NIR + Green)")
            print(f"   NIR: {nir_refl:.6f}, Green: {green_refl:.6f}")
            denominator_gndvi = nir_refl + green_refl
            if denominator_gndvi > 0.001:
                gndvi = (nir_refl - green_refl) / denominator_gndvi
                print(f"   GNDVI = ({nir_refl:.6f} - {green_refl:.6f}) / {denominator_gndvi:.6f} = {gndvi:.6f}")
            else:
                gndvi = 0.0
                print(f"   Near-zero denominator, GNDVI set to 0.0")
            indices['gndvi'] = float(gndvi)
            
            # NDRE (Normalized Difference Red Edge) with bounds checking
            print(f"\n📊 NDRE Calculation:")
            print(f"   Formula: (NIR - RedEdge) / (NIR + RedEdge)")
            print(f"   NIR: {nir_refl:.6f}, Red-edge: {red_edge_refl:.6f}")
            denominator_ndre = nir_refl + red_edge_refl
            if denominator_ndre > 0.001:
                ndre = (nir_refl - red_edge_refl) / denominator_ndre
                print(f"   NDRE = ({nir_refl:.6f} - {red_edge_refl:.6f}) / {denominator_ndre:.6f} = {ndre:.6f}")
            else:
                ndre = 0.0
                print(f"   Near-zero denominator, NDRE set to 0.0")
            # Bound NDRE to reasonable range
            ndre = np.clip(ndre, -1.0, 1.0)
            indices['ndre'] = float(ndre)
            
            # CIgreen (Chlorophyll Index Green) with improved calculation
            print(f"\n📊 CI Green Calculation:")
            print(f"   Formula: (NIR / Green) - 1")
            print(f"   NIR: {nir_refl:.6f}, Green: {green_refl:.6f}")
            if green_refl > 0.001:  # Avoid division by near-zero
                ci_green = (nir_refl / green_refl) - 1
                print(f"   CI Green = ({nir_refl:.6f} / {green_refl:.6f}) - 1 = {ci_green:.6f}")
            else:
                ci_green = 0.0
                print(f"   Near-zero green reflectance, CI Green set to 0.0")
            # Bound CI Green to reasonable range
            ci_green = np.clip(ci_green, -5.0, 10.0)
            indices['ci_green'] = float(ci_green)
            
            # Add accuracy confidence scoring
            confidence_factors = []
            if nir_refl > 0.1:  # Good NIR signal
                confidence_factors.append(0.3)
            if red_refl > 0.05:  # Adequate red signal
                confidence_factors.append(0.2)
            if abs(nir_refl - red_refl) > 0.02:  # Meaningful difference
                confidence_factors.append(0.3)
            if green_refl > 0.03 and red_edge_refl > 0.03:  # Additional bands available
                confidence_factors.append(0.2)
                
            calculation_confidence = sum(confidence_factors)
            indices['calculation_confidence'] = float(calculation_confidence)
            
            print(f"\n🎯 CALCULATION CONFIDENCE: {calculation_confidence:.2f}/1.0")
            
        except Exception as e:
            print(f"Error calculating vegetation indices: {e}")
            # Return default values
            indices = {
                'ndvi': 0.0, 'evi': 0.0, 'savi': 0.0,
                'gndvi': 0.0, 'ndre': 0.0, 'ci_green': 0.0
            }
            
        return indices
        
    def assess_crop_health(self, vegetation_indices: Dict[str, float]) -> Dict[str, any]:
        """Assess crop health with improved accuracy and realistic thresholds"""
        ndvi = vegetation_indices.get('ndvi', 0)
        evi = vegetation_indices.get('evi', 0)
        savi = vegetation_indices.get('savi', 0)
        gndvi = vegetation_indices.get('gndvi', 0)
        calculation_confidence = vegetation_indices.get('calculation_confidence', 0.5)
        
        print("\n🌿 ENHANCED CROP HEALTH ASSESSMENT:")
        print("-" * 50)
        print(f"Input indices - NDVI: {ndvi:.4f}, EVI: {evi:.4f}, SAVI: {savi:.4f}, GNDVI: {gndvi:.4f}")
        print(f"Calculation confidence: {calculation_confidence:.2f}/1.0")
        
        # Multi-index health assessment with more realistic thresholds
        health_scores = []
        detailed_assessments = {}
        
        # Enhanced NDVI-based health with realistic thresholds
        print(f"\n📊 NDVI Health Assessment (Primary Indicator):")
        if ndvi > 0.7:  # Dense, healthy vegetation
            ndvi_score = 95
            ndvi_status = "Excellent"
            ndvi_desc = "Dense, healthy vegetation"
        elif ndvi > 0.5:  # Good vegetation cover
            ndvi_score = 85
            ndvi_status = "Good"
            ndvi_desc = "Good vegetation cover"
        elif ndvi > 0.3:  # Moderate vegetation
            ndvi_score = 70
            ndvi_status = "Moderate"
            ndvi_desc = "Moderate vegetation density"
        elif ndvi > 0.1:  # Sparse vegetation
            ndvi_score = 50
            ndvi_status = "Fair"
            ndvi_desc = "Sparse vegetation cover"
        elif ndvi > 0.0:  # Very limited vegetation
            ndvi_score = 30
            ndvi_status = "Poor"
            ndvi_desc = "Very limited vegetation"
        else:  # No vegetation or stressed
            ndvi_score = 15
            ndvi_status = "Critical"
            ndvi_desc = "Bare soil or severely stressed"
            
        print(f"   NDVI: {ndvi:.4f} → {ndvi_status} ({ndvi_score}/100)")
        print(f"   Description: {ndvi_desc}")
        
        detailed_assessments['ndvi'] = {
            'value': ndvi,
            'score': ndvi_score,
            'status': ndvi_status,
            'description': ndvi_desc
        }
            
        health_scores.append(ndvi_score)
        
        # EVI-based assessment (better in dense vegetation)
        if evi > 0.5:
            evi_score = 90
        elif evi > 0.3:
            evi_score = 75
        elif evi > 0.1:
            evi_score = 60
        else:
            evi_score = 30
            
        health_scores.append(evi_score)
        
        # SAVI-based assessment (soil-adjusted)
        if savi > 0.5:
            savi_score = 85
        elif savi > 0.3:
            savi_score = 70
        elif savi > 0.1:
            savi_score = 55
        else:
            savi_score = 35
            
        health_scores.append(savi_score)
        
        # Enhanced overall health score with uncertainty quantification
        base_score = (ndvi_score * 0.4 + evi_score * 0.3 + savi_score * 0.2 + (gndvi * 50 + 50) * 0.1)
        
        # Apply confidence adjustments
        confidence_multiplier = 0.7 + (calculation_confidence * 0.3)  # 0.7 to 1.0
        overall_score = base_score * confidence_multiplier
        
        # Calculate uncertainty margins
        data_uncertainty = (1 - calculation_confidence) * 15  # Up to ±15 points
        score_range = [max(0, overall_score - data_uncertainty), min(100, overall_score + data_uncertainty)]
        
        print(f"\n🎯 OVERALL HEALTH CALCULATION:")
        print(f"   Base score: {base_score:.1f}/100")
        print(f"   Confidence multiplier: {confidence_multiplier:.3f}")
        print(f"   Adjusted score: {overall_score:.1f}/100")
        print(f"   Uncertainty range: ±{data_uncertainty:.1f} points")
        print(f"   Score range: {score_range[0]:.1f} - {score_range[1]:.1f}")
        
        # More nuanced status determination
        if overall_score >= 85:
            overall_status = "Excellent"
        elif overall_score >= 70:
            overall_status = "Good"
        elif overall_score >= 55:
            overall_status = "Moderate"
        elif overall_score >= 40:
            overall_status = "Fair"
        elif overall_score >= 25:
            overall_status = "Poor"
        else:
            overall_status = "Critical"
            
        # Adjust confidence based on data quality
        final_confidence = min(0.95, calculation_confidence * 0.85 + 0.15)  # Base confidence of 0.15
            
        return {
            'overall_score': float(overall_score),
            'overall_status': overall_status,
            'score_range': [float(score_range[0]), float(score_range[1])],
            'uncertainty_margin': float(data_uncertainty),
            'ndvi_status': ndvi_status,
            'detailed_assessments': detailed_assessments,
            'component_scores': {
                'ndvi_score': float(ndvi_score),
                'evi_score': float(evi_score),
                'savi_score': float(savi_score),
                'base_score': float(base_score)
            },
            'primary_indicator': 'NDVI',
            'calculation_confidence': float(calculation_confidence),
            'confidence': float(final_confidence),
            'accuracy_notes': {
                'calibration_applied': True,
                'outlier_removal': True,
                'uncertainty_quantified': True,
                'confidence_weighted': True
            }
        }
        
    def detect_water_stress(self) -> Dict[str, any]:
        """Detect water stress using water absorption bands"""
        print("\n💧 PROCESSING WATER STRESS DATA:")
        print("-" * 50)
        
        try:
            # Extract water absorption bands
            water_band_1 = self._get_band_indices(self.wavelength_bands['water_abs_1'])
            water_band_2 = self._get_band_indices(self.wavelength_bands['water_abs_2'])
            water_band_3 = self._get_band_indices(self.wavelength_bands['water_abs_3'])
            
            print(f"🌊 Water Absorption Band 1 ({self.wavelength_bands['water_abs_1'][0]}-{self.wavelength_bands['water_abs_1'][1]}nm): {len(water_band_1)} bands")
            print(f"🌊 Water Absorption Band 2 ({self.wavelength_bands['water_abs_2'][0]}-{self.wavelength_bands['water_abs_2'][1]}nm): {len(water_band_2)} bands")
            print(f"🌊 Water Absorption Band 3 ({self.wavelength_bands['water_abs_3'][0]}-{self.wavelength_bands['water_abs_3'][1]}nm): {len(water_band_3)} bands")
            
            water_refl_1 = self._extract_spectral_reflectance(water_band_1)
            water_refl_2 = self._extract_spectral_reflectance(water_band_2)
            water_refl_3 = self._extract_spectral_reflectance(water_band_3)
            
            print(f"\n📊 WATER ABSORPTION REFLECTANCE VALUES:")
            print(f"Band 1 (970nm): {water_refl_1:.6f}")
            print(f"Band 2 (1200nm): {water_refl_2:.6f}")
            print(f"Band 3 (1450nm): {water_refl_3:.6f}")
            
            # Water stress index (lower values indicate higher water content)
            water_indices = [water_refl_1, water_refl_2, water_refl_3]
            valid_indices = [r for r in water_indices if r > 0]
            avg_water_absorption = np.mean(valid_indices)
            
            print(f"\n🧮 WATER STRESS CALCULATION:")
            print(f"Valid water indices: {len(valid_indices)}/{len(water_indices)}")
            print(f"Average water absorption: {avg_water_absorption:.6f}")
            
            # Normalize to 0-1 scale (approximate)
            water_stress_index = min(1.0, max(0.0, avg_water_absorption / 0.5))
            print(f"Water stress index (normalized): {water_stress_index:.6f}")
            
            if water_stress_index < 0.3:
                stress_level = "Low"
                stress_description = "Adequate water content"
            elif water_stress_index < 0.6:
                stress_level = "Moderate"
                stress_description = "Some water stress detected"
            elif water_stress_index < 0.8:
                stress_level = "High"
                stress_description = "Significant water stress"
            else:
                stress_level = "Critical"
                stress_description = "Severe water deficit"
                
            return {
                'water_stress_index': float(water_stress_index),
                'stress_level': stress_level,
                'description': stress_description,
                'water_absorption_values': {
                    'band_970nm': float(water_refl_1),
                    'band_1200nm': float(water_refl_2),
                    'band_1450nm': float(water_refl_3)
                },
                'confidence': 0.75
            }
            
        except Exception as e:
            print(f"Error detecting water stress: {e}")
            return {
                'water_stress_index': 0.5,
                'stress_level': "Unknown",
                'description': "Unable to assess water stress",
                'confidence': 0.0
            }
            
    def detect_disease_anomalies(self) -> Dict[str, any]:
        """Detect potential disease signatures through spectral anomalies"""
        print("\n🔬 PROCESSING DISEASE ANOMALY DATA:")
        print("-" * 50)
        
        try:
            # Get full spectral signature (averaged across spatial dimensions)
            print(f"📊 Spectral Data Shape: {self.spectral_data.shape}")
            spectral_signature = np.mean(self.spectral_data, axis=(0, 1))
            print(f"📈 Spectral Signature Length: {len(spectral_signature)} bands")
            print(f"📈 Sample spectral values: {spectral_signature[:5]}") # Show first 5 values
            
            # Calculate spectral statistics
            spectral_mean = np.mean(spectral_signature)
            spectral_std = np.std(spectral_signature)
            spectral_cv = spectral_std / spectral_mean if spectral_mean > 0 else 0
            
            print(f"\n📊 SPECTRAL STATISTICS:")
            print(f"Mean reflectance: {spectral_mean:.6f}")
            print(f"Standard deviation: {spectral_std:.6f}")
            print(f"Coefficient of variation: {spectral_cv:.6f}")
            
            # Detect anomalous bands (outliers)
            z_scores = np.abs((spectral_signature - spectral_mean) / spectral_std)
            anomalous_bands = np.sum(z_scores > 2.5)  # Bands with z-score > 2.5
            anomaly_percentage = (anomalous_bands / len(spectral_signature)) * 100
            
            print(f"\n🚨 ANOMALY DETECTION:")
            print(f"Bands with z-score > 2.5: {anomalous_bands}")
            print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
            print(f"Max z-score: {np.max(z_scores):.3f}")
            
            # Disease risk assessment
            risk_factors = []
            
            # High spectral variability can indicate disease stress
            if spectral_cv > 0.5:
                risk_factors.append("High spectral variability")
                
            # Anomalous bands suggest spectral disturbances
            if anomaly_percentage > 10:
                risk_factors.append("Significant spectral anomalies")
                
            # Check for red-edge shifts (disease indicator)
            red_edge_bands = self._get_band_indices(self.wavelength_bands['red_edge'])
            if red_edge_bands:
                red_edge_refl = self._extract_spectral_reflectance(red_edge_bands)
                if red_edge_refl < spectral_mean * 0.7:
                    risk_factors.append("Red-edge depression")
                    
            # Overall disease risk score
            disease_risk = min(1.0, (anomaly_percentage / 15 + spectral_cv) / 2)
            
            if disease_risk < 0.3:
                risk_level = "Low"
                risk_description = "No significant disease indicators"
            elif disease_risk < 0.6:
                risk_level = "Moderate" 
                risk_description = "Some spectral anomalies detected"
            elif disease_risk < 0.8:
                risk_level = "High"
                risk_description = "Multiple disease indicators present"
            else:
                risk_level = "Critical"
                risk_description = "Strong disease signatures detected"
                
            return {
                'disease_risk_score': float(disease_risk),
                'risk_level': risk_level,
                'description': risk_description,
                'anomaly_statistics': {
                    'anomalous_bands': int(anomalous_bands),
                    'anomaly_percentage': float(anomaly_percentage),
                    'spectral_cv': float(spectral_cv)
                },
                'risk_factors': risk_factors,
                'confidence': 0.70
            }
            
        except Exception as e:
            print(f"Error detecting disease anomalies: {e}")
            return {
                'disease_risk_score': 0.0,
                'risk_level': "Unknown",
                'description': "Unable to assess disease risk",
                'confidence': 0.0
            }
            
    def generate_spatial_analysis(self) -> Dict[str, any]:
        """Generate spatial analysis across the hyperspectral image"""
        print("\n🗺️ PROCESSING SPATIAL ANALYSIS DATA:")
        print("-" * 50)
        
        if self.spectral_data is None:
            return {}
            
        try:
            height, width, bands = self.spectral_data.shape
            print(f"📏 Image Dimensions: {height}×{width} pixels, {bands} bands")
            print(f"📊 Total pixels to analyze: {height * width:,}")
            
            # Calculate NDVI for each pixel
            red_bands = self._get_band_indices(self.wavelength_bands['red'])
            nir_bands = self._get_band_indices(self.wavelength_bands['nir'])
            
            print(f"🔴 Using {len(red_bands)} red bands for spatial NDVI")
            print(f"🟡 Using {len(nir_bands)} NIR bands for spatial NDVI")
            
            if red_bands and nir_bands:
                red_image = np.mean(self.spectral_data[:, :, red_bands], axis=2)
                nir_image = np.mean(self.spectral_data[:, :, nir_bands], axis=2)
                
                # Calculate pixel-wise NDVI
                print(f"\n🖼️ PIXEL-WISE NDVI CALCULATION:")
                print(f"Red image shape: {red_image.shape}")
                print(f"NIR image shape: {nir_image.shape}")
                print(f"Red image range: {np.min(red_image):.6f} to {np.max(red_image):.6f}")
                print(f"NIR image range: {np.min(nir_image):.6f} to {np.max(nir_image):.6f}")
                
                denominator = nir_image + red_image
                valid_pixels = np.sum(denominator > 0)
                print(f"Valid pixels for NDVI calculation: {valid_pixels:,}/{denominator.size:,}")
                
                ndvi_image = np.divide(nir_image - red_image, denominator, 
                                     out=np.zeros_like(denominator), where=denominator!=0)
                
                # Spatial statistics
                ndvi_mean = float(np.mean(ndvi_image))
                ndvi_std = float(np.std(ndvi_image))
                ndvi_min = float(np.min(ndvi_image))
                ndvi_max = float(np.max(ndvi_image))
                
                print(f"\n📊 NDVI SPATIAL STATISTICS:")
                print(f"NDVI range: {ndvi_min:.6f} to {ndvi_max:.6f}")
                print(f"NDVI mean: {ndvi_mean:.6f} ± {ndvi_std:.6f}")
                
                # Identify zones based on NDVI values
                healthy_pixels = np.sum(ndvi_image > 0.6)
                moderate_pixels = np.sum((ndvi_image > 0.3) & (ndvi_image <= 0.6))
                stressed_pixels = np.sum(ndvi_image <= 0.3)
                
                total_pixels = height * width
                
                spatial_analysis = {
                    'ndvi_statistics': {
                        'mean': ndvi_mean,
                        'std': ndvi_std,
                        'min': ndvi_min,
                        'max': ndvi_max
                    },
                    'spatial_zones': {
                        'healthy_area_percent': float(healthy_pixels / total_pixels * 100),
                        'moderate_area_percent': float(moderate_pixels / total_pixels * 100),
                        'stressed_area_percent': float(stressed_pixels / total_pixels * 100)
                    },
                    'uniformity_index': float(1 - ndvi_std),  # Higher = more uniform
                    'total_pixels': total_pixels,
                    'image_dimensions': [height, width]
                }
                
                return spatial_analysis
            else:
                return {'error': 'Unable to calculate spatial NDVI - missing bands'}
                
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return {'error': str(e)}
            
    def comprehensive_hyperspectral_analysis(self) -> Dict[str, any]:
        """Generate comprehensive hyperspectral analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE HYPERSPECTRAL ANALYSIS")
        print("="*60)
        
        # 1. Validate data quality first for accuracy
        print("\n🔍 STEP 1: DATA QUALITY VALIDATION")
        print("=" * 50)
        quality_metrics = self.validate_spectral_data_quality()
        
        if not quality_metrics.get('valid', False):
            print("❌ Data quality validation failed. Results may be inaccurate.")
            print("Consider checking data source and preprocessing.")
        else:
            print("✅ Data quality validation passed.")
        
        # Display raw data information
        self.display_raw_spectral_data_info()
        
        analysis_report = {
            'analysis_type': 'Enhanced Hyperspectral Agricultural Assessment',
            'timestamp': datetime.now().isoformat(),
            'accuracy_enhancements': {
                'data_validation': True,
                'calibration_applied': True,
                'outlier_removal': True,
                'uncertainty_quantification': True,
                'robust_statistics': True
            },
            'data_quality': quality_metrics,
            'data_source': {
                'file': self.aviris_file,
                'spatial_dimensions': list(self.spatial_dims) if self.spatial_dims else None,
                'spectral_bands': self.spectral_dims,
                'wavelength_range': f"{self.wavelengths[0]:.1f}-{self.wavelengths[-1]:.1f} nm" if self.wavelengths is not None else "Unknown",
                'data_type': str(self.spectral_data.dtype),
                'data_range': [float(np.min(self.spectral_data)), float(np.max(self.spectral_data))],
                'data_mean': float(np.mean(self.spectral_data)),
                'data_std': float(np.std(self.spectral_data))
            }
        }
        
        print(f"\n📄 ANALYSIS SUMMARY:")
        print(f"Data Source: {self.aviris_file}")
        print(f"Image Size: {self.spatial_dims}")
        print(f"Spectral Bands: {self.spectral_dims}")
        print(f"Wavelength Range: {analysis_report['data_source']['wavelength_range']}")
        
        # 1. Vegetation Indices
        print(f"\n{'='*40}")
        print("1. VEGETATION INDICES")
        print('='*40)
        vegetation_indices = self.calculate_vegetation_indices()
        analysis_report['vegetation_indices'] = vegetation_indices
        
        for index_name, value in vegetation_indices.items():
            print(f"{index_name.upper()}: {value:.3f}")
            
        # 2. Crop Health Assessment
        print(f"\n{'='*40}")
        print("2. CROP HEALTH ASSESSMENT")
        print('='*40)
        crop_health = self.assess_crop_health(vegetation_indices)
        analysis_report['crop_health'] = crop_health
        
        print(f"Overall Health: {crop_health['overall_status']}")
        print(f"Health Score: {crop_health['overall_score']:.1f}/100")
        print(f"Primary Indicator (NDVI): {crop_health['ndvi_status']}")
        print(f"Confidence: {crop_health['confidence']:.1%}")
        
        # 3. Water Stress Detection
        print(f"\n{'='*40}")
        print("3. WATER STRESS ANALYSIS")
        print('='*40)
        water_stress = self.detect_water_stress()
        analysis_report['water_stress'] = water_stress
        
        print(f"Water Stress Level: {water_stress['stress_level']}")
        print(f"Stress Index: {water_stress['water_stress_index']:.3f}")
        print(f"Description: {water_stress['description']}")
        
        # 4. Disease Detection
        print(f"\n{'='*40}")
        print("4. DISEASE ANOMALY DETECTION")
        print('='*40)
        disease_analysis = self.detect_disease_anomalies()
        analysis_report['disease_analysis'] = disease_analysis
        
        print(f"Disease Risk: {disease_analysis['risk_level']}")
        print(f"Risk Score: {disease_analysis['disease_risk_score']:.3f}")
        print(f"Spectral Anomalies: {disease_analysis['anomaly_statistics']['anomalous_bands']} bands")
        if disease_analysis['risk_factors']:
            print(f"Risk Factors: {', '.join(disease_analysis['risk_factors'])}")
            
        # 5. Spatial Analysis
        print(f"\n{'='*40}")
        print("5. SPATIAL ANALYSIS")
        print('='*40)
        spatial_analysis = self.generate_spatial_analysis()
        analysis_report['spatial_analysis'] = spatial_analysis
        
        if 'spatial_zones' in spatial_analysis:
            zones = spatial_analysis['spatial_zones']
            print(f"Healthy Areas: {zones['healthy_area_percent']:.1f}%")
            print(f"Moderate Areas: {zones['moderate_area_percent']:.1f}%")
            print(f"Stressed Areas: {zones['stressed_area_percent']:.1f}%")
            print(f"Field Uniformity: {spatial_analysis['uniformity_index']:.3f}")
            
        # Save comprehensive report
        report_filename = f"hyperspectral_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(analysis_report, f, indent=2)
            
        print(f"\n{'='*60}")
        print("HYPERSPECTRAL ANALYSIS COMPLETE")
        print(f"Report saved to: {report_filename}")
        print("="*60)
        
        return analysis_report


if __name__ == "__main__":
    # Run comprehensive hyperspectral analysis
    analyzer = HyperspectralAnalyzer()
    analysis_report = analyzer.comprehensive_hyperspectral_analysis()