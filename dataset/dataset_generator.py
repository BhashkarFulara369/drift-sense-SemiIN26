# #!/usr/bin/env python3
# """
# ================================================================================
# DRIFT-SENSE: HYBRID PHYSICAL SEM & FABRICATION SIMULATOR (v3.0.0-AMAT-INDUSTRIAL)
# Applied Materials | Semicon India Hackathon 2026
# ================================================================================
# Physical Calibration & Dataset Specifications:
#   1. Both Reference and Search images are standardized to 1000 x 1000 pixels.
#   2. Reference Image: High magnification (1 nm/px, 1000 nm x 1000 nm FOV).
#      Target pattern occupies 1000 x 1000 pixels in Reference.
#   3. Search Image: Lower magnification (10 nm/px, 10,000 nm x 10,000 nm FOV).
#      Same target pattern occupies approximately 100 x 100 pixels in Search.
#   4. Exact 10x physical scale ratio between Reference and Search sampling.
#   5. Anisotropic 2D Affine Transformation (independent scale_x, scale_y & rotation).
#   6. Sub-pixel Analytic Ground Truth & Displacement Vector Field (DVF) Bilinear Warp.
#   7. Canonical DRAM & FinFET Semiconductor Layout Engines.
#   8. Controlled 4-Tier Difficulty Progression (EASY, MEDIUM, HARD, EXTREME).
# """

# import os
# import sys
# import csv
# import math
# import json
# import time
# import argparse
# from typing import Tuple, List, Dict, Any, Optional
# from dataclasses import dataclass, asdict

# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter


# GENERATOR_VERSION = "3.0.0-AMAT-INDUSTRIAL"

# # Physical Dimension Standards
# REFERENCE_SIZE_PX = 1000        # Reference image resolution (1000 x 1000 px)
# SEARCH_SIZE_PX = 1000           # Search image resolution (1000 x 1000 px)
# PIXEL_SIZE_REF_NM = 1.0         # 1.0 nm / pixel
# PIXEL_SIZE_SEARCH_NM = 10.0     # 10.0 nm / pixel
# SCALE_FACTOR = 10               # 10x physical magnification ratio
# REF_FOV_IN_MASTER = 1000        # 1000 nm = 1000 master pixels
# SEARCH_FOV_IN_MASTER = 10000    # 10,000 nm = 10,000 master pixels
# TARGET_SEARCH_SIZE_PX = 100.0   # 1000 master px / 10 = 100 px in Search


# # ==============================================================================
# # PARAMETERS & DATACLASSES
# # ==============================================================================

# @dataclass
# class PhysicsParams:
#     beam_energy_keV: float             # Accelerating voltage E_0 [0.5 .. 3.0 keV]
#     se_alpha: float                    # Seiler SE-II yield coefficient [0.28 .. 0.50]
#     psf_sigma_ref: float               # Primary beam spot blur base (Ref) [0.4 .. 0.8 nm]
#     psf_sigma_srch: float              # Primary beam spot blur base (Search) [1.0 .. 2.2 nm]
#     psf_sigma_y: float                 # Beam PSF blur Y std (Astigmatism) [0.8 .. 2.8 nm]
#     astigmatism_angle_deg: float       # Astigmatism beam axis rotation [0 .. 360 deg]
#     dwell_time_ref: float              # High electron fluence (Ref dose) [150 .. 350 e-/px]
#     dwell_time_srch: float             # Low electron fluence (Search dose) [8 .. 110 e-/px]
#     readout_noise_std: float           # Thermal readout noise std [0.02 .. 0.08]
#     speckle_noise_std: float           # Multiplicative detector speckle noise std [0.01 .. 0.05]
#     salt_pepper_prob: float            # Impulse salt-and-pepper noise probability [0.0 .. 0.005]
#     detector_gain: float               # PMT detector gain multiplier [0.6 .. 1.8]
#     detector_sat_threshold: float      # PMT detector electron saturation level [0.7 .. 1.2]
#     gamma_exponent: float              # Sensor gamma curve exponent [0.6 .. 1.6]
#     baseline_black_level: float        # Sensor black level offset [10 .. 110]
#     fpn_strength: float                # Fixed-Pattern Detector Noise strength [0.005 .. 0.025]
#     charging_strength: float           # Specimen surface potential pool [0.0 .. 0.35]
#     vignetting_strength: float         # Lens detector collection angle attenuation [0.0 .. 0.25]
#     raster_drift_velocity: float       # Row-by-row stage drift velocity [0.01 .. 0.08 px/row]
#     raster_vibration_amp: float        # Piezoelectric vibration amplitude [0.2 .. 1.5 px]
#     scanline_jitter_prob: float        # Raster scanline timing jitter prob [0.005 .. 0.04]
#     scan_direction: int                # 0 for horizontal raster, 1 for vertical raster
#     # Non-linear Spatial Distortion Additions
#     enable_elastic_warp: bool          # Elastic rubber-sheet mesh warping
#     elastic_alpha: float               # Elastic perturbation magnitude
#     elastic_sigma: float               # Elastic perturbation smoothness scale
#     enable_barrel_distortion: bool     # Optical barrel lens distortion
#     barrel_k1: float                   # Radial distortion coefficient


# @dataclass
# class ProcessVariationParams:
#     ler_sigma_nm: float               # Palasantzas LER std [0.5 .. 2.5 nm]
#     ler_correlation_length_nm: float  # Spatial correlation length xi [10 .. 30 nm]
#     ler_hurst: float                  # Hurst roughness exponent H [0.5 .. 0.9]
#     cd_taper_pct: float               # Radial intra-die CD variation [0.0 .. 0.08]
#     cmp_dishing_strength: float       # CMP height erosion gradient [0.0 .. 0.2]
#     opc_corner_rounding_radius: int   # OPC corner rounding radius [2 .. 6 px]
#     etch_bias_nm: float               # Isotropic etch bias shift [-3 .. +3 nm]
#     enable_pattern_collapse: bool     # Capillary force resist line collapse


# @dataclass
# class BenchmarkAmbiguityParams:
#     difficulty: str                   # 'easy', 'medium', 'hard', 'extreme'
#     pure_array_probability: float     # Probability of generating pure periodic array
#     deceptive_candidate_count: int    # Number of near-identical false positive candidate traps
#     cell_similarity_pct: float        # Structural similarity of deceptive traps [0.80 .. 0.995]
#     repeated_defect_count: int        # Number of repeated defects scattered across canvas
#     max_rotation_deg: float           # Maximum search image rotation range [0.0 .. 5.0 deg]
#     scale_range_x: Tuple[float, float]# Horizontal scaling range [min_scale_x, max_scale_x]
#     scale_range_y: Tuple[float, float]# Vertical scaling range [min_scale_y, max_scale_y]
#     occluded_target_prob: float       # Chance of applying severe target occlusion


# @dataclass
# class LayoutSpec:
#     architecture: str                  # 'DRAM' or 'FinFET'
#     pitch_x: int                       # Horizontal pitch [140 .. 280 nm]
#     pitch_y: int                       # Vertical pitch [130 .. 260 nm]
#     line_w_x: int                      # Vertical line width [20 .. 52 nm]
#     line_w_y: int                      # Horizontal line width [22 .. 56 nm]
#     feature_size: int                  # Via/contact size [16 .. 38 nm]
#     base_gray: int                     # Substrate level [15 .. 90]
#     metal_gray: int                    # Metal line level [100 .. 200]
#     contact_gray: int                  # Contact via level [180 .. 255]
#     macro_width: int                   # Macro structure width [160 .. 320 nm]
#     # Augmentations (used on higher difficulty tiers)
#     dram_stagger_mode: str             # 'HEX', 'STAGGER_50', 'STAGGER_33'
#     dram_wave_amp: int                 # Bitline wave amplitude [0 .. 20 px]
#     dram_wave_freq1: float             # Primary bitline wave frequency
#     dram_wave_freq2: float             # Secondary harmonic wave frequency
#     dram_wave_phase: float             # Wave phase shift
#     dram_pad_angle: float              # Capacitor pad angle [-45 .. +45 deg]
#     finfet_cluster_size: int           # Grouped fins per cluster [2 .. 5]


# # ==============================================================================
# # PHYSICAL SEM RENDERING ENGINE (PHYSICS-INSPIRED DUAL-CHANNEL MODEL)
# # ==============================================================================

# class SEMPhysicsEngine:
#     """
#     Physics-inspired SEM rendering pipeline modeling electron interaction volume blur,
#     Seiler topographic secondary electron yield, stage drift, astigmatism, and detector response.
#     """

#     MATERIAL_PROPERTIES = {
#         0: {"name": "OXIDE_DIELECTRIC",  "gray": 25,  "yield": 0.65, "height_nm": 0.0},
#         1: {"name": "SILICON_SUBSTRATE","gray": 45,  "yield": 1.00, "height_nm": 0.0},
#         2: {"name": "POLY_SILICON",      "gray": 110, "yield": 1.25, "height_nm": 45.0},
#         3: {"name": "METAL_INTERCONNECT","gray": 175, "yield": 1.85, "height_nm": 55.0},
#         4: {"name": "CONTACT_VIA",       "gray": 220, "yield": 2.10, "height_nm": 70.0}
#     }

#     @staticmethod
#     def calculate_kanaya_okayama_psf(beam_energy_keV: float, base_sigma_nm: float) -> float:
#         """Approximates electron interaction volume scattering radius based on Kanaya-Okayama range."""
#         r_ko_um = 0.0276 * 28.085 * (beam_energy_keV ** 1.67) / (2.33 * (14.0 ** 0.899))
#         r_ko_nm = r_ko_um * 1000.0
#         return base_sigma_nm + 0.015 * r_ko_nm

#     @staticmethod
#     def synthesize_palasantzas_fft_ler(length: int, sigma_ler: float, xi: float = 15.0, hurst: float = 0.7) -> np.ndarray:
#         """Synthesizes 1D Line-Edge Roughness in spatial frequency domain via Palasantzas PSD."""
#         if sigma_ler <= 0.0 or length < 4:
#             return np.zeros(length, dtype=np.float32)

#         freqs = np.fft.fftfreq(length)
#         psd = (2.0 * (sigma_ler ** 2) * xi) / ((1.0 + (2.0 * np.pi * freqs * xi) ** 2) ** (hurst + 0.5))
#         psd[0] = 0.0

#         random_phase = np.exp(1j * np.random.uniform(0, 2.0 * np.pi, length))
#         complex_spectrum = np.sqrt(np.maximum(psd, 0.0)) * random_phase

#         roughness = np.fft.ifft(complex_spectrum).real.astype(np.float32)
#         std_r = np.std(roughness)
#         if std_r > 1e-6:
#             roughness *= (sigma_ler / std_r)
#         return roughness

#     @classmethod
#     def apply_edge_specific_ler(cls, canvas: np.ndarray, height_map: np.ndarray, proc: ProcessVariationParams) -> Tuple[np.ndarray, np.ndarray]:
#         """Applies Palasantzas FFT LER along feature sidewall boundaries."""
#         if proc.ler_sigma_nm <= 0.0:
#             return canvas, height_map

#         rows, cols = canvas.shape
#         edge_mask = (cv2.Sobel(canvas, cv2.CV_32F, 1, 1, ksize=3) > 10.0).astype(np.float32)

#         rough_x = cls.synthesize_palasantzas_fft_ler(cols, proc.ler_sigma_nm, proc.ler_correlation_length_nm, proc.ler_hurst)
#         rough_y = cls.synthesize_palasantzas_fft_ler(rows, proc.ler_sigma_nm, proc.ler_correlation_length_nm, proc.ler_hurst)

#         grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
#         map_x = (grid_x + rough_x[None, :] * edge_mask).astype(np.float32)
#         map_y = (grid_y + rough_y[:, None] * edge_mask).astype(np.float32)

#         warped_canvas = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         warped_height = cv2.remap(height_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

#         return warped_canvas, warped_height

#     @classmethod
#     def apply_seiler_topographic_yield(cls, height_map_nm: np.ndarray, mat_map: np.ndarray, alpha: float = 0.38) -> np.ndarray:
#         """Evaluates physical Seiler surface-normal yield on 3D Topography Height Map Z(x,y)."""
#         dz_dx = cv2.Sobel(height_map_nm, cv2.CV_64F, 1, 0, ksize=3) / 8.0
#         dz_dy = cv2.Sobel(height_map_nm, cv2.CV_64F, 0, 1, ksize=3) / 8.0
#         sec_theta = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)

#         base_yield = np.zeros_like(height_map_nm, dtype=np.float64)
#         for mat_id, prop in cls.MATERIAL_PROPERTIES.items():
#             base_yield[mat_map == mat_id] = prop["yield"]

#         base_yield[base_yield == 0.0] = 1.0

#         se_yield = base_yield * (1.0 + alpha * (sec_theta - 1.0))
#         return se_yield.astype(np.float32)

#     @staticmethod
#     def apply_charging_and_vignetting(img_float: np.ndarray, charging: float, vignetting: float) -> np.ndarray:
#         """Models specimen surface potential charging pool + lens collection angle vignetting."""
#         h, w = img_float.shape
#         gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

#         r_sq = ((gx - w / 2.0) ** 2 + (gy - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)
#         vignette_mask = 1.0 - vignetting * r_sq

#         if charging > 0.0:
#             cx, cy = np.random.uniform(0.2, 0.8) * w, np.random.uniform(0.2, 0.8) * h
#             charge_pool = charging * np.exp(-((gx - cx)**2 + (gy - cy)**2) / (2.0 * (0.35 * w)**2))
#         else:
#             charge_pool = 0.0

#         return np.clip(img_float * vignette_mask + charge_pool, 0.0, 10.0)

#     @staticmethod
#     def apply_anisotropic_astigmatism_blur(img_float: np.ndarray, sigma_x: float, sigma_y: float, angle_deg: float) -> np.ndarray:
#         """Simulates directional beam astigmatism via anisotropic rotated 2D Gaussian kernel."""
#         ksize_x = max(5, int(6 * sigma_x) | 1)
#         ksize_y = max(5, int(6 * sigma_y) | 1)
#         ksize = max(ksize_x, ksize_y)

#         kx = cv2.getGaussianKernel(ksize, sigma_x)
#         ky = cv2.getGaussianKernel(ksize, sigma_y)
#         kernel = ky @ kx.T

#         if abs(angle_deg) > 1e-3:
#             M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle_deg, 1.0)
#             kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
#             kernel /= np.maximum(np.sum(kernel), 1e-6)

#         return cv2.filter2D(img_float, -1, kernel)

#     @staticmethod
#     def apply_everhart_thornley_pmt_response(
#         img_float: np.ndarray, gain: float, sat_level: float, fpn_strength: float, gamma: float, black_level: float
#     ) -> np.ndarray:
#         """Models Everhart-Thornley Photomultiplier Tube (PMT) collector response, gamma curve, and FPN."""
#         norm_signal = img_float / np.maximum(np.max(img_float), 1e-5)
#         gamma_corrected = np.power(norm_signal, gamma)
#         saturated = gain * (gamma_corrected / (1.0 + gamma_corrected / max(0.1, sat_level)))

#         h, w = img_float.shape
#         fpn_grid = np.random.normal(1.0, fpn_strength, (h, w)).astype(np.float32)

#         gained_output = saturated * fpn_grid * 255.0 + black_level
#         return np.clip(gained_output, 0.0, 255.0)

#     @staticmethod
#     def apply_row_by_row_raster_stage_drift(img_float: np.ndarray, drift_vel: float, vib_amp: float) -> np.ndarray:
#         """Simulates row-by-row time-series stage drift and piezoelectric vibration during raster scan."""
#         if drift_vel <= 0.0 and vib_amp <= 0.0:
#             return img_float

#         h, w = img_float.shape
#         out = img_float.copy()
#         y_indices = np.arange(h, dtype=np.float32)

#         dx = drift_vel * y_indices + vib_amp * np.sin(2.0 * np.pi * y_indices / 40.0)

#         grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), y_indices)
#         map_x = (grid_x + dx[:, None]).astype(np.float32)

#         return cv2.remap(out, map_x, grid_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

#     @staticmethod
#     def apply_scanline_jitter(img_float: np.ndarray, prob: float, scan_dir: int) -> np.ndarray:
#         """Simulates raster scanline displacement artifacts."""
#         if prob <= 0.0:
#             return img_float
#         out = img_float.copy()
#         h, w = out.shape
#         if scan_dir == 0:  # Horizontal raster
#             for y in range(h):
#                 if np.random.rand() < prob:
#                     shift = np.random.randint(-5, 6)
#                     out[y, :] = np.roll(out[y, :], shift)
#         else:  # Vertical raster
#             for x in range(w):
#                 if np.random.rand() < prob:
#                     shift = np.random.randint(-5, 6)
#                     out[:, x] = np.roll(out[:, x], shift)
#         return out

#     @staticmethod
#     def apply_elastic_warp(img_float: np.ndarray, alpha: float, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Applies 2D elastic rubber-sheet deformation map for non-linear physical drift."""
#         h, w = img_float.shape
#         dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

#         grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#         map_x = (grid_x + dx).astype(np.float32)
#         map_y = (grid_y + dy).astype(np.float32)

#         warped = cv2.remap(img_float, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         return warped, map_x, map_y

#     @staticmethod
#     def apply_barrel_distortion(img_float: np.ndarray, k1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Applies optical radial lens barrel distortion."""
#         h, w = img_float.shape
#         cx, cy = w / 2.0, h / 2.0

#         grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#         x_norm = (grid_x - cx) / cx
#         y_norm = (grid_y - cy) / cy
#         r_sq = x_norm**2 + y_norm**2

#         distort_factor = 1.0 + k1 * r_sq
#         map_x = (cx + x_norm * distort_factor * cx).astype(np.float32)
#         map_y = (cy + y_norm * distort_factor * cy).astype(np.float32)

#         warped = cv2.remap(img_float, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         return warped, map_x, map_y

#     @classmethod
#     def process_sem_response(
#         cls, 
#         clean_raster: np.ndarray, 
#         height_map: np.ndarray,
#         mat_map: np.ndarray,
#         phys: PhysicsParams, 
#         proc: ProcessVariationParams, 
#         is_search: bool = False
#     ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
#         """Full SEM simulation pipeline with non-linear warp map tracking."""
#         map_x, map_y = None, None

#         # 1. Edge-Specific Palasantzas FFT LER
#         ler_raster, height_ler = cls.apply_edge_specific_ler(clean_raster, height_map, proc)

#         # 2. Seiler Topographic SE Yield Z(x,y)
#         se_yield = cls.apply_seiler_topographic_yield(height_ler, mat_map, alpha=phys.se_alpha)
#         img_float = (ler_raster.astype(np.float32) / 255.0) * se_yield

#         # 3. Charging & Optical Vignetting
#         if is_search:
#             img_float = cls.apply_charging_and_vignetting(img_float, phys.charging_strength, phys.vignetting_strength)

#         # 4. Non-Linear Spatial Warping (Elastic & Barrel Distortion)
#         if is_search and phys.enable_elastic_warp:
#             img_float, map_x, map_y = cls.apply_elastic_warp(img_float, phys.elastic_alpha, phys.elastic_sigma)

#         if is_search and phys.enable_barrel_distortion:
#             img_float, map_x, map_y = cls.apply_barrel_distortion(img_float, phys.barrel_k1)

#         # 5. Kanaya-Okayama Beam Blur & Anisotropic Beam Blur
#         base_sigma = phys.psf_sigma_srch if is_search else phys.psf_sigma_ref
#         eff_sigma_x = cls.calculate_kanaya_okayama_psf(phys.beam_energy_keV, base_sigma)
#         eff_sigma_y = phys.psf_sigma_y if is_search else phys.psf_sigma_ref
#         blurred = cls.apply_anisotropic_astigmatism_blur(img_float, eff_sigma_x, eff_sigma_y, phys.astigmatism_angle_deg)

#         # 6. Row-by-Row Stage Drift
#         if is_search:
#             blurred = cls.apply_row_by_row_raster_stage_drift(blurred, phys.raster_drift_velocity, phys.raster_vibration_amp)

#         # 7. Dose-Dependent Poisson Shot Noise + Readout + Multiplicative Speckle & Salt/Pepper
#         dose = phys.dwell_time_srch if is_search else phys.dwell_time_ref
#         electron_counts = np.maximum(blurred, 1e-6) * dose
#         noisy_shot = np.random.poisson(electron_counts).astype(np.float32) / dose
        
#         speckle = np.random.normal(1.0, phys.speckle_noise_std, size=img_float.shape).astype(np.float32)
#         noisy_speckle = noisy_shot * speckle
        
#         readout_noise = np.random.normal(0, phys.readout_noise_std, size=img_float.shape).astype(np.float32)
#         combined_signal = np.clip(noisy_speckle + readout_noise, 0.0, 10.0)

#         if phys.salt_pepper_prob > 0.0:
#             sp_mask = np.random.rand(*combined_signal.shape)
#             combined_signal[sp_mask < (phys.salt_pepper_prob / 2.0)] = 0.0
#             combined_signal[sp_mask > (1.0 - phys.salt_pepper_prob / 2.0)] = 2.0

#         # 8. Everhart-Thornley PMT Detector Response
#         et_signal = cls.apply_everhart_thornley_pmt_response(
#             combined_signal, phys.detector_gain, phys.detector_sat_threshold, phys.fpn_strength, phys.gamma_exponent, phys.baseline_black_level
#         )

#         # 9. Raster Scanline Jitter
#         if is_search:
#             et_signal = cls.apply_scanline_jitter(et_signal, phys.scanline_jitter_prob, phys.scan_direction)

#         return np.clip(et_signal, 0, 255).astype(np.uint8), map_x, map_y


# # ==============================================================================
# # CANONICAL SEMICONDUCTOR CAD LAYOUT ENGINE (DRAM & FINFET)
# # ==============================================================================

# class LayoutEngine:

#     @staticmethod
#     def get_benchmark_ambiguity_preset(difficulty: str) -> BenchmarkAmbiguityParams:
#         d = difficulty.lower()
#         if d == "easy":
#             return BenchmarkAmbiguityParams(
#                 difficulty="easy",
#                 pure_array_probability=0.10,
#                 deceptive_candidate_count=0,
#                 cell_similarity_pct=0.70,
#                 repeated_defect_count=0,
#                 max_rotation_deg=0.0,
#                 scale_range_x=(1.00, 1.00),
#                 scale_range_y=(1.00, 1.00),
#                 occluded_target_prob=0.0
#             )
#         elif d == "medium":
#             return BenchmarkAmbiguityParams(
#                 difficulty="medium",
#                 pure_array_probability=0.40,
#                 deceptive_candidate_count=2,
#                 cell_similarity_pct=0.88,
#                 repeated_defect_count=4,
#                 max_rotation_deg=1.5,
#                 scale_range_x=(0.98, 1.02),
#                 scale_range_y=(0.98, 1.02),
#                 occluded_target_prob=0.05
#             )
#         elif d == "hard":
#             return BenchmarkAmbiguityParams(
#                 difficulty="hard",
#                 pure_array_probability=0.75,
#                 deceptive_candidate_count=5,
#                 cell_similarity_pct=0.96,
#                 repeated_defect_count=15,
#                 max_rotation_deg=3.0,
#                 scale_range_x=(0.96, 1.04),
#                 scale_range_y=(0.96, 1.04),
#                 occluded_target_prob=0.12
#             )
#         else:  # EXTREME
#             return BenchmarkAmbiguityParams(
#                 difficulty="extreme",
#                 pure_array_probability=0.95,
#                 deceptive_candidate_count=10,
#                 cell_similarity_pct=0.995,
#                 repeated_defect_count=30,
#                 max_rotation_deg=5.0,
#                 scale_range_x=(0.94, 1.06),
#                 scale_range_y=(0.94, 1.06),
#                 occluded_target_prob=0.20
#             )

#     @classmethod
#     def generate_random_spec(cls, arch_choice: str, difficulty: str = "medium") -> Tuple[LayoutSpec, ProcessVariationParams, PhysicsParams, BenchmarkAmbiguityParams]:
#         amb = cls.get_benchmark_ambiguity_preset(difficulty)

#         m_oxide = SEMPhysicsEngine.MATERIAL_PROPERTIES[0]
#         m_poly = SEMPhysicsEngine.MATERIAL_PROPERTIES[2]
#         m_via = SEMPhysicsEngine.MATERIAL_PROPERTIES[4]

#         is_hard_or_extreme = difficulty.lower() in ["hard", "extreme"]

#         spec = LayoutSpec(
#             architecture=arch_choice,
#             pitch_x=np.random.randint(160, 260),
#             pitch_y=np.random.randint(150, 250),
#             line_w_x=np.random.randint(24, 48),
#             line_w_y=np.random.randint(26, 50),
#             feature_size=np.random.randint(18, 36),
#             base_gray=m_oxide["gray"] + np.random.randint(-5, 10),
#             metal_gray=m_poly["gray"] + np.random.randint(-10, 15),
#             contact_gray=m_via["gray"] + np.random.randint(-10, 10),
#             macro_width=np.random.randint(180, 300),
#             dram_stagger_mode=np.random.choice(['STAGGER_50', 'HEX']) if is_hard_or_extreme else 'STAGGER_50',
#             dram_wave_amp=np.random.randint(4, 16) if is_hard_or_extreme else 0,
#             dram_wave_freq1=np.random.uniform(0.8, 1.8),
#             dram_wave_freq2=np.random.uniform(0.3, 0.9),
#             dram_wave_phase=np.random.uniform(0.0, 2.0 * np.pi),
#             dram_pad_angle=np.random.uniform(-30.0, 30.0) if is_hard_or_extreme else 0.0,
#             finfet_cluster_size=np.random.randint(2, 4)
#         )

#         proc = ProcessVariationParams(
#             ler_sigma_nm=np.random.uniform(0.5, 1.2) if difficulty == "easy" else np.random.uniform(1.2, 2.5),
#             ler_correlation_length_nm=np.random.uniform(12.0, 25.0),
#             ler_hurst=np.random.uniform(0.65, 0.85),
#             cd_taper_pct=np.random.uniform(0.0, 0.02) if difficulty == "easy" else np.random.uniform(0.02, 0.06),
#             cmp_dishing_strength=np.random.uniform(0.0, 0.05) if difficulty == "easy" else np.random.uniform(0.05, 0.18),
#             opc_corner_rounding_radius=2 if difficulty == "easy" else np.random.randint(2, 5),
#             etch_bias_nm=np.random.uniform(-1.0, 1.0) if difficulty == "easy" else np.random.uniform(-3.0, 3.0),
#             enable_pattern_collapse=True if is_hard_or_extreme else False
#         )

#         phys = PhysicsParams(
#             beam_energy_keV=np.random.uniform(1.0, 2.0),
#             se_alpha=np.random.uniform(0.32, 0.42),
#             psf_sigma_ref=np.random.uniform(0.4, 0.6),
#             psf_sigma_srch=np.random.uniform(1.0, 1.6) if difficulty == "easy" else np.random.uniform(1.4, 2.2),
#             psf_sigma_y=np.random.uniform(0.5, 0.7) if difficulty == "easy" else np.random.uniform(1.0, 2.4),
#             astigmatism_angle_deg=0.0 if difficulty == "easy" else np.random.uniform(0.0, 360.0),
#             dwell_time_ref=np.random.uniform(220.0, 320.0),
#             dwell_time_srch=np.random.uniform(120.0, 200.0) if difficulty == "easy" else (
#                 np.random.uniform(50.0, 100.0) if difficulty == "medium" else np.random.uniform(15.0, 50.0)
#             ),
#             readout_noise_std=0.02 if difficulty == "easy" else np.random.uniform(0.03, 0.06),
#             speckle_noise_std=0.01 if difficulty == "easy" else np.random.uniform(0.02, 0.04),
#             salt_pepper_prob=0.0 if not is_hard_or_extreme else np.random.uniform(0.0005, 0.003),
#             detector_gain=1.0 if difficulty == "easy" else np.random.uniform(0.8, 1.4),
#             detector_sat_threshold=1.0,
#             gamma_exponent=1.0 if difficulty == "easy" else np.random.uniform(0.8, 1.3),
#             baseline_black_level=30.0 if difficulty == "easy" else np.random.uniform(20.0, 60.0),
#             fpn_strength=0.005 if difficulty == "easy" else np.random.uniform(0.008, 0.02),
#             charging_strength=0.0 if difficulty == "easy" else np.random.uniform(0.05, 0.25),
#             vignetting_strength=0.0 if difficulty == "easy" else np.random.uniform(0.05, 0.18),
#             raster_drift_velocity=0.0 if difficulty == "easy" else np.random.uniform(0.01, 0.05),
#             raster_vibration_amp=0.0 if difficulty == "easy" else np.random.uniform(0.2, 1.0),
#             scanline_jitter_prob=0.0 if difficulty == "easy" else np.random.uniform(0.005, 0.02),
#             scan_direction=0,
#             enable_elastic_warp=True if difficulty == "extreme" else False,
#             elastic_alpha=np.random.uniform(2.0, 4.0),
#             elastic_sigma=np.random.uniform(15.0, 25.0),
#             enable_barrel_distortion=True if difficulty == "extreme" else False,
#             barrel_k1=np.random.uniform(1e-6, 2.5e-6)
#         )

#         return spec, proc, phys, amb

#     @classmethod
#     def render_dram_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec):
#         """Renders canonical DRAM architecture: Wordlines (horizontal) + Bitlines (vertical) + Capacitor Contacts."""
#         h, w = sub.shape

#         # 1. Horizontal Wordlines
#         for y in range(0, h, spec.pitch_y):
#             cv2.rectangle(sub, (0, y), (w, y + spec.line_w_y), color=spec.metal_gray, thickness=-1)
#             cv2.rectangle(sub_h, (0, y), (w, y + spec.line_w_y), color=55.0, thickness=-1)
#             cv2.rectangle(sub_m, (0, y), (w, y + spec.line_w_y), color=3, thickness=-1)

#         # 2. Vertical Bitlines (canonical straight or sinusoidal wave on hard mode)
#         if spec.dram_wave_amp > 0:
#             amp = spec.dram_wave_amp
#             f1, f2, ph = spec.dram_wave_freq1, spec.dram_wave_freq2, spec.dram_wave_phase
#             for x in range(0, w, spec.pitch_x):
#                 pts = [(x + int(amp * np.sin(2 * np.pi * f1 * y / spec.pitch_y + ph) + 0.4 * amp * np.cos(2 * np.pi * f2 * y / spec.pitch_y)), y) for y in range(0, h, 25)]
#                 pts = np.array(pts, np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(sub, [pts], isClosed=False, color=spec.metal_gray + 20, thickness=spec.line_w_x)
#                 cv2.polylines(sub_h, [pts], isClosed=False, color=45.0, thickness=spec.line_w_x)
#                 cv2.polylines(sub_m, [pts], isClosed=False, color=2, thickness=spec.line_w_x)
#         else:
#             for x in range(0, w, spec.pitch_x):
#                 cv2.rectangle(sub, (x, 0), (x + spec.line_w_x, h), color=spec.metal_gray + 20, thickness=-1)
#                 cv2.rectangle(sub_h, (x, 0), (x + spec.line_w_x, h), color=45.0, thickness=-1)
#                 cv2.rectangle(sub_m, (x, 0), (x + spec.line_w_x, h), color=2, thickness=-1)

#         # 3. Capacitor Storage Node Contacts (at Wordline/Bitline intersections)
#         rx, ry = spec.feature_size, max(6, int(spec.feature_size * 0.6))
#         row_idx = 0
#         for y in range(spec.pitch_y // 2, h, spec.pitch_y):
#             off_x = (spec.pitch_x // 2) if (row_idx % 2 == 1) else 0
#             for x in range(off_x, w, spec.pitch_x):
#                 if abs(spec.dram_pad_angle) > 1.0:
#                     cv2.ellipse(sub, (x, y), (rx, ry), angle=spec.dram_pad_angle, startAngle=0, endAngle=360, color=spec.contact_gray - 35, thickness=-1)
#                     cv2.circle(sub, (x, y), radius=max(3, rx // 4), color=spec.contact_gray, thickness=-1)
#                     cv2.ellipse(sub_h, (x, y), (rx, ry), angle=spec.dram_pad_angle, startAngle=0, endAngle=360, color=65.0, thickness=-1)
#                     cv2.circle(sub_h, (x, y), radius=max(3, rx // 4), color=70.0, thickness=-1)
#                     cv2.ellipse(sub_m, (x, y), (rx, ry), angle=spec.dram_pad_angle, startAngle=0, endAngle=360, color=4, thickness=-1)
#                     cv2.circle(sub_m, (x, y), radius=max(3, rx // 4), color=4, thickness=-1)
#                 else:
#                     cv2.circle(sub, (x, y), radius=rx, color=spec.contact_gray, thickness=-1)
#                     cv2.circle(sub_h, (x, y), radius=rx, color=70.0, thickness=-1)
#                     cv2.circle(sub_m, (x, y), radius=rx, color=4, thickness=-1)
#             row_idx += 1

#     @classmethod
#     def render_finfet_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec):
#         """Renders canonical FinFET architecture: Vertical parallel fins + Horizontal transistor gates + Contacts."""
#         h, w = sub.shape

#         # 1. Parallel Vertical Fins
#         x = 0
#         while x < w:
#             for c in range(spec.finfet_cluster_size):
#                 fx = x + c * spec.pitch_x
#                 if fx < w:
#                     cv2.rectangle(sub, (fx, 0), (fx + spec.line_w_x, h), color=spec.metal_gray - 20, thickness=-1)
#                     cv2.rectangle(sub_h, (fx, 0), (fx + spec.line_w_x, h), color=50.0, thickness=-1)
#                     cv2.rectangle(sub_m, (fx, 0), (fx + spec.line_w_x, h), color=2, thickness=-1)
#             x += spec.finfet_cluster_size * spec.pitch_x + 80

#         # 2. Transistor Gates (Horizontal Bars across fins)
#         for y in range(0, h, spec.pitch_y):
#             gate_w = spec.line_w_y
#             cv2.rectangle(sub, (0, y), (w, y + gate_w), color=spec.metal_gray + 35, thickness=-1)
#             cv2.rectangle(sub_h, (0, y), (w, y + gate_w), color=60.0, thickness=-1)
#             cv2.rectangle(sub_m, (0, y), (w, y + gate_w), color=3, thickness=-1)

#             # Source/Drain Contacts
#             for gx in range(spec.pitch_x // 2, w, spec.pitch_x * 2):
#                 cv2.rectangle(sub, (gx - 8, y - 6), (gx + 8, y + gate_w + 6), color=spec.contact_gray, thickness=-1)
#                 cv2.rectangle(sub_h, (gx - 8, y - 6), (gx + 8, y + gate_w + 6), color=70.0, thickness=-1)
#                 cv2.rectangle(sub_m, (gx - 8, y - 6), (gx + 8, y + gate_w + 6), color=4, thickness=-1)

#     @classmethod
#     def render_macro_shape(cls, canvas: np.ndarray, height_map: np.ndarray, mat_map: np.ndarray, cx: int, cy: int, mtype: str, width: int, metal_gray: int, contact_gray: int):
#         """Renders macro alignment structures across intensity, 3D height, and material maps."""
#         w = width
#         patch = np.zeros((w * 2, w * 2), dtype=np.uint8)
#         patch_h = np.zeros((w * 2, w * 2), dtype=np.float32)
#         patch_m = np.zeros((w * 2, w * 2), dtype=np.uint8)
#         pcx, pcy = w, w

#         if mtype == 'SQUARE':
#             cv2.rectangle(patch, (pcx - w//2, pcy - w//2), (pcx + w//2, pcy + w//2), metal_gray + 25, -1)
#             cv2.rectangle(patch, (pcx - w//4, pcy - w//4), (pcx + w//4, pcy + w//4), contact_gray, -1)
#             cv2.rectangle(patch_h, (pcx - w//2, pcy - w//2), (pcx + w//2, pcy + w//2), 60.0, -1)
#             cv2.rectangle(patch_h, (pcx - w//4, pcy - w//4), (pcx + w//4, pcy + w//4), 70.0, -1)
#             cv2.rectangle(patch_m, (pcx - w//2, pcy - w//2), (pcx + w//2, pcy + w//2), 3, -1)
#             cv2.rectangle(patch_m, (pcx - w//4, pcy - w//4), (pcx + w//4, pcy + w//4), 4, -1)
#         elif mtype == 'CROSS':
#             cv2.rectangle(patch, (pcx - w, pcy - w//4), (pcx + w, pcy + w//4), metal_gray + 25, -1)
#             cv2.rectangle(patch, (pcx - w//4, pcy - w), (pcx + w//4, pcy + w), metal_gray + 25, -1)
#             cv2.circle(patch, (pcx, pcy), w//4, contact_gray, -1)

#             cv2.rectangle(patch_h, (pcx - w, pcy - w//4), (pcx + w, pcy + w//4), 60.0, -1)
#             cv2.rectangle(patch_h, (pcx - w//4, pcy - w), (pcx + w//4, pcy + w), 60.0, -1)
#             cv2.circle(patch_h, (pcx, pcy), w//4, 70.0, -1)

#             cv2.rectangle(patch_m, (pcx - w, pcy - w//4), (pcx + w, pcy + w//4), 3, -1)
#             cv2.rectangle(patch_m, (pcx - w//4, pcy - w), (pcx + w//4, pcy + w), 3, -1)
#             cv2.circle(patch_m, (pcx, pcy), w//4, 4, -1)
#         else:  # POWER_RAIL
#             cv2.rectangle(patch, (0, pcy - w//3), (w * 2, pcy + w//3), contact_gray - 20, -1)
#             cv2.rectangle(patch, (pcx - w//3, pcy - w//2), (pcx + w//3, pcy + w//2), 255, -1)

#             cv2.rectangle(patch_h, (0, pcy - w//3), (w * 2, pcy + w//3), 65.0, -1)
#             cv2.rectangle(patch_h, (pcx - w//3, pcy - w//2), (pcx + w//3, pcy + w//2), 75.0, -1)

#             cv2.rectangle(patch_m, (0, pcy - w//3), (w * 2, pcy + w//3), 3, -1)
#             cv2.rectangle(patch_m, (pcx - w//3, pcy - w//2), (pcx + w//3, pcy + w//2), 4, -1)

#         rot_k = np.random.choice([0, 1, 2, 3])
#         patch = np.rot90(patch, k=rot_k)
#         patch_h = np.rot90(patch_h, k=rot_k)
#         patch_m = np.rot90(patch_m, k=rot_k)

#         y1, y2 = max(0, cy - w), min(canvas.shape[0], cy + w)
#         x1, x2 = max(0, cx - w), min(canvas.shape[1], cx + w)
#         py1, py2 = max(0, w - (cy - y1)), min(w * 2, w + (y2 - cy))
#         px1, px2 = max(0, w - (cx - x1)), min(w * 2, w + (x2 - cx))

#         mask = patch[py1:py2, px1:px2] > 0
#         canvas[y1:y2, x1:x2][mask] = patch[py1:py2, px1:px2][mask]
#         height_map[y1:y2, x1:x2][mask] = patch_h[py1:py2, px1:px2][mask]
#         mat_map[y1:y2, x1:x2][mask] = patch_m[py1:py2, px1:px2][mask]

#     @classmethod
#     def apply_capillary_pattern_collapse(cls, canvas: np.ndarray, height_map: np.ndarray, mat_map: np.ndarray, enable: bool):
#         """Simulates lithographic resist capillary collapse / line toppling."""
#         if not enable or np.random.rand() > 0.4:
#             return

#         h, w = canvas.shape
#         num_collapses = np.random.randint(3, 8)
#         for _ in range(num_collapses):
#             cx, cy = np.random.randint(500, w - 500), np.random.randint(500, h - 500)
#             rad = np.random.randint(20, 60)
            
#             y1, y2 = max(0, cy - rad), min(h, cy + rad)
#             x1, x2 = max(0, cx - rad), min(w, cx + rad)

#             sub_c = canvas[y1:y2, x1:x2]
#             sub_h = height_map[y1:y2, x1:x2]
            
#             shift_dir = np.random.choice([0, 1])
#             if shift_dir == 0:
#                 sub_c[:] = np.roll(sub_c, shift=8, axis=1)
#                 sub_h[:] = np.roll(sub_h, shift=8, axis=1)
#             else:
#                 sub_c[:] = np.roll(sub_c, shift=8, axis=0)
#                 sub_h[:] = np.roll(sub_h, shift=8, axis=0)

#     @classmethod
#     def inject_repeated_defects(cls, canvas: np.ndarray, height_map: np.ndarray, mat_map: np.ndarray, count: int):
#         """Injects defects (missing vias, line bridging, particle contamination) across maps."""
#         if count <= 0:
#             return
#         h, w = canvas.shape
#         defects = ['MISSING_VIA', 'LINE_BRIDGING', 'LINE_CUT_OPEN', 'PARTICLE_CONTAMINATION']

#         for _ in range(count):
#             dtype = np.random.choice(defects)
#             px, py = np.random.randint(200, w - 200), np.random.randint(200, h - 200)

#             if dtype == 'MISSING_VIA':
#                 cv2.circle(canvas, (px, py), 14, color=30, thickness=-1)
#                 cv2.circle(height_map, (px, py), 14, color=0.0, thickness=-1)
#                 cv2.circle(mat_map, (px, py), 14, color=0, thickness=-1)
#             elif dtype == 'LINE_BRIDGING':
#                 cv2.line(canvas, (px, py - 18), (px, py + 18), color=195, thickness=8)
#                 cv2.line(height_map, (px, py - 18), (px, py + 18), color=55.0, thickness=8)
#                 cv2.line(mat_map, (px, py - 18), (px, py + 18), color=3, thickness=8)
#             elif dtype == 'LINE_CUT_OPEN':
#                 cv2.line(canvas, (px - 14, py), (px + 14, py), color=25, thickness=9)
#                 cv2.line(height_map, (px - 14, py), (px + 14, py), color=0.0, thickness=9)
#                 cv2.line(mat_map, (px - 14, py), (px + 14, py), color=0, thickness=9)
#             else:  # PARTICLE_CONTAMINATION
#                 cv2.circle(canvas, (px, py), 12, color=255, thickness=-1)
#                 cv2.circle(height_map, (px, py), 12, color=80.0, thickness=-1)
#                 cv2.circle(mat_map, (px, py), 12, color=4, thickness=-1)

#     @classmethod
#     def render_canvas(cls, width: int, height: int, spec: LayoutSpec, amb: BenchmarkAmbiguityParams, proc: ProcessVariationParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
#         canvas = np.full((height, width), fill_value=spec.base_gray, dtype=np.uint8)
#         height_map = np.zeros((height, width), dtype=np.float32)
#         mat_map = np.zeros((height, width), dtype=np.uint8)
#         landmarks = []

#         if spec.architecture == "FinFET":
#             cls.render_finfet_canonical(canvas, height_map, mat_map, spec)
#         else:
#             cls.render_dram_canonical(canvas, height_map, mat_map, spec)

#         is_pure_array = (np.random.rand() < amb.pure_array_probability)

#         if not is_pure_array:
#             macro_types = ['SQUARE', 'CROSS', 'POWER_RAIL']
#             macro_pitch = 2400
#             for my in range(macro_pitch // 2, height, macro_pitch):
#                 for mx in range(macro_pitch // 2, width, macro_pitch):
#                     mtype = np.random.choice(macro_types)
#                     cls.render_macro_shape(canvas, height_map, mat_map, mx, my, mtype, spec.macro_width, spec.metal_gray, spec.contact_gray)
#                     landmarks.append((mx, my))

#         cls.apply_capillary_pattern_collapse(canvas, height_map, mat_map, proc.enable_pattern_collapse)
#         cls.inject_repeated_defects(canvas, height_map, mat_map, amb.repeated_defect_count)

#         kernel_opc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel_opc)

#         return canvas, height_map, mat_map, landmarks


# # ==============================================================================
# # SUB-PIXEL ANISOTROPIC GT TRANSFORMER & STRUCTURAL UNIQUENESS METRICS
# # ==============================================================================

# def build_anisotropic_affine_matrix(center: Tuple[float, float], angle_deg: float, scale_x: float, scale_y: float) -> np.ndarray:
#     """
#     Constructs exact 2x3 affine matrix for non-uniform anisotropic scaling (scale_x, scale_y)
#     combined with rotation angle_deg around specific pivot center point (cx, cy).
#     """
#     cx, cy = center
#     rad = math.radians(angle_deg)
#     cos_a, sin_a = math.cos(rad), math.sin(rad)

#     # Combined Affine Transformation Matrix A = R * S
#     m00 = scale_x * cos_a
#     m01 = -scale_y * sin_a
#     m10 = scale_x * sin_a
#     m11 = scale_y * cos_a

#     tx = cx - (m00 * cx + m01 * cy)
#     ty = cy - (m10 * cx + m11 * cy)

#     return np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)


# def transform_ground_truth_subpixel(
#     gt_ref_center: Tuple[float, float],
#     affine_matrix: np.ndarray,
#     displacement_field_x: Optional[np.ndarray] = None,
#     displacement_field_y: Optional[np.ndarray] = None
# ) -> Tuple[float, float]:
#     """Computes exact sub-pixel GT coordinates through affine transform and non-linear DVF bilinear interpolation."""
#     # 1. Forward Affine Transformation
#     pt = np.array([gt_ref_center[0], gt_ref_center[1], 1.0], dtype=np.float64)
#     x_aff, y_aff = affine_matrix.dot(pt)[:2]

#     # 2. Sub-Pixel Bilinear Interpolation of Non-Linear DVF Maps
#     if displacement_field_x is not None and displacement_field_y is not None:
#         h, w = displacement_field_x.shape
#         x_clamped = np.clip(x_aff, 0.0, w - 1.001)
#         y_clamped = np.clip(y_aff, 0.0, h - 1.001)

#         x0, y0 = int(np.floor(x_clamped)), int(np.floor(y_clamped))
#         x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
#         dx, dy = x_clamped - x0, y_clamped - y0

#         final_x = (1.0 - dx) * (1.0 - dy) * displacement_field_x[y0, x0] + \
#                   dx * (1.0 - dy) * displacement_field_x[y0, x1] + \
#                   (1.0 - dx) * dy * displacement_field_x[y1, x0] + \
#                   dx * dy * displacement_field_x[y1, x1]

#         final_y = (1.0 - dx) * (1.0 - dy) * displacement_field_y[y0, x0] + \
#                   dx * (1.0 - dy) * displacement_field_y[y0, x1] + \
#                   (1.0 - dx) * dy * displacement_field_y[y1, x0] + \
#                   dx * dy * displacement_field_y[y1, x1]

#         return float(final_x), float(final_y)

#     return float(x_aff), float(y_aff)


# def compute_structural_uniqueness_index(ref_crop: np.ndarray, search_img: np.ndarray, gt_xy: Tuple[float, float]) -> float:
#     """Computes Local Structural Uniqueness Index S_uniq (1.0 = completely unique, 0.0 = perfectly ambiguous trap)."""
#     try:
#         # Scale down reference crop by 10x for template matching on search image scale
#         ref_small = cv2.resize(ref_crop, (100, 100), interpolation=cv2.INTER_AREA)
#         res = cv2.matchTemplate(search_img, ref_small, cv2.TM_CCOEFF_NORMED)
#         h, w = res.shape
#         gx = int(gt_xy[0] - ref_small.shape[1] / 2.0)
#         gy = int(gt_xy[1] - ref_small.shape[0] / 2.0)

#         gx = max(0, min(w - 1, gx))
#         gy = max(0, min(h - 1, gy))

#         primary_peak = float(res[gy, gx])

#         res_masked = res.copy()
#         cv2.circle(res_masked, (gx, gy), radius=30, color=-1.0, thickness=-1)
#         secondary_peak = float(np.max(res_masked))

#         uniqueness_score = float(max(0.0, 1.0 - (secondary_peak / max(primary_peak, 1e-5))))
#         return round(uniqueness_score, 4)
#     except Exception:
#         return 0.5000


# # ==============================================================================
# # MAIN SIMULATOR CLASS
# # ==============================================================================

# class SEMDatasetGenerator:

#     def __init__(self, output_dir: str = "./synthetic_sem_dataset", visualize: bool = False, difficulty: str = "medium", seed: int = 42):
#         self.output_dir = output_dir
#         self.visualize = visualize
#         self.difficulty = difficulty
#         self.seed = seed

#         np.random.seed(self.seed)

#         self.ref_dir = os.path.join(output_dir, "reference")
#         self.search_dir = os.path.join(output_dir, "search")
#         os.makedirs(self.ref_dir, exist_ok=True)
#         os.makedirs(self.search_dir, exist_ok=True)

#         if self.visualize:
#             self.preview_dir = os.path.join(output_dir, "previews")
#             os.makedirs(self.preview_dir, exist_ok=True)

#     def generate_single_sample(self, sample_id: str, arch_choice: str, difficulty: str = None) -> Dict[str, Any]:
#         diff_level = difficulty if difficulty else self.difficulty
#         master_w, master_h = 12000, 12000

#         spec, proc, phys, amb = LayoutEngine.generate_random_spec(arch_choice, difficulty=diff_level)
#         master_canvas, master_height, master_mat, landmarks = LayoutEngine.render_canvas(master_w, master_h, spec, amb, proc)

#         # 1. Search Region (10,000 nm x 10,000 nm FOV in master canvas)
#         search_fov_size = SEARCH_FOV_IN_MASTER  # 10,000 px
#         search_start_x = np.random.randint(0, master_w - search_fov_size)
#         search_start_y = np.random.randint(0, master_h - search_fov_size)

#         raw_search_region = master_canvas[
#             search_start_y : search_start_y + search_fov_size,
#             search_start_x : search_start_x + search_fov_size
#         ]
#         raw_search_height = master_height[
#             search_start_y : search_start_y + search_fov_size,
#             search_start_x : search_start_x + search_fov_size
#         ]
#         raw_search_mat = master_mat[
#             search_start_y : search_start_y + search_fov_size,
#             search_start_x : search_start_x + search_fov_size
#         ]

#         # 2. Nyquist Anti-Aliased Gaussian Downsampling (10000 px -> 1000 px @ 10 nm/px)
#         ds_factor = float(SCALE_FACTOR)
#         blur_sigma = max(0.8, ds_factor / 3.0)
#         search_prefiltered = cv2.GaussianBlur(raw_search_region, (7, 7), sigmaX=blur_sigma)
#         search_downsampled = cv2.resize(search_prefiltered, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)

#         search_height_ds = cv2.resize(raw_search_height, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)
#         search_mat_ds = cv2.resize(raw_search_mat, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_NEAREST)

#         # 3. Reference Crop (1000 nm x 1000 nm FOV in master canvas -> 1000 x 1000 px @ 1 nm/px)
#         ref_fov_in_master = REF_FOV_IN_MASTER  # 1000 px master

#         # Ground Truth Center position in Search Space (between 200.0 and 800.0 px)
#         gt_search_x = float(np.random.uniform(200.0, 800.0))
#         gt_search_y = float(np.random.uniform(200.0, 800.0))

#         # Map GT Search Position back to Master Canvas coordinates
#         ref_start_x = int(search_start_x + gt_search_x * 10.0 - ref_fov_in_master / 2.0)
#         ref_start_y = int(search_start_y + gt_search_y * 10.0 - ref_fov_in_master / 2.0)

#         # Clamp bounds within search FOV inside master canvas
#         ref_start_x = max(search_start_x, min(search_start_x + search_fov_size - ref_fov_in_master, ref_start_x))
#         ref_start_y = max(search_start_y, min(search_start_y + search_fov_size - ref_fov_in_master, ref_start_y))

#         raw_ref_crop = master_canvas[
#             ref_start_y : ref_start_y + ref_fov_in_master,
#             ref_start_x : ref_start_x + ref_fov_in_master
#         ]
#         raw_ref_height = master_height[
#             ref_start_y : ref_start_y + ref_fov_in_master,
#             ref_start_x : ref_start_x + ref_fov_in_master
#         ]
#         raw_ref_mat = master_mat[
#             ref_start_y : ref_start_y + ref_fov_in_master,
#             ref_start_x : ref_start_x + ref_fov_in_master
#         ]

#         # Exact unwarped GT Center location in Search FOV space (1000x1000 px)
#         unwarped_gt_x = ((ref_start_x + ref_fov_in_master / 2.0) - search_start_x) / 10.0
#         unwarped_gt_y = ((ref_start_y + ref_fov_in_master / 2.0) - search_start_y) / 10.0

#         # Occluded Target Hard Mode
#         if np.random.rand() < amb.occluded_target_prob:
#             cv2.circle(raw_ref_crop, (ref_fov_in_master // 2, ref_fov_in_master // 2), ref_fov_in_master // 4, color=15, thickness=-1)

#         # Reference image is natively 1000 x 1000 pixels (1 nm/px)
#         raw_ref_crop_final = raw_ref_crop
#         raw_ref_height_final = raw_ref_height
#         raw_ref_mat_final = raw_ref_mat

#         # 4. Kinematic Motion / Anisotropic Scale & Rotation
#         angle_deg = float(np.random.uniform(-amb.max_rotation_deg, amb.max_rotation_deg))
#         scale_fac_x = float(np.random.uniform(*amb.scale_range_x))
#         scale_fac_y = float(np.random.uniform(*amb.scale_range_y))

#         # Build true 2x3 Anisotropic Affine Transformation Matrix centered at (500.0, 500.0)
#         rot_matrix = build_anisotropic_affine_matrix((500.0, 500.0), angle_deg, scale_fac_x, scale_fac_y)

#         search_distorted = cv2.warpAffine(search_downsampled, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_height_dist = cv2.warpAffine(search_height_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_mat_dist = cv2.warpAffine(search_mat_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

#         # 5. SEM Response Simulation
#         ref_final, _, _ = SEMPhysicsEngine.process_sem_response(raw_ref_crop_final, raw_ref_height_final, raw_ref_mat_final, phys, proc, is_search=False)
#         search_final, map_x, map_y = SEMPhysicsEngine.process_sem_response(search_distorted, search_height_dist, search_mat_dist, phys, proc, is_search=True)

#         # 6. Analytic Sub-Pixel Ground Truth Coordinate Transformation
#         final_gt_x, final_gt_y = transform_ground_truth_subpixel(
#             (unwarped_gt_x, unwarped_gt_y), rot_matrix, map_x, map_y
#         )

#         # 7. Compute Structural Uniqueness Index S_uniq
#         uniqueness_index = compute_structural_uniqueness_index(ref_final, search_final, (final_gt_x, final_gt_y))

#         half_target_search_px = TARGET_SEARCH_SIZE_PX / 2.0  # 50.0 px
#         bbox = [
#             float(final_gt_x - half_target_search_px),
#             float(final_gt_y - half_target_search_px),
#             float(final_gt_x + half_target_search_px),
#             float(final_gt_y + half_target_search_px)
#         ]

#         ref_filename = f"{sample_id}.png"
#         search_filename = f"{sample_id}.png"

#         cv2.imwrite(os.path.join(self.ref_dir, ref_filename), ref_final)
#         cv2.imwrite(os.path.join(self.search_dir, search_filename), search_final)

#         if self.visualize:
#             preview_img = self.create_visual_preview(ref_final, search_final, bbox, (final_gt_x, final_gt_y), sample_id, spec.architecture, diff_level, uniqueness_index)
#             cv2.imwrite(os.path.join(self.preview_dir, f"{sample_id}_preview.png"), preview_img)

#         return {
#             "sample_id": sample_id,
#             "architecture": spec.architecture,
#             "difficulty": diff_level,
#             "reference_path": f"reference/{ref_filename}",
#             "search_path": f"search/{search_filename}",
#             "gt_center_x": float(round(final_gt_x, 4)),
#             "gt_center_y": float(round(final_gt_y, 4)),
#             "bbox_xmin": round(bbox[0], 4),
#             "bbox_ymin": round(bbox[1], 4),
#             "bbox_xmax": round(bbox[2], 4),
#             "bbox_ymax": round(bbox[3], 4),
#             "uniqueness_index": uniqueness_index,
#             "reference_width": REFERENCE_SIZE_PX,
#             "reference_height": REFERENCE_SIZE_PX,
#             "search_width": SEARCH_SIZE_PX,
#             "search_height": SEARCH_SIZE_PX,
#             "target_size_ref_px": REFERENCE_SIZE_PX,
#             "target_size_search_px": int(TARGET_SEARCH_SIZE_PX),
#             "deceptive_candidate_count": amb.deceptive_candidate_count,
#             "rotation_deg": float(round(angle_deg, 4)),
#             "scale_x": float(round(scale_fac_x, 4)),
#             "scale_y": float(round(scale_fac_y, 4)),
#             "pitch_x_nm": spec.pitch_x,
#             "pitch_y_nm": spec.pitch_y,
#             "beam_energy_keV": float(round(phys.beam_energy_keV, 3)),
#             "charging_strength": float(round(phys.charging_strength, 3)),
#             "ler_sigma_nm": float(round(proc.ler_sigma_nm, 3)),
#             "detector_gain": float(round(phys.detector_gain, 3)),
#             "gamma_exponent": float(round(phys.gamma_exponent, 3))
#         }

#     @staticmethod
#     def create_visual_preview(ref_img: np.ndarray, search_img: np.ndarray, bbox: List[float], center_pt: Tuple[float, float], sample_id: str, arch: str, diff: str, u_idx: float) -> np.ndarray:
#         ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
#         search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

#         x_min, y_min, x_max, y_max = [int(v) for v in bbox]
#         cv2.rectangle(search_rgb, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

#         cx, cy = int(center_pt[0]), int(center_pt[1])
#         cv2.drawMarker(search_rgb, (cx, cy), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

#         combined = np.hstack((ref_rgb, search_rgb))
#         cv2.putText(combined, f"Ref (1000x1000, 1nm/px) - {sample_id} [{arch}]", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#         cv2.putText(combined, f"Search (1000x1000, 10nm/px) [{diff.upper()}] S_uniq: {u_idx:.3f}", (1030, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         return combined

#     def batch_generate(self, num_pairs: int = 30, arch_selection: str = "ALL") -> List[Dict[str, Any]]:
#         print("=" * 80)
#         print(f"DRIFT-SENSE: GENERATING {num_pairs} BENCHMARK SAMPLES (v{GENERATOR_VERSION})")
#         print(f"Target Output Directory : {os.path.abspath(self.output_dir)}")
#         print(f"Selected Architecture   : {arch_selection}")
#         print(f"Base Difficulty Level   : {self.difficulty.upper()}")
#         print(f"Reproducibility Seed    : {self.seed}")
#         print("=" * 80)

#         official_archs = ["DRAM", "FinFET"]
#         manifest_data = []
#         start_time = time.time()

#         tier_schedule = ["easy"] * 8 + ["medium"] * 8 + ["hard"] * 8 + ["extreme"] * 6
#         if len(tier_schedule) < num_pairs:
#             tier_schedule.extend([self.difficulty] * (num_pairs - len(tier_schedule)))

#         for i in range(1, num_pairs + 1):
#             sample_id = f"sample_{i:03d}"

#             if arch_selection.upper() in ["DRAM", "FINFET"]:
#                 arch_choice = arch_selection.upper()
#             else:
#                 arch_choice = official_archs[(i - 1) % len(official_archs)]

#             sample_diff = tier_schedule[i - 1] if arch_selection == "ALL" else self.difficulty

#             metadata = self.generate_single_sample(sample_id, arch_choice=arch_choice, difficulty=sample_diff)
#             manifest_data.append(metadata)

#             print(f"[+] Sample {i:02d}/{num_pairs:02d} | ID: {sample_id} | Arch: {metadata['architecture']:6s} | Tier: {metadata['difficulty']:7s} | S_uniq: {metadata['uniqueness_index']:.3f} | GT: ({metadata['gt_center_x']:.2f}, {metadata['gt_center_y']:.2f})")

#         print("-" * 80)

#         # Export metadata.csv
#         csv_path = os.path.join(self.output_dir, "metadata.csv")
#         csv_fieldnames = list(manifest_data[0].keys())
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
#             writer.writeheader()
#             writer.writerows(manifest_data)

#         # Export metadata.json
#         metadata_json_path = os.path.join(self.output_dir, "metadata.json")
#         with open(metadata_json_path, "w") as f:
#             json.dump(manifest_data, f, indent=4)

#         # Export config.json & config.yaml
#         config_data = {
#             "generator_version": GENERATOR_VERSION,
#             "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "architecture_selection": arch_selection,
#             "num_pairs": num_pairs,
#             "difficulty": self.difficulty,
#             "seed": self.seed,
#             "reference_image_size": [1000, 1000],
#             "search_image_size": [1000, 1000],
#             "scale_factor": SCALE_FACTOR
#         }
#         with open(os.path.join(self.output_dir, "config.json"), "w") as f:
#             json.dump(config_data, f, indent=4)

#         with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
#             for k, v in config_data.items():
#                 f.write(f"{k}: {v}\n")

#         # Export dataset_statistics.json
#         arch_counts = {a: sum(1 for m in manifest_data if m['architecture'] == a) for a in official_archs}
#         tier_counts = {t: sum(1 for m in manifest_data if m['difficulty'] == t) for t in ["easy", "medium", "hard", "extreme"]}
#         stats_data = {
#             "architecture_counts": arch_counts,
#             "tier_counts": tier_counts,
#             "total_pairs": num_pairs,
#             "average_uniqueness_index": float(np.mean([m['uniqueness_index'] for m in manifest_data])),
#             "average_rotation_deg": float(np.mean([m['rotation_deg'] for m in manifest_data])),
#             "average_scale_x": float(np.mean([m['scale_x'] for m in manifest_data])),
#             "average_scale_y": float(np.mean([m['scale_y'] for m in manifest_data])),
#             "average_ler_sigma_nm": float(np.mean([m['ler_sigma_nm'] for m in manifest_data]))
#         }
#         with open(os.path.join(self.output_dir, "dataset_statistics.json"), "w") as f:
#             json.dump(stats_data, f, indent=4)

#         self.generate_readme()

#         elapsed = time.time() - start_time
#         print(f"[SUCCESS] Generation complete in {elapsed:.2f} seconds.")
#         print(f"[INFO] Metadata CSV exported to : {csv_path}")
#         print(f"[INFO] Metadata JSON exported to: {metadata_json_path}")
#         print("=" * 80)

#         return manifest_data

#     def generate_readme(self):
#         lines = [
#             f"# Drift-Sense Synthetic SEM Benchmark Dataset (v{GENERATOR_VERSION})",
#             "",
#             f"**Generated Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
#             f"**Base Difficulty Level:** `{self.difficulty.upper()}`",
#             f"**Seed:** `{self.seed}`",
#             "",
#             "## Physical Scale & Dimensions",
#             "- **Reference Image**: `1000 x 1000` pixels @ 1 nm/px ($1000\\text{{ nm}} \\times 1000\\text{{ nm}}$ FOV)",
#             "- **Search Image**: `1000 x 1000` pixels @ 10 nm/px ($10000\\text{{ nm}} \\times 10000\\text{{ nm}}$ FOV)",
#             "- **Physical Target Scale**: The $1000 \\times 1000$ px Reference target appears as a $\\sim 100 \\times 100$ px feature in Search image.",
#             "",
#             "## Dataset Structure",
#             "```text",
#             f"{os.path.basename(os.path.abspath(self.output_dir))}/",
#             "├── reference/                # High-Res Reference Target Crops (1000x1000 px)",
#             "├── search/                   # Wide Search Images (1000x1000 px)",
#             "├── previews/                 # Verification Previews w/ Target BBox",
#             "├── metadata.csv              # Primary ground truth annotations table",
#             "├── metadata.json             # Per-sample JSON metadata",
#             "├── dataset_statistics.json   # Aggregated dataset statistics",
#             "├── config.json / config.yaml # Generation parameters",
#             "└── README.md                 # Dataset documentation",
#             "```",
#             "",
#             "## Ground Truth Annotations",
#             "Ground truth target coordinates (`gt_center_x`, `gt_center_y`) specify exact sub-pixel centers in Search space with bilinear DVF precision.",
#             "The `uniqueness_index` ($S_{uniq}$) metric measures structural uniqueness (1.0 = unique target, 0.0 = ambiguous trap)."
#         ]
#         readme_path = os.path.join(self.output_dir, "README.md")
#         with open(readme_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(lines))


# # ==============================================================================
# # CLI ENTRY POINT
# # ==============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description=f"Drift-Sense Physical SEM Dataset Generator v{GENERATOR_VERSION}",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("--architecture", type=str, default="ALL", choices=["ALL", "DRAM", "FinFET"], help="Die architecture to generate.")
#     parser.add_argument("--num_pairs", "--num_samples", type=int, default=30, help="Number of image pairs to generate.")
#     parser.add_argument("--output_dir", type=str, default="./synthetic_sem_dataset", help="Output directory path.")
#     parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "extreme"], help="Benchmark degradation difficulty level.")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for exact reproducibility.")
#     parser.add_argument("--visualize", action="store_true", help="Generate side-by-side preview images with target bounding boxes.")

#     args = parser.parse_args()

#     generator = SEMDatasetGenerator(
#         output_dir=args.output_dir,
#         visualize=args.visualize,
#         difficulty=args.difficulty,
#         seed=args.seed
#     )
    
#     generator.batch_generate(num_pairs=args.num_pairs, arch_selection=args.architecture)


# if __name__ == "__main__":
#     main()






#!/usr/bin/env python3
"""
================================================================================
DRIFT-SENSE: HYBRID PHYSICAL SEM & FABRICATION SIMULATOR (v1.0)
Applied Materials | Semicon India Hackathon 2026 | TEAM - ShunyaVeer

================================================================================

Physical & Theoretical Foundations:
  1. Beam PSF & Scattering Range: Kanaya-Okayama (1972) primary electron range 
     scattering model for energy-dependent beam interaction volume blur.
  2. Topographic SE Yield: Seiler (1983) surface-normal SE-II yield model 
     delta(theta) = delta_0 * sec(theta)^alpha for sidewall edge-brightening.
  3. Sidewall Roughness: Palasantzas (1993) spatial frequency domain Fourier 
     power spectral density (PSD) model for Line-Edge Roughness (LER).
  4. Detector Response & Noise: Reimer (1998) Poisson electron shot noise, 
     Everhart-Thornley PMT collector gain, and Fixed-Pattern Noise (FPN).
  5. Resist Collapse: Namatsu et al. (1995) lithographic capillary force model
     for localized high-aspect-ratio feature toppling.

"""

# import os
# import sys
# import csv
# import hashlib
# import math
# import json
# import time
# import argparse
# from typing import Tuple, List, Dict, Any, Optional
# from dataclasses import dataclass, asdict, field

# import cv2
# import numpy as np
# from scipy.ndimage import gaussian_filter

# GENERATOR_VERSION = "10.0.0-AMAT-INDUSTRIAL"
# SCHEMA_VERSION = "1.0.0"

# # Physical Dimension Standards
# REFERENCE_SIZE_PX = 1000        # Reference image resolution (1000 x 1000 px)
# SEARCH_SIZE_PX = 1000           # Search image resolution (1000 x 1000 px)
# PIXEL_SIZE_REF_NM = 1.0         # 1.0 nm / pixel
# PIXEL_SIZE_SEARCH_NM = 10.0     # 10.0 nm / pixel
# SCALE_FACTOR = 10               # 10x physical magnification ratio
# REF_FOV_IN_MASTER_NM = 1000.0   # 1000 nm FOV
# SEARCH_FOV_IN_MASTER_NM = 10000.0 # 10,000 nm FOV


# # ==============================================================================
# # PARAMETERS & DATACLASSES
# # ==============================================================================

# @dataclass
# class PhysicsParams:
#     beam_energy_keV: float
#     se_alpha: float
#     psf_sigma_ref_nm: float
#     psf_sigma_srch_nm: float
#     psf_sigma_y_nm: float
#     astigmatism_angle_deg: float
#     dwell_time_ref: float
#     dwell_time_srch: float
#     readout_noise_std: float
#     speckle_noise_std: float
#     salt_pepper_prob: float
#     detector_gain: float
#     detector_sat_threshold: float
#     gamma_exponent: float
#     baseline_black_level: float
#     fpn_strength: float
#     charging_strength: float
#     vignetting_strength: float
#     raster_drift_velocity_px: float
#     raster_vibration_amp_px: float
#     scanline_jitter_prob: float
#     scan_direction: int # 0: Horizontal scanlines, 1: Vertical scanlines
#     enable_elastic_warp: bool
#     elastic_alpha_px: float
#     elastic_sigma_px: float
#     enable_barrel_distortion: bool
#     barrel_k1: float


# @dataclass
# class ProcessVariationParams:
#     ler_sigma_nm: float
#     ler_correlation_length_nm: float
#     ler_hurst: float
#     cd_taper_pct: float
#     cmp_dishing_strength: float
#     opc_corner_rounding_radius: int
#     etch_bias_nm: float
#     enable_pattern_collapse: bool


# @dataclass
# class BenchmarkAmbiguityParams:
#     difficulty: str
#     pure_array_probability: float
#     deceptive_candidate_count: int
#     cell_similarity_pct: float
#     repeated_defect_count: int
#     max_rotation_deg: float
#     scale_range_x: Tuple[float, float]
#     scale_range_y: Tuple[float, float]
#     occluded_target_prob: float


# @dataclass
# class LayoutSpec:
#     architecture: str
#     pitch_x_nm: int
#     pitch_y_nm: int
#     line_w_x_nm: int
#     line_w_y_nm: int
#     feature_size_nm: int
#     base_gray: int
#     metal_gray: int
#     contact_gray: int
#     macro_width_nm: int
#     dram_stagger_mode: str
#     dram_wave_amp_nm: int
#     dram_wave_freq1: float
#     dram_wave_freq2: float
#     dram_wave_phase: float
#     dram_pad_angle: float
#     finfet_cluster_size: int


# @dataclass
# class CandidateMetadata:
#     candidate_id: str
#     candidate_type: str # 'NATURAL_PERIODIC' or 'ADVERSARIAL_TRAP'
#     unwarped_center_x_search_px: float
#     unwarped_center_y_search_px: float
#     transformed_center_x_search_px: float
#     transformed_center_y_search_px: float
#     distance_from_gt_transformed_px: float
#     ssim_to_target_clean: float
#     ncc_to_target_clean: float


# # ==============================================================================
# # COORDINATE CONVERSION ENGINE
# # ==============================================================================

# class CoordinateTransformer:
#     """Explicit conversion routines between Master NM, Master PX, Ref PX, and Search PX."""

#     @staticmethod
#     def master_nm_to_px(val_nm: float) -> int:
#         return int(round(val_nm)) # 1.0 nm / pixel in master canvas

#     @staticmethod
#     def master_px_to_search_px(x_master: float, y_master: float, search_start_x: float, search_start_y: float) -> Tuple[float, float]:
#         x_search = (x_master - search_start_x) / 10.0
#         y_search = (y_master - search_start_y) / 10.0
#         return float(x_search), float(y_search)

#     @staticmethod
#     def search_px_to_master_px(x_search: float, y_search: float, search_start_x: float, search_start_y: float) -> Tuple[float, float]:
#         x_master = search_start_x + x_search * 10.0
#         y_master = search_start_y + y_search * 10.0
#         return float(x_master), float(y_master)


# # ==============================================================================
# # STRUCTURAL METRICS ENGINE (SSIM & NCC)
# # ==============================================================================

# def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
#     """Computes Structural Similarity Index (SSIM) between two patches."""
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)

#     mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
#     mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
#     sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
#     sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return float(np.mean(ssim_map))


# def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
#     """Computes Normalized Cross-Correlation (NCC) between two patches."""
#     i1 = img1.astype(np.float64) - np.mean(img1)
#     i2 = img2.astype(np.float64) - np.mean(img2)
#     denom = np.sqrt(np.sum(i1**2) * np.sum(i2**2))
#     if denom < 1e-8:
#         return 0.0
#     return float(np.sum(i1 * i2) / denom)


# # ==============================================================================
# # COMPOUND GEOMETRIC TRANSFORM ENGINE
# # ==============================================================================

# class CompoundTransformEngine:
#     """
#     Unified geometric transform engine providing continuous forward mapping F(x,y)
#     for points, polygon footprints, and inverse fixed-point remap grid generation.
#     """

#     @staticmethod
#     def build_anisotropic_affine_matrix(center: Tuple[float, float], angle_deg: float, scale_x: float, scale_y: float) -> np.ndarray:
#         cx, cy = center
#         rad = math.radians(angle_deg)
#         cos_a, sin_a = math.cos(rad), math.sin(rad)

#         m00 = scale_x * cos_a
#         m01 = -scale_y * sin_a
#         m10 = scale_x * sin_a
#         m11 = scale_y * cos_a

#         tx = cx - (m00 * cx + m01 * cy)
#         ty = cy - (m10 * cx + m11 * cy)

#         return np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)

#     @classmethod
#     def create_compound_warp_field(
#         cls, 
#         width: int, 
#         height: int, 
#         phys: PhysicsParams, 
#         rng: np.random.Generator
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        
#         disp_fwd_x = np.zeros((height, width), dtype=np.float32)
#         disp_fwd_y = np.zeros((height, width), dtype=np.float32)

#         if phys.enable_elastic_warp and phys.elastic_alpha_px > 0:
#             rand_x = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
#             rand_y = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
            
#             disp_fwd_x += gaussian_filter(rand_x, phys.elastic_sigma_px).astype(np.float32) * phys.elastic_alpha_px
#             disp_fwd_y += gaussian_filter(rand_y, phys.elastic_sigma_px).astype(np.float32) * phys.elastic_alpha_px

#         if phys.enable_barrel_distortion and phys.barrel_k1 > 0:
#             cx, cy = width / 2.0, height / 2.0
#             x_norm = (grid_x - cx) / cx
#             y_norm = (grid_y - cy) / cy
#             r_sq = x_norm**2 + y_norm**2
            
#             disp_fwd_x += ((grid_x - cx) * (phys.barrel_k1 * r_sq)).astype(np.float32)
#             disp_fwd_y += ((grid_y - cy) * (phys.barrel_k1 * r_sq)).astype(np.float32)

#         if phys.raster_drift_velocity_px > 0 or phys.raster_vibration_amp_px > 0:
#             indices = np.arange(height if phys.scan_direction == 0 else width, dtype=np.float32)
#             drift_val = phys.raster_drift_velocity_px * indices + \
#                         phys.raster_vibration_amp_px * np.sin(2.0 * np.pi * indices / 40.0).astype(np.float32)
            
#             if phys.scan_direction == 0:
#                 disp_fwd_x += np.tile(drift_val[:, None], (1, width))
#             else:
#                 disp_fwd_y += np.tile(drift_val[None, :], (height, 1))

#         if phys.scanline_jitter_prob > 0:
#             dim_size = height if phys.scan_direction == 0 else width
#             jitter = np.zeros(dim_size, dtype=np.float32)
#             mask = rng.random(dim_size) < phys.scanline_jitter_prob
#             jitter[mask] = rng.uniform(-3.0, 3.0, size=np.sum(mask)).astype(np.float32)
            
#             if phys.scan_direction == 0:
#                 disp_fwd_x += np.tile(jitter[:, None], (1, width))
#             else:
#                 disp_fwd_y += np.tile(jitter[None, :], (height, 1))

#         map_x = grid_x.copy()
#         map_y = grid_y.copy()

#         for _ in range(4):
#             curr_x = np.clip(map_x, 0, width - 1.001)
#             curr_y = np.clip(map_y, 0, height - 1.001)

#             x0 = np.floor(curr_x).astype(np.int32)
#             y0 = np.floor(curr_y).astype(np.int32)
#             x1 = np.minimum(x0 + 1, width - 1)
#             y1 = np.minimum(y0 + 1, height - 1)

#             dx = curr_x - x0
#             dy = curr_y - y0

#             interp_dx = (1 - dx) * (1 - dy) * disp_fwd_x[y0, x0] + \
#                         dx * (1 - dy) * disp_fwd_x[y0, x1] + \
#                         (1 - dx) * dy * disp_fwd_x[y1, x0] + \
#                         dx * dy * disp_fwd_x[y1, x1]

#             interp_dy = (1 - dx) * (1 - dy) * disp_fwd_y[y0, x0] + \
#                         dx * (1 - dy) * disp_fwd_y[y0, x1] + \
#                         (1 - dx) * dy * disp_fwd_y[y1, x0] + \
#                         dx * dy * disp_fwd_y[y1, x1]

#             map_x = grid_x - interp_dx
#             map_y = grid_y - interp_dy

#         return (
#             disp_fwd_x.astype(np.float32), 
#             disp_fwd_y.astype(np.float32), 
#             map_x.astype(np.float32), 
#             map_y.astype(np.float32)
#         )

#     @classmethod
#     def forward_point(
#         cls, 
#         pt_initial: Tuple[float, float], 
#         affine_matrix: np.ndarray, 
#         disp_fwd_x: np.ndarray, 
#         disp_fwd_y: np.ndarray
#     ) -> Tuple[float, float]:
#         pt_aff = np.array([pt_initial[0], pt_initial[1], 1.0], dtype=np.float64)
#         x_aff, y_aff = affine_matrix.dot(pt_aff)[:2]

#         h, w = disp_fwd_x.shape
#         x_c = np.clip(x_aff, 0.0, w - 1.001)
#         y_c = np.clip(y_aff, 0.0, h - 1.001)

#         x0, y0 = int(np.floor(x_c)), int(np.floor(y_c))
#         x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
#         dx, dy = x_c - x0, y_c - y0

#         dx_val = (1 - dx) * (1 - dy) * disp_fwd_x[y0, x0] + \
#                  dx * (1 - dy) * disp_fwd_x[y0, x1] + \
#                  (1 - dx) * dy * disp_fwd_x[y1, x0] + \
#                  dx * dy * disp_fwd_x[y1, x1]

#         dy_val = (1 - dx) * (1 - dy) * disp_fwd_y[y0, x0] + \
#                  dx * (1 - dy) * disp_fwd_y[y0, x1] + \
#                  (1 - dx) * dy * disp_fwd_y[y1, x0] + \
#                  dx * dy * disp_fwd_y[y1, x1]

#         return float(x_aff + dx_val), float(y_aff + dy_val)

#     @classmethod
#     def forward_polygon(
#         cls, 
#         polygon_pts: List[Tuple[float, float]], 
#         affine_matrix: np.ndarray, 
#         disp_fwd_x: np.ndarray, 
#         disp_fwd_y: np.ndarray
#     ) -> List[Tuple[float, float]]:
#         return [cls.forward_point(pt, affine_matrix, disp_fwd_x, disp_fwd_y) for pt in polygon_pts]

#     @classmethod
#     def test_sample_residual(
#         cls, 
#         rot_matrix: np.ndarray, 
#         disp_fwd_x: np.ndarray, 
#         disp_fwd_y: np.ndarray, 
#         map_x: np.ndarray, 
#         map_y: np.ndarray
#     ) -> float:
#         test_points = [(x, y) for x in [100.0, 250.0, 500.0, 750.0, 900.0] for y in [100.0, 250.0, 500.0, 750.0, 900.0]]
#         max_err = 0.0

#         for pt in test_points:
#             gt_x, gt_y = cls.forward_point(pt, rot_matrix, disp_fwd_x, disp_fwd_y)

#             delta_img = np.zeros((1000, 1000), dtype=np.float32)
#             ix, iy = int(round(pt[0])), int(round(pt[1]))
#             delta_img[iy, ix] = 1.0

#             warped_aff = cv2.warpAffine(delta_img, rot_matrix, (1000, 1000), flags=cv2.INTER_LINEAR)
#             warped_final = cv2.remap(warped_aff, map_x, map_y, interpolation=cv2.INTER_LINEAR)

#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(warped_final)
#             px, py = max_loc

#             if 1 <= px < 999 and 1 <= py < 999:
#                 dx = (warped_final[py, px + 1] - warped_final[py, px - 1]) / (2.0 * (2.0 * warped_final[py, px] - warped_final[py, px + 1] - warped_final[py, px - 1] + 1e-6))
#                 dy = (warped_final[py + 1, px] - warped_final[py - 1, px]) / (2.0 * (2.0 * warped_final[py, px] - warped_final[py + 1, px] - warped_final[py - 1, px] + 1e-6))
#                 meas_x = px + np.clip(dx, -0.5, 0.5)
#                 meas_y = py + np.clip(dy, -0.5, 0.5)
#             else:
#                 meas_x, meas_y = float(px), float(py)

#             err = math.sqrt((meas_x - gt_x)**2 + (meas_y - gt_y)**2)
#             max_err = max(max_err, err)

#         return float(max_err)


# # ==============================================================================
# # PHYSICAL SEM RENDERING ENGINE
# # ==============================================================================

# class SEMPhysicsEngine:

#     MATERIAL_PROPERTIES = {
#         0: {"name": "OXIDE_DIELECTRIC",   "yield": 0.65, "height_nm": 0.0},
#         1: {"name": "SILICON_SUBSTRATE", "yield": 1.00, "height_nm": 0.0},
#         2: {"name": "POLY_SILICON",       "yield": 1.25, "height_nm": 45.0},
#         3: {"name": "METAL_INTERCONNECT", "yield": 1.85, "height_nm": 55.0},
#         4: {"name": "CONTACT_VIA",        "yield": 2.10, "height_nm": 70.0}
#     }

#     @staticmethod
#     def calculate_kanaya_okayama_psf_px(beam_energy_keV: float, base_sigma_nm: float, pixel_size_nm: float) -> float:
#         r_ko_um = 0.0276 * 28.085 * (beam_energy_keV ** 1.67) / (2.33 * (14.0 ** 0.899))
#         r_ko_nm = r_ko_um * 1000.0
#         eff_sigma_nm = base_sigma_nm + 0.015 * r_ko_nm
#         return float(eff_sigma_nm / pixel_size_nm)

#     @staticmethod
#     def synthesize_palasantzas_ler_field(shape: Tuple[int, int], sigma_px: float, xi_px: float, hurst: float, rng: np.random.Generator) -> np.ndarray:
#         if sigma_px <= 0.0:
#             return np.zeros(shape, dtype=np.float32)

#         h, w = shape
#         fy = np.fft.fftfreq(h)[:, None]
#         fx = np.fft.fftfreq(w)[None, :]
#         f_sq = fx**2 + fy**2

#         psd = (2.0 * (sigma_px ** 2) * (xi_px**2)) / ((1.0 + (2.0 * np.pi * np.sqrt(f_sq) * xi_px) ** 2) ** (hurst + 1.0))
#         psd[0, 0] = 0.0

#         random_phase = np.exp(1j * rng.uniform(0, 2.0 * np.pi, (h, w)))
#         spectrum = np.sqrt(np.maximum(psd, 0.0)) * random_phase

#         field = np.fft.ifft2(spectrum).real.astype(np.float32)
#         std_val = np.std(field)
#         if std_val > 1e-6:
#             field *= (sigma_px / std_val)
#         return field

#     @classmethod
#     def apply_edge_specific_ler(
#         cls, 
#         canvas: np.ndarray, 
#         height_map: np.ndarray, 
#         proc: ProcessVariationParams, 
#         pixel_size_nm: float, 
#         rng: np.random.Generator
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         if proc.ler_sigma_nm <= 0.0:
#             return canvas, height_map

#         sigma_px = proc.ler_sigma_nm / pixel_size_nm
#         xi_px = proc.ler_correlation_length_nm / pixel_size_nm

#         gx = cv2.Sobel(canvas.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(canvas.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
#         edge_mag = np.sqrt(gx**2 + gy**2)
#         edge_mask = (edge_mag > 15.0).astype(np.float32)

#         ler_field_x = cls.synthesize_palasantzas_ler_field(canvas.shape, sigma_px, xi_px, proc.ler_hurst, rng)
#         ler_field_y = cls.synthesize_palasantzas_ler_field(canvas.shape, sigma_px, xi_px, proc.ler_hurst, rng)

#         h, w = canvas.shape
#         grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

#         map_x = (grid_x + ler_field_x * edge_mask).astype(np.float32)
#         map_y = (grid_y + ler_field_y * edge_mask).astype(np.float32)

#         warped_canvas = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         warped_height = cv2.remap(height_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

#         return warped_canvas, warped_height

#     @classmethod
#     def apply_seiler_topographic_yield(cls, height_map_nm: np.ndarray, mat_map: np.ndarray, alpha: float = 0.38) -> np.ndarray:
#         dz_dx = cv2.Sobel(height_map_nm, cv2.CV_64F, 1, 0, ksize=3) / 8.0
#         dz_dy = cv2.Sobel(height_map_nm, cv2.CV_64F, 0, 1, ksize=3) / 8.0
#         sec_theta = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)

#         base_yield = np.ones_like(height_map_nm, dtype=np.float64)
#         for mat_id, prop in cls.MATERIAL_PROPERTIES.items():
#             base_yield[mat_map == mat_id] = prop["yield"]

#         se_yield = base_yield * (1.0 + alpha * (sec_theta - 1.0))
#         return se_yield.astype(np.float32)

#     @staticmethod
#     def apply_anisotropic_blur(img_float: np.ndarray, sigma_x_px: float, sigma_y_px: float, angle_deg: float) -> np.ndarray:
#         kx_size = max(3, int(6 * sigma_x_px) | 1)
#         ky_size = max(3, int(6 * sigma_y_px) | 1)
#         ksize = max(kx_size, ky_size)

#         kx = cv2.getGaussianKernel(ksize, sigma_x_px)
#         ky = cv2.getGaussianKernel(ksize, sigma_y_px)
#         kernel = ky @ kx.T

#         if abs(angle_deg) > 1e-3:
#             M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle_deg, 1.0)
#             kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
#             kernel /= np.maximum(np.sum(kernel), 1e-6)

#         return cv2.filter2D(img_float, -1, kernel)

#     @classmethod
#     def process_sem_response(
#         cls, 
#         clean_raster: np.ndarray, 
#         height_map: np.ndarray, 
#         mat_map: np.ndarray, 
#         phys: PhysicsParams, 
#         proc: ProcessVariationParams, 
#         pixel_size_nm: float, 
#         rng: np.random.Generator, 
#         is_search: bool = False
#     ) -> np.ndarray:
#         ler_raster, height_ler = cls.apply_edge_specific_ler(clean_raster, height_map, proc, pixel_size_nm, rng)

#         se_yield = cls.apply_seiler_topographic_yield(height_ler, mat_map, alpha=phys.se_alpha)
#         img_float = (ler_raster.astype(np.float32) / 255.0) * se_yield

#         if is_search and (phys.charging_strength > 0 or phys.vignetting_strength > 0):
#             h, w = img_float.shape
#             gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
#             r_sq = ((gx - w / 2.0) ** 2 + (gy - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)
#             vignette = 1.0 - phys.vignetting_strength * r_sq
            
#             charge_pool = 0.0
#             if phys.charging_strength > 0:
#                 cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
#                 charge_pool = phys.charging_strength * np.exp(-((gx - cx)**2 + (gy - cy)**2) / (2.0 * (0.35 * w)**2))
            
#             img_float = np.clip(img_float * vignette + charge_pool, 0.0, 10.0)

#         base_sigma_nm = phys.psf_sigma_srch_nm if is_search else phys.psf_sigma_ref_nm
#         eff_sigma_x_px = cls.calculate_kanaya_okayama_psf_px(phys.beam_energy_keV, base_sigma_nm, pixel_size_nm)
#         eff_sigma_y_px = phys.psf_sigma_y_nm / pixel_size_nm
#         blurred = cls.apply_anisotropic_blur(img_float, eff_sigma_x_px, eff_sigma_y_px, phys.astigmatism_angle_deg)

#         dose = phys.dwell_time_srch if is_search else phys.dwell_time_ref
#         electron_counts = np.maximum(blurred, 1e-6) * dose
#         noisy_shot = rng.poisson(electron_counts).astype(np.float32) / dose
        
#         speckle = rng.normal(1.0, phys.speckle_noise_std, size=img_float.shape).astype(np.float32)
#         readout = rng.normal(0, phys.readout_noise_std, size=img_float.shape).astype(np.float32)
#         combined = np.clip(noisy_shot * speckle + readout, 0.0, 10.0)

#         if phys.salt_pepper_prob > 0:
#             sp = rng.random(combined.shape)
#             combined[sp < (phys.salt_pepper_prob / 2.0)] = 0.0
#             combined[sp > (1.0 - phys.salt_pepper_prob / 2.0)] = 2.0

#         norm_signal = combined / np.maximum(np.percentile(combined, 99.5), 1e-5)
#         gamma_corr = np.power(np.clip(norm_signal, 0, 1), phys.gamma_exponent)
#         saturated = phys.detector_gain * (gamma_corr / (1.0 + gamma_corr / max(0.1, phys.detector_sat_threshold)))

#         fpn = rng.normal(1.0, phys.fpn_strength, img_float.shape).astype(np.float32)
#         output = saturated * fpn * 255.0 + phys.baseline_black_level

#         return np.clip(output, 0, 255).astype(np.uint8)


# # ==============================================================================
# # CANONICAL LAYOUT ENGINE
# # ==============================================================================

# class LayoutEngine:

#     @staticmethod
#     def get_benchmark_ambiguity_preset(difficulty: str) -> BenchmarkAmbiguityParams:
#         d = difficulty.lower()
#         if d == "easy":
#             return BenchmarkAmbiguityParams("easy", 0.10, 1, 0.70, 0, 0.0, (1.0, 1.0), (1.0, 1.0), 0.0)
#         elif d == "medium":
#             return BenchmarkAmbiguityParams("medium", 0.40, 2, 0.88, 4, 1.5, (0.98, 1.02), (0.98, 1.02), 0.0)
#         elif d == "hard":
#             return BenchmarkAmbiguityParams("hard", 0.75, 5, 0.96, 15, 3.0, (0.96, 1.04), (0.96, 1.04), 0.10)
#         elif d == "extreme_plus":
#             return BenchmarkAmbiguityParams("extreme_plus", 0.99, 12, 0.998, 40, 7.5, (0.92, 1.08), (0.92, 1.08), 0.30)
#         else: # extreme
#             return BenchmarkAmbiguityParams("extreme", 0.95, 10, 0.995, 30, 5.0, (0.94, 1.06), (0.94, 1.06), 0.20)

#     @classmethod
#     def generate_random_spec(cls, arch_choice: str, difficulty: str, rng: np.random.Generator) -> Tuple[LayoutSpec, ProcessVariationParams, PhysicsParams, BenchmarkAmbiguityParams]:
#         amb = cls.get_benchmark_ambiguity_preset(difficulty)
#         is_hard_or_extreme = difficulty.lower() in ["hard", "extreme", "extreme_plus"]

#         spec = LayoutSpec(
#             architecture=arch_choice,
#             pitch_x_nm=rng.integers(180, 260),
#             pitch_y_nm=rng.integers(160, 240),
#             line_w_x_nm=rng.integers(28, 48),
#             line_w_y_nm=rng.integers(30, 50),
#             feature_size_nm=rng.integers(20, 36),
#             base_gray=25,
#             metal_gray=175,
#             contact_gray=220,
#             macro_width_nm=rng.integers(200, 300),
#             dram_stagger_mode=str(rng.choice(['STAGGER_50', 'HEX'])) if is_hard_or_extreme else 'STAGGER_50',
#             dram_wave_amp_nm=rng.integers(4, 16) if is_hard_or_extreme else 0,
#             dram_wave_freq1=rng.uniform(0.8, 1.8),
#             dram_wave_freq2=rng.uniform(0.3, 0.9),
#             dram_wave_phase=rng.uniform(0.0, 2.0 * np.pi),
#             dram_pad_angle=rng.uniform(-30.0, 30.0) if is_hard_or_extreme else 0.0,
#             finfet_cluster_size=rng.integers(2, 4)
#         )

#         proc = ProcessVariationParams(
#             ler_sigma_nm=rng.uniform(0.5, 1.2) if difficulty == "easy" else rng.uniform(1.2, 3.0),
#             ler_correlation_length_nm=rng.uniform(12.0, 25.0),
#             ler_hurst=rng.uniform(0.65, 0.85),
#             cd_taper_pct=rng.uniform(0.0, 0.02) if difficulty == "easy" else rng.uniform(0.02, 0.08),
#             cmp_dishing_strength=rng.uniform(0.0, 0.05) if difficulty == "easy" else rng.uniform(0.05, 0.22),
#             opc_corner_rounding_radius=2 if difficulty == "easy" else rng.integers(2, 6),
#             etch_bias_nm=rng.uniform(-1.0, 1.0) if difficulty == "easy" else rng.uniform(-3.5, 3.5),
#             enable_pattern_collapse=is_hard_or_extreme
#         )

#         dwell_srch = rng.uniform(120.0, 200.0) if difficulty == "easy" else (
#             rng.uniform(50.0, 100.0) if difficulty == "medium" else (
#                 rng.uniform(15.0, 50.0) if difficulty == "hard" else rng.uniform(8.0, 25.0)
#             )
#         )

#         phys = PhysicsParams(
#             beam_energy_keV=rng.uniform(1.0, 2.0),
#             se_alpha=rng.uniform(0.32, 0.42),
#             psf_sigma_ref_nm=rng.uniform(0.4, 0.6),
#             psf_sigma_srch_nm=rng.uniform(1.0, 1.6) if difficulty == "easy" else rng.uniform(1.4, 2.5),
#             psf_sigma_y_nm=rng.uniform(0.5, 0.7) if difficulty == "easy" else rng.uniform(1.0, 2.8),
#             astigmatism_angle_deg=0.0 if difficulty == "easy" else rng.uniform(0.0, 360.0),
#             dwell_time_ref=rng.uniform(220.0, 320.0),
#             dwell_time_srch=dwell_srch,
#             readout_noise_std=0.02 if difficulty == "easy" else rng.uniform(0.03, 0.08),
#             speckle_noise_std=0.01 if difficulty == "easy" else rng.uniform(0.02, 0.05),
#             salt_pepper_prob=0.0 if not is_hard_or_extreme else rng.uniform(0.0005, 0.004),
#             detector_gain=1.0 if difficulty == "easy" else rng.uniform(0.8, 1.5),
#             detector_sat_threshold=1.0,
#             gamma_exponent=1.0 if difficulty == "easy" else rng.uniform(0.7, 1.4),
#             baseline_black_level=30.0 if difficulty == "easy" else rng.uniform(20.0, 60.0),
#             fpn_strength=0.005 if difficulty == "easy" else rng.uniform(0.008, 0.025),
#             charging_strength=0.0 if difficulty == "easy" else rng.uniform(0.05, 0.35),
#             vignetting_strength=0.0 if difficulty == "easy" else rng.uniform(0.05, 0.22),
#             raster_drift_velocity_px=0.0 if difficulty == "easy" else rng.uniform(0.01, 0.06),
#             raster_vibration_amp_px=0.0 if difficulty == "easy" else rng.uniform(0.2, 1.2),
#             scanline_jitter_prob=0.0 if difficulty == "easy" else rng.uniform(0.005, 0.03),
#             scan_direction=int(rng.choice([0, 1])),
#             enable_elastic_warp=is_hard_or_extreme,
#             elastic_alpha_px=rng.uniform(2.0, 4.5),
#             elastic_sigma_px=rng.uniform(15.0, 25.0),
#             enable_barrel_distortion=is_hard_or_extreme,
#             barrel_k1=rng.uniform(1e-6, 3.0e-6)
#         )

#         return spec, proc, phys, amb

#     @classmethod
#     def render_dram_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec) -> List[Tuple[int, int]]:
#         h, w = sub.shape
#         contact_coords = []

#         for y in range(0, h, spec.pitch_y_nm):
#             cv2.rectangle(sub, (0, y), (w, y + spec.line_w_y_nm), spec.metal_gray, -1)
#             cv2.rectangle(sub_h, (0, y), (w, y + spec.line_w_y_nm), 55.0, -1)
#             cv2.rectangle(sub_m, (0, y), (w, y + spec.line_w_y_nm), 3, -1)

#         if spec.dram_wave_amp_nm > 0:
#             amp = spec.dram_wave_amp_nm
#             f1, ph = spec.dram_wave_freq1, spec.dram_wave_phase
#             for x in range(0, w, spec.pitch_x_nm):
#                 pts = [(int(x + amp * np.sin(2 * np.pi * f1 * y / spec.pitch_y_nm + ph)), y) for y in range(0, h, 20)]
#                 pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(sub, [pts_arr], False, spec.metal_gray + 20, spec.line_w_x_nm)
#                 cv2.polylines(sub_h, [pts_arr], False, 45.0, spec.line_w_x_nm)
#                 cv2.polylines(sub_m, [pts_arr], False, 2, spec.line_w_x_nm)
#         else:
#             for x in range(0, w, spec.pitch_x_nm):
#                 cv2.rectangle(sub, (x, 0), (x + spec.line_w_x_nm, h), spec.metal_gray + 20, -1)
#                 cv2.rectangle(sub_h, (x, 0), (x + spec.line_w_x_nm, h), 45.0, -1)
#                 cv2.rectangle(sub_m, (x, 0), (x + spec.line_w_x_nm, h), 2, -1)

#         rx = spec.feature_size_nm
#         ry = max(6, int(spec.feature_size_nm * 0.6))
#         for y in range(spec.pitch_y_nm // 2, h, spec.pitch_y_nm):
#             for x in range(spec.pitch_x_nm // 2, w, spec.pitch_x_nm):
#                 if abs(spec.dram_pad_angle) > 1.0:
#                     cv2.ellipse(sub, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, spec.contact_gray, -1)
#                     cv2.ellipse(sub_h, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, 70.0, -1)
#                     cv2.ellipse(sub_m, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, 4, -1)
#                 else:
#                     cv2.circle(sub, (x, y), rx, spec.contact_gray, -1)
#                     cv2.circle(sub_h, (x, y), rx, 70.0, -1)
#                     cv2.circle(sub_m, (x, y), rx, 4, -1)
#                 contact_coords.append((x, y))

#         return contact_coords

#     @classmethod
#     def render_finfet_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec) -> List[Tuple[int, int]]:
#         h, w = sub.shape
#         contact_coords = []

#         fin_x_positions = []
#         x = 0
#         while x < w:
#             for c in range(spec.finfet_cluster_size):
#                 fx = x + c * spec.pitch_x_nm
#                 if fx < w:
#                     cv2.rectangle(sub, (fx, 0), (fx + spec.line_w_x_nm, h), spec.metal_gray - 20, -1)
#                     cv2.rectangle(sub_h, (fx, 0), (fx + spec.line_w_x_nm, h), 50.0, -1)
#                     cv2.rectangle(sub_m, (fx, 0), (fx + spec.line_w_x_nm, h), 2, -1)
#                     fin_x_positions.append(fx + spec.line_w_x_nm // 2)
#             x += spec.finfet_cluster_size * spec.pitch_x_nm + 120

#         for y in range(0, h, spec.pitch_y_nm):
#             gate_w = spec.line_w_y_nm
#             cv2.rectangle(sub, (0, y), (w, y + gate_w), spec.metal_gray + 35, -1)
#             cv2.rectangle(sub_h, (0, y), (w, y + gate_w), 60.0, -1)
#             cv2.rectangle(sub_m, (0, y), (w, y + gate_w), 3, -1)

#             for fx in fin_x_positions:
#                 cy = y + gate_w + spec.pitch_y_nm // 4
#                 if cy < h:
#                     cv2.rectangle(sub, (fx - 10, cy - 8), (fx + 10, cy + 8), spec.contact_gray, -1)
#                     cv2.rectangle(sub_h, (fx - 10, cy - 8), (fx + 10, cy + 8), 70.0, -1)
#                     cv2.rectangle(sub_m, (fx - 10, cy - 8), (fx + 10, cy + 8), 4, -1)
#                     contact_coords.append((fx, cy))

#         return contact_coords

#     @classmethod
#     def apply_process_variations(
#         cls, 
#         canvas: np.ndarray, 
#         height_map: np.ndarray, 
#         mat_map: np.ndarray, 
#         proc: ProcessVariationParams
#     ):
#         h, w = canvas.shape

#         if proc.cd_taper_pct > 0:
#             gy, gx = np.ogrid[:h, :w]
#             r_norm = np.sqrt((gx - w / 2.0)**2 + (gy - h / 2.0)**2) / (w / 2.0)
#             taper_factor = 1.0 + proc.cd_taper_pct * (r_norm - 0.5)
#             canvas[:] = np.clip(canvas.astype(np.float32) * taper_factor, 0, 255).astype(np.uint8)

#         if proc.cmp_dishing_strength > 0:
#             gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
#             dishing_mask = 1.0 - proc.cmp_dishing_strength * gy
#             height_map *= dishing_mask

#         if abs(proc.etch_bias_nm) >= 1.0:
#             ksize = max(3, int(abs(proc.etch_bias_nm)) | 1)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#             if proc.etch_bias_nm > 0:
#                 canvas[:] = cv2.dilate(canvas, kernel)
#             else:
#                 canvas[:] = cv2.erode(canvas, kernel)

#         if proc.opc_corner_rounding_radius > 1:
#             r = proc.opc_corner_rounding_radius
#             kernel_opc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
#             canvas[:] = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel_opc)

#     @classmethod
#     def apply_physical_pattern_collapse(cls, canvas: np.ndarray, height_map: np.ndarray, rng: np.random.Generator):
#         h, w = canvas.shape
#         num = rng.integers(2, 5)
#         for _ in range(num):
#             cx, cy = rng.integers(1000, w - 1000), rng.integers(1000, h - 1000)
#             rad = rng.integers(100, 250)

#             gy, gx = np.ogrid[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]
#             dist_sq = (gx - cx)**2 + (gy - cy)**2
#             mask = np.exp(-dist_sq / (2.0 * (rad / 2.0)**2)).astype(np.float32)

#             shift_x = mask * 12.0
#             grid_y, grid_x = np.mgrid[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)].astype(np.float32)

#             sub_c = canvas[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]
#             sub_h = height_map[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]

#             canvas[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)] = cv2.remap(
#                 sub_c, grid_x - shift_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
#             )
#             height_map[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)] = cv2.remap(
#                 sub_h, grid_x - shift_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
#             )

#     @classmethod
#     def inject_geometry_aware_defects(
#         cls, 
#         canvas: np.ndarray, 
#         height_map: np.ndarray, 
#         mat_map: np.ndarray, 
#         contact_coords: List[Tuple[int, int]], 
#         count: int, 
#         rng: np.random.Generator
#     ):
#         if count <= 0 or not contact_coords:
#             return

#         defects = ['MISSING_VIA', 'LINE_BRIDGING', 'LINE_CUT']
#         indices = rng.choice(len(contact_coords), size=min(count, len(contact_coords)), replace=False)

#         for idx in indices:
#             px, py = contact_coords[idx]
#             dtype = rng.choice(defects)

#             if dtype == 'MISSING_VIA':
#                 cv2.circle(canvas, (px, py), 14, 25, -1)
#                 cv2.circle(height_map, (px, py), 14, 0.0, -1)
#                 cv2.circle(mat_map, (px, py), 14, 0, -1)
#             elif dtype == 'LINE_BRIDGING':
#                 cv2.line(canvas, (px, py - 20), (px, py + 20), 195, 8)
#                 cv2.line(height_map, (px, py - 20), (px, py + 20), 55.0, 8)
#                 cv2.line(mat_map, (px, py - 20), (px, py + 20), 3, 8)
#             else:
#                 cv2.line(canvas, (px - 20, py), (px + 20, py), 25, 10)
#                 cv2.line(height_map, (px - 20, py), (px + 20, py), 0.0, 10)
#                 cv2.line(mat_map, (px - 20, py), (px + 20, py), 0, 10)

#     @classmethod
#     def render_canvas(
#         cls, 
#         width: int, 
#         height: int, 
#         spec: LayoutSpec, 
#         amb: BenchmarkAmbiguityParams, 
#         proc: ProcessVariationParams, 
#         rng: np.random.Generator
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
#         canvas = np.full((height, width), spec.base_gray, dtype=np.uint8)
#         height_map = np.zeros((height, width), dtype=np.float32)
#         mat_map = np.zeros((height, width), dtype=np.uint8)

#         if spec.architecture == "FinFET":
#             contacts = cls.render_finfet_canonical(canvas, height_map, mat_map, spec)
#         else:
#             contacts = cls.render_dram_canonical(canvas, height_map, mat_map, spec)

#         cls.apply_process_variations(canvas, height_map, mat_map, proc)

#         if proc.enable_pattern_collapse:
#             cls.apply_physical_pattern_collapse(canvas, height_map, rng)

#         cls.inject_geometry_aware_defects(canvas, height_map, mat_map, contacts, amb.repeated_defect_count, rng)

#         return canvas, height_map, mat_map, contacts

#     @classmethod
#     def render_deceptive_candidates_in_search_fov(
#         cls, 
#         master_canvas: np.ndarray, 
#         master_height: np.ndarray, 
#         master_mat: np.ndarray, 
#         spec: LayoutSpec, 
#         amb: BenchmarkAmbiguityParams, 
#         search_start_x: int, 
#         search_start_y: int, 
#         ref_start_x: int, 
#         ref_start_y: int, 
#         gt_transformed_center_search_px: Tuple[float, float],
#         rot_matrix: np.ndarray,
#         disp_fwd_x: np.ndarray,
#         disp_fwd_y: np.ndarray,
#         rng: np.random.Generator
#     ) -> Tuple[int, List[CandidateMetadata]]:
#         if amb.deceptive_candidate_count <= 0:
#             return 0, []

#         block_w = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)
#         block_h = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)

#         target_crop = master_canvas[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()
#         target_h = master_height[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()
#         target_m = master_mat[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()

#         placed_centers = []
#         candidate_metas: List[CandidateMetadata] = []

#         search_fov_px = CoordinateTransformer.master_nm_to_px(SEARCH_FOV_IN_MASTER_NM)
#         min_x = search_start_x + 200
#         max_x = search_start_x + search_fov_px - block_w - 200
#         min_y = search_start_y + 200
#         max_y = search_start_y + search_fov_px - block_h - 200

#         attempts = 0
#         max_attempts = amb.deceptive_candidate_count * 10

#         while len(placed_centers) < amb.deceptive_candidate_count and attempts < max_attempts:
#             attempts += 1
#             cx = int(rng.integers(min_x, max_x))
#             cy = int(rng.integers(min_y, max_y))

#             if abs(cx - ref_start_x) < block_w and abs(cy - ref_start_y) < block_h:
#                 continue

#             if any(abs(cx - px) < block_w and abs(cy - py) < block_h for px, py in placed_centers):
#                 continue

#             cand_crop = target_crop.copy()
#             cand_height = target_h.copy()
#             cand_mat = target_m.copy()

#             num_mods = max(1, int((1.0 - amb.cell_similarity_pct) * 100))
#             for _ in range(num_mods):
#                 mx = rng.integers(50, block_w - 50)
#                 my = rng.integers(50, block_h - 50)
#                 val = spec.base_gray if rng.random() > 0.5 else spec.contact_gray
                
#                 cv2.rectangle(cand_crop, (mx - 8, my - 8), (mx + 8, my + 8), val, -1)
#                 cv2.rectangle(cand_height, (mx - 8, my - 8), (mx + 8, my + 8), 70.0 if val > 100 else 0.0, -1)
#                 cv2.rectangle(cand_mat, (mx - 8, my - 8), (mx + 8, my + 8), 4 if val > 100 else 0, -1)

#             # Metrics calculated on clean layout patches before non-linear SEM response
#             ssim_score = compute_ssim(target_crop, cand_crop)
#             ncc_score = compute_ncc(target_crop, cand_crop)

#             master_canvas[cy:cy + block_h, cx:cx + block_w] = cand_crop
#             master_height[cy:cy + block_h, cx:cx + block_w] = cand_height
#             master_mat[cy:cy + block_h, cx:cx + block_w] = cand_mat

#             placed_centers.append((cx, cy))

#             unwarped_px_x, unwarped_px_y = CoordinateTransformer.master_px_to_search_px(
#                 cx + block_w / 2.0, cy + block_h / 2.0, search_start_x, search_start_y
#             )

#             trans_px_x, trans_px_y = CompoundTransformEngine.forward_point(
#                 (unwarped_px_x, unwarped_px_y), rot_matrix, disp_fwd_x, disp_fwd_y
#             )

#             dist_gt = math.sqrt((trans_px_x - gt_transformed_center_search_px[0])**2 + (trans_px_y - gt_transformed_center_search_px[1])**2)

#             cand_meta = CandidateMetadata(
#                 candidate_id=f"cand_{len(placed_centers):02d}",
#                 candidate_type="ADVERSARIAL_TRAP",
#                 unwarped_center_x_search_px=round(unwarped_px_x, 4),
#                 unwarped_center_y_search_px=round(unwarped_px_y, 4),
#                 transformed_center_x_search_px=round(trans_px_x, 4),
#                 transformed_center_y_search_px=round(trans_px_y, 4),
#                 distance_from_gt_transformed_px=round(dist_gt, 4),
#                 ssim_to_target_clean=round(ssim_score, 4),
#                 ncc_to_target_clean=round(ncc_score, 4)
#             )
#             candidate_metas.append(cand_meta)

#         return len(placed_centers), candidate_metas


# # ==============================================================================
# # QA & DATA INTEGRITY PIPELINE
# # ==============================================================================

# class DatasetQAPipeline:

#     @staticmethod
#     def compute_sha256(filepath: str) -> str:
#         hasher = hashlib.sha256()
#         with open(filepath, 'rb') as f:
#             while chunk := f.read(65536):
#                 hasher.update(chunk)
#         return hasher.hexdigest()

#     @classmethod
#     def validate_sample(cls, sample_dir: str, meta: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
#         errors = []

#         ref_full_path = os.path.join(sample_dir, meta["reference_path"])
#         search_full_path = os.path.join(sample_dir, meta["search_path"])

#         if not os.path.exists(ref_full_path) or not os.path.exists(search_full_path):
#             errors.append("Reference or Search image file missing.")
#             return False, errors

#         ref_img = cv2.imread(ref_full_path, cv2.IMREAD_GRAYSCALE)
#         search_img = cv2.imread(search_full_path, cv2.IMREAD_GRAYSCALE)

#         if ref_img is None or search_img is None:
#             errors.append("Failed to load Reference or Search PNG.")
#             return False, errors

#         if ref_img.shape != (REFERENCE_SIZE_PX, REFERENCE_SIZE_PX):
#             errors.append(f"Invalid Reference size: {ref_img.shape}")

#         if search_img.shape != (SEARCH_SIZE_PX, SEARCH_SIZE_PX):
#             errors.append(f"Invalid Search size: {search_img.shape}")

#         gt_x, gt_y = meta["gt_center_x"], meta["gt_center_y"]
#         if not (0 <= gt_x <= SEARCH_SIZE_PX and 0 <= gt_y <= SEARCH_SIZE_PX):
#             errors.append(f"GT Center ({gt_x}, {gt_y}) outside Search bounds.")

#         polygon = meta.get("transformed_polygon", [])
#         if len(polygon) < 4:
#             errors.append("Transformed GT polygon missing or malformed.")
#         else:
#             poly_np = np.array(polygon, dtype=np.float32)
#             area = cv2.contourArea(poly_np)
#             if area < 100.0:
#                 errors.append(f"Transformed polygon area too small: {area}")

#         # Relaxed QA thresholds specifically calibrated for extreme non-linear warp fields
#         if meta.get("max_transform_residual_px", 0.0) > 0.10:
#             errors.append(f"Transform residual exceeded threshold: {meta['max_transform_residual_px']} px")

#         if meta.get("direct_marker_verification_err_px", 0.0) > 0.08:
#             errors.append(f"Direct Marker Verification Error exceeded threshold: {meta['direct_marker_verification_err_px']} px")

#         for cand in meta.get("candidates", []):
#             if cand["distance_from_gt_transformed_px"] < 50.0:
#                 errors.append(f"Candidate {cand['candidate_id']} too close to Ground Truth target.")

#         is_valid = (len(errors) == 0)
#         return is_valid, errors


# # ==============================================================================
# # EVALUATION & BENCHMARKING ENGINE
# # ==============================================================================

# class EvaluationEngine:

#     @staticmethod
#     def evaluate_predictions(gt_metadata_list: List[Dict[str, Any]], predictions_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#         pred_map = {p["sample_id"]: p for p in predictions_list}
#         errors = []
#         confidences = []

#         for gt in gt_metadata_list:
#             sid = gt["sample_id"]
#             if sid not in pred_map:
#                 continue

#             pred = pred_map[sid]
#             px, py = pred["predicted_x"], pred["predicted_y"]
#             conf = pred.get("confidence", 1.0)
#             tx, ty = gt["gt_center_x"], gt["gt_center_y"]

#             err = math.sqrt((px - tx)**2 + (py - ty)**2)
#             errors.append(err)
#             confidences.append(conf)

#         n = max(1, len(errors))
#         err_arr = np.array(errors)
#         conf_arr = np.array(confidences)

#         ap_metrics = {}
#         for tol in [1.0, 3.0, 5.0, 10.0]:
#             tp = (err_arr <= tol).astype(np.float32)
#             sort_idx = np.argsort(-conf_arr)
#             tp_sorted = tp[sort_idx]
            
#             acc_tp = np.cumsum(tp_sorted)
#             recalls = acc_tp / max(1.0, np.sum(tp))
#             precisions = acc_tp / (np.arange(len(tp_sorted)) + 1)
            
#             ap = 0.0
#             for t in np.arange(0.0, 1.1, 0.1):
#                 p_mask = recalls >= t
#                 p_max = np.max(precisions[p_mask]) if np.any(p_mask) else 0.0
#                 ap += p_max / 11.0
            
#             ap_metrics[f"AP@{int(tol)}px"] = round(float(ap), 4)

#         return {
#             "num_evaluated": len(errors),
#             "mae_px": float(np.mean(err_arr)),
#             "median_err_px": float(np.median(err_arr)),
#             "rmse_px": float(np.sqrt(np.mean(err_arr**2))),
#             "p95_err_px": float(np.percentile(err_arr, 95)),
#             "ap_scores": ap_metrics,
#             "accuracy_pct_at_1px": round(float(np.mean(err_arr <= 1.0) * 100.0), 2),
#             "accuracy_pct_at_3px": round(float(np.mean(err_arr <= 3.0) * 100.0), 2),
#             "accuracy_pct_at_5px": round(float(np.mean(err_arr <= 5.0) * 100.0), 2),
#             "accuracy_pct_at_10px": round(float(np.mean(err_arr <= 10.0) * 100.0), 2)
#         }


# # ==============================================================================
# # MAIN DATASET GENERATOR CLASS
# # ==============================================================================

# class SEMDatasetGenerator:

#     def __init__(self, output_dir: str = "./synthetic_sem_dataset", visualize: bool = False, difficulty: str = "medium", seed: int = 42, strict: bool = False):
#         self.output_dir = output_dir
#         self.visualize = visualize
#         self.difficulty = difficulty
#         self.global_seed = seed
#         self.strict = strict

#         self.ref_dir = os.path.join(output_dir, "reference")
#         self.search_dir = os.path.join(output_dir, "search")
#         os.makedirs(self.ref_dir, exist_ok=True)
#         os.makedirs(self.search_dir, exist_ok=True)

#         if self.visualize:
#             self.preview_dir = os.path.join(output_dir, "previews")
#             os.makedirs(self.preview_dir, exist_ok=True)

#     def generate_single_sample(self, sample_id: str, arch_choice: str, difficulty: str, sample_seed: int) -> Dict[str, Any]:
#         rng = np.random.default_rng(sample_seed)
#         master_w = CoordinateTransformer.master_nm_to_px(12000.0)
#         master_h = CoordinateTransformer.master_nm_to_px(12000.0)

#         spec, proc, phys, amb = LayoutEngine.generate_random_spec(arch_choice, difficulty, rng)
#         master_canvas, master_height, master_mat, contacts = LayoutEngine.render_canvas(master_w, master_h, spec, amb, proc, rng)

#         search_fov_px = CoordinateTransformer.master_nm_to_px(SEARCH_FOV_IN_MASTER_NM)
#         ref_fov_px = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)

#         search_start_x = int(rng.integers(500, master_w - search_fov_px - 500))
#         search_start_y = int(rng.integers(500, master_h - search_fov_px - 500))

#         unwarped_gt_x = float(rng.uniform(200.0, 800.0))
#         unwarped_gt_y = float(rng.uniform(200.0, 800.0))

#         ref_start_x = int(search_start_x + unwarped_gt_x * 10.0 - ref_fov_px / 2.0)
#         ref_start_y = int(search_start_y + unwarped_gt_y * 10.0 - ref_fov_px / 2.0)

#         # Build Continuous Forward Geometric Transform Engine
#         angle_deg = float(rng.uniform(-amb.max_rotation_deg, amb.max_rotation_deg))
#         scale_x = float(rng.uniform(*amb.scale_range_x))
#         scale_y = float(rng.uniform(*amb.scale_range_y))

#         rot_matrix = CompoundTransformEngine.build_anisotropic_affine_matrix((500.0, 500.0), angle_deg, scale_x, scale_y)
#         disp_fwd_x, disp_fwd_y, map_x, map_y = CompoundTransformEngine.create_compound_warp_field(SEARCH_SIZE_PX, SEARCH_SIZE_PX, phys, rng)

#         # Transformed Target GT Center
#         final_gt_x, final_gt_y = CompoundTransformEngine.forward_point((unwarped_gt_x, unwarped_gt_y), rot_matrix, disp_fwd_x, disp_fwd_y)

#         # DIRECT CONTINUOUS SUB-PIXEL GAUSSIAN PULSE VERIFICATION PASS
#         gy_m, gx_m = np.mgrid[0:SEARCH_SIZE_PX, 0:SEARCH_SIZE_PX].astype(np.float32)
#         dist_sq_m = (gx_m - unwarped_gt_x)**2 + (gy_m - unwarped_gt_y)**2
#         marker_canvas = np.exp(-dist_sq_m / (2.0 * (1.0**2))).astype(np.float32)

#         marker_aff = cv2.warpAffine(marker_canvas, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR)
#         marker_warped = cv2.remap(marker_aff, map_x, map_y, interpolation=cv2.INTER_LINEAR)

#         min_v, max_v, min_l, max_l = cv2.minMaxLoc(marker_warped)
#         px_m, py_m = max_l
#         if 1 <= px_m < SEARCH_SIZE_PX - 1 and 1 <= py_m < SEARCH_SIZE_PX - 1:
#             dx_m = (marker_warped[py_m, px_m + 1] - marker_warped[py_m, px_m - 1]) / (2.0 * (2.0 * marker_warped[py_m, px_m] - marker_warped[py_m, px_m + 1] - marker_warped[py_m, px_m - 1] + 1e-6))
#             dy_m = (marker_warped[py_m + 1, px_m] - marker_warped[py_m - 1, px_m]) / (2.0 * (2.0 * marker_warped[py_m, px_m] - marker_warped[py_m + 1, px_m] - marker_warped[py_m - 1, px_m] + 1e-6))
#             meas_gt_x = px_m + np.clip(dx_m, -0.5, 0.5)
#             meas_gt_y = py_m + np.clip(dy_m, -0.5, 0.5)
#         else:
#             meas_gt_x, meas_gt_y = float(px_m), float(py_m)

#         direct_marker_verification_err_px = float(math.sqrt((meas_gt_x - final_gt_x)**2 + (meas_gt_y - final_gt_y)**2))

#         # Render Deceptive Candidates and Forward-Transform Coordinates
#         actual_cand_count, cand_metas = LayoutEngine.render_deceptive_candidates_in_search_fov(
#             master_canvas, master_height, master_mat, spec, amb, 
#             search_start_x, search_start_y, ref_start_x, ref_start_y,
#             (final_gt_x, final_gt_y), rot_matrix, disp_fwd_x, disp_fwd_y, rng
#         )

#         # Crop Search region
#         raw_search = master_canvas[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]
#         raw_search_h = master_height[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]
#         raw_search_m = master_mat[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]

#         search_ds = cv2.resize(cv2.GaussianBlur(raw_search, (7, 7), 1.5), (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)
#         search_h_ds = cv2.resize(raw_search_h, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)
#         search_m_ds = cv2.resize(raw_search_m, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_NEAREST)

#         # Crop Reference region
#         raw_ref_crop = master_canvas[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()
#         raw_ref_h = master_height[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()
#         raw_ref_m = master_mat[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()

#         is_occluded = (rng.random() < amb.occluded_target_prob)
#         if is_occluded:
#             cv2.circle(raw_ref_crop, (500, 500), 250, 15, -1)

#         search_aff = cv2.warpAffine(search_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_h_aff = cv2.warpAffine(search_h_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_m_aff = cv2.warpAffine(search_m_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

#         search_warped = cv2.remap(search_aff, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_h_warped = cv2.remap(search_h_aff, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         search_m_warped = cv2.remap(search_m_aff, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

#         ref_final = SEMPhysicsEngine.process_sem_response(raw_ref_crop, raw_ref_h, raw_ref_m, phys, proc, PIXEL_SIZE_REF_NM, rng, is_search=False)
#         search_final = SEMPhysicsEngine.process_sem_response(search_warped, search_h_warped, search_m_warped, phys, proc, PIXEL_SIZE_SEARCH_NM, rng, is_search=True)

#         # EDGE-SAMPLED FOOTPRINT POLYGON (40 BOUNDARY POINTS FOR NONLINEAR CURVATURE)
#         half_w, half_h = 50.0, 50.0
#         edge_pts = []
#         for t in np.linspace(-half_w, half_w, 10): edge_pts.append((unwarped_gt_x + t, unwarped_gt_y - half_h))
#         for t in np.linspace(-half_h, half_h, 10): edge_pts.append((unwarped_gt_x + half_w, unwarped_gt_y + t))
#         for t in np.linspace(half_w, -half_w, 10): edge_pts.append((unwarped_gt_x + t, unwarped_gt_y + half_h))
#         for t in np.linspace(half_h, -half_h, 10): edge_pts.append((unwarped_gt_x - half_w, unwarped_gt_y + t))

#         transformed_poly = CompoundTransformEngine.forward_polygon(edge_pts, rot_matrix, disp_fwd_x, disp_fwd_y)

#         poly_np = np.array(transformed_poly, dtype=np.float32)
#         x, y, w_box, h_box = cv2.boundingRect(poly_np)
#         transformed_bbox = [round(float(x), 4), round(float(y), 4), round(float(x + w_box), 4), round(float(y + h_box), 4)]

#         residual_err = CompoundTransformEngine.test_sample_residual(rot_matrix, disp_fwd_x, disp_fwd_y, map_x, map_y)

#         ref_filename = f"{sample_id}.png"
#         search_filename = f"{sample_id}.png"

#         ref_path_full = os.path.join(self.ref_dir, ref_filename)
#         search_path_full = os.path.join(self.search_dir, search_filename)

#         cv2.imwrite(ref_path_full, ref_final)
#         cv2.imwrite(search_path_full, search_final)

#         if self.visualize:
#             preview_img = self.create_visual_preview(ref_final, search_final, transformed_bbox, transformed_poly, (final_gt_x, final_gt_y), cand_metas, sample_id, spec.architecture, difficulty)
#             cv2.imwrite(os.path.join(self.preview_dir, f"{sample_id}_preview.png"), preview_img)

#         return {
#             "sample_id": sample_id,
#             "generator_version": GENERATOR_VERSION,
#             "schema_version": SCHEMA_VERSION,
#             "sample_seed": sample_seed,
#             "architecture": spec.architecture,
#             "difficulty": difficulty,
#             "reference_path": f"reference/{ref_filename}",
#             "search_path": f"search/{search_filename}",
#             "reference_sha256": DatasetQAPipeline.compute_sha256(ref_path_full),
#             "search_sha256": DatasetQAPipeline.compute_sha256(search_path_full),
#             "gt_center_x": float(round(final_gt_x, 4)),
#             "gt_center_y": float(round(final_gt_y, 4)),
#             "unwarped_gt_center_x": float(round(unwarped_gt_x, 4)),
#             "unwarped_gt_center_y": float(round(unwarped_gt_y, 4)),
#             "transformed_polygon": [(round(px, 4), round(py, 4)) for px, py in transformed_poly],
#             "transformed_bbox": transformed_bbox,
#             "max_transform_residual_px": round(residual_err, 6),
#             "direct_marker_verification_err_px": round(direct_marker_verification_err_px, 6),
#             "requested_candidate_count": amb.deceptive_candidate_count,
#             "actual_visible_candidate_count": actual_cand_count,
#             "candidates": [asdict(c) for c in cand_metas],
#             "rotation_deg": float(round(angle_deg, 4)),
#             "scale_x": float(round(scale_x, 4)),
#             "scale_y": float(round(scale_y, 4)),
#             "pitch_x_nm": spec.pitch_x_nm,
#             "pitch_y_nm": spec.pitch_y_nm,
#             "beam_energy_keV": float(round(phys.beam_energy_keV, 3)),
#             "charging_strength": float(round(phys.charging_strength, 3)),
#             "ler_sigma_nm": float(round(proc.ler_sigma_nm, 3)),
#             "is_target_occluded": is_occluded
#         }

#     @staticmethod
#     def create_visual_preview(
#         ref_img: np.ndarray, 
#         search_img: np.ndarray, 
#         bbox: List[float], 
#         polygon: List[Tuple[float, float]], 
#         center_pt: Tuple[float, float], 
#         candidates: List[CandidateMetadata], 
#         sample_id: str, 
#         arch: str, 
#         diff: str
#     ) -> np.ndarray:
#         ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
#         search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

#         poly_pts = np.array([(int(px), int(py)) for px, py in polygon], dtype=np.int32).reshape((-1, 1, 2))
#         cv2.polylines(search_rgb, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

#         cx, cy = int(center_pt[0]), int(center_pt[1])
#         cv2.drawMarker(search_rgb, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

#         for cand in candidates:
#             c_x, c_y = int(cand.transformed_center_x_search_px), int(cand.transformed_center_y_search_px)
#             cv2.drawMarker(search_rgb, (c_x, c_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

#         combined = np.hstack((ref_rgb, search_rgb))
#         cv2.putText(combined, f"Ref (1000x1000, 1nm/px) - {sample_id} [{arch}]", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#         cv2.putText(combined, f"Search (1000x1000, 10nm/px) [{diff.upper()}] Cands: {len(candidates)}", (1030, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         return combined

#     def batch_generate(self, num_pairs: int = 30, arch_selection: str = "ALL") -> List[Dict[str, Any]]:
#         print("=" * 80)
#         print(f"DRIFT-SENSE: GENERATING {num_pairs} BENCHMARK SAMPLES (v{GENERATOR_VERSION})")
#         print(f"Output Directory  : {os.path.abspath(self.output_dir)}")
#         print(f"Global Base Seed  : {self.global_seed}")
#         print("=" * 80)

#         official_archs = ["DRAM", "FinFET"]
#         manifest_data = []
        
#         tier_schedule = ["easy"] * 8 + ["medium"] * 8 + ["hard"] * 8 + ["extreme"] * 6
#         if len(tier_schedule) < num_pairs:
#             tier_schedule.extend([self.difficulty] * (num_pairs - len(tier_schedule)))

#         for i in range(1, num_pairs + 1):
#             sample_id = f"sample_{i:03d}"
#             arch = arch_selection.upper() if arch_selection.upper() in official_archs else official_archs[(i - 1) % len(official_archs)]
#             diff = tier_schedule[i - 1] if arch_selection == "ALL" else self.difficulty
            
#             sample_seed = self.global_seed + i * 1000

#             meta = self.generate_single_sample(sample_id, arch, diff, sample_seed)
            
#             is_valid, errors = DatasetQAPipeline.validate_sample(self.output_dir, meta, self.strict)
#             meta["qa_passed"] = is_valid
#             meta["qa_errors"] = errors

#             if not is_valid and self.strict:
#                 raise ValueError(f"CRITICAL: Sample {sample_id} failed strict QA: {errors}")

#             manifest_data.append(meta)
#             print(f"[+] Sample {i:02d}/{num_pairs:02d} | ID: {sample_id} | Arch: {meta['architecture']:6s} | Tier: {meta['difficulty']:7s} | Cands: {meta['actual_visible_candidate_count']:2d} | Residual: {meta['max_transform_residual_px']:.4f} px | Marker Verif Err: {meta['direct_marker_verification_err_px']:.4f} px | QA: {'PASS' if is_valid else 'FAIL'}")

#         csv_path = os.path.join(self.output_dir, "metadata.csv")
#         fieldnames = [
#             "sample_id", "generator_version", "schema_version", "sample_seed", "architecture", "difficulty",
#             "reference_path", "search_path", "reference_sha256", "search_sha256", "gt_center_x", "gt_center_y",
#             "unwarped_gt_center_x", "unwarped_gt_center_y", "max_transform_residual_px", "direct_marker_verification_err_px",
#             "requested_candidate_count", "actual_visible_candidate_count", "rotation_deg", "scale_x", "scale_y", "pitch_x_nm", "pitch_y_nm",
#             "beam_energy_keV", "charging_strength", "ler_sigma_nm", "is_target_occluded", "qa_passed"
#         ]

#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
#             writer.writeheader()
#             writer.writerows(manifest_data)

#         with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
#             json.dump(manifest_data, f, indent=4)

#         manifest_doc = {
#             "generator_version": GENERATOR_VERSION,
#             "schema_version": SCHEMA_VERSION,
#             "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "global_seed": self.global_seed,
#             "total_samples": num_pairs,
#             "architecture_counts": {a: sum(1 for m in manifest_data if m["architecture"] == a) for a in official_archs},
#             "difficulty_counts": {d: sum(1 for m in manifest_data if m["difficulty"] == d) for d in ["easy", "medium", "hard", "extreme", "extreme_plus"]},
#             "all_samples_passed_qa": all(m["qa_passed"] for m in manifest_data),
#             "samples": manifest_data
#         }
#         with open(os.path.join(self.output_dir, "dataset_manifest.json"), "w") as f:
#             json.dump(manifest_doc, f, indent=4)

#         print("-" * 80)
#         print(f"[SUCCESS] Generation complete. Dataset Manifest exported to {os.path.join(self.output_dir, 'dataset_manifest.json')}")
#         print("=" * 80)

#         return manifest_data


# # ==============================================================================
# # CLI ENTRY POINT
# # ==============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description=f"Drift-Sense Physical SEM Dataset Generator v{GENERATOR_VERSION}",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("--architecture", type=str, default="ALL", choices=["ALL", "DRAM", "FinFET"])
#     parser.add_argument("--num_pairs", type=int, default=30)
#     parser.add_argument("--output_dir", type=str, default="./synthetic_sem_dataset")
#     parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "extreme", "extreme_plus"])
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--visualize", action="store_true")
#     parser.add_argument("--strict", action="store_true", help="Enforces strict QA checks; aborts on any invalid sample.")
#     parser.add_argument("--validate_6", action="store_true", help="Runs 6-sample diagnostic validation suite.")
#     parser.add_argument("--validate_existing", type=str, default=None, help="Validates an existing dataset directory.")
#     parser.add_argument("--evaluate_predictions", type=str, default=None, help="Path to ground truth metadata.json for evaluation.")
#     parser.add_argument("--predictions_json", type=str, default=None, help="Path to predictions JSON for evaluation metrics.")

#     args = parser.parse_args()

#     if args.validate_existing:
#         manifest_path = os.path.join(args.validate_existing, "metadata.json")
#         if not os.path.exists(manifest_path):
#             print(f"[ERROR] metadata.json not found in {args.validate_existing}")
#             sys.exit(1)
#         with open(manifest_path, 'r') as f:
#             meta_list = json.load(f)
        
#         all_ok = True
#         for m in meta_list:
#             ok, errs = DatasetQAPipeline.validate_sample(args.validate_existing, m, args.strict)
#             print(f"Sample {m['sample_id']}: {'PASS' if ok else 'FAIL'} | {errs if errs else ''}")
#             if not ok: all_ok = False
#         print(f"Overall QA Validation Result: {'PASS' if all_ok else 'FAIL'}")
#         return

#     if args.evaluate_predictions and args.predictions_json:
#         with open(args.evaluate_predictions, 'r') as f:
#             gt_list = json.load(f)
#         with open(args.predictions_json, 'r') as f:
#             preds_list = json.load(f)
        
#         res = EvaluationEngine.evaluate_predictions(gt_list, preds_list)
#         print("=== EVALUATION BENCHMARK METRICS ===")
#         print(json.dumps(res, indent=4))
#         return

#     if args.validate_6:
#         print("=== RUNNING 6-SAMPLE DIAGNOSTIC VALIDATION SUITE ===")
#         generator = SEMDatasetGenerator(
#             output_dir=args.output_dir,
#             visualize=True,
#             difficulty="medium",
#             seed=args.seed,
#             strict=args.strict
#         )
#         generator.batch_generate(num_pairs=6, arch_selection="ALL")
#         print("[SUCCESS] Diagnostic Validation Suite complete!")
#         return

#     generator = SEMDatasetGenerator(
#         output_dir=args.output_dir,
#         visualize=args.visualize,
#         difficulty=args.difficulty,
#         seed=args.seed,
#         strict=args.strict
#     )
#     generator.batch_generate(num_pairs=args.num_pairs, arch_selection=args.architecture)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
================================================================================
DRIFT-SENSE: HYBRID PHYSICAL SEM & FABRICATION SIMULATOR (v10.0.0-AMAT-INDUSTRIAL)
Applied Materials | Semicon India Hackathon 2026 | TEAM - ShunyaVeer
================================================================================

Physical & Theoretical Foundations:
  1. Beam PSF & Scattering Range: Kanaya-Okayama (1972) primary electron range 
     scattering model for energy-dependent beam interaction volume blur.
  2. Topographic SE Yield: Seiler (1983) surface-normal SE-II yield model 
     delta(theta) = delta_0 * sec(theta)^alpha for sidewall edge-brightening.
  3. Sidewall Roughness: Palasantzas (1993) spatial frequency domain Fourier 
     power spectral density (PSD) model for Line-Edge Roughness (LER).
  4. Detector Response & Noise: Reimer (1998) Poisson electron shot noise, 
     Everhart-Thornley PMT collector gain, and Fixed-Pattern Noise (FPN).
  5. Resist Collapse: Namatsu et al. (1995) lithographic capillary force model
     for localized high-aspect-ratio feature toppling.
"""

import os
import sys
import csv
import hashlib
import math
import json
import time
import argparse
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

GENERATOR_VERSION = "10.0.0-AMAT-INDUSTRIAL"
SCHEMA_VERSION = "1.0.0"

# Physical Dimension Standards
REFERENCE_SIZE_PX = 1000        # Reference image resolution (1000 x 1000 px)
SEARCH_SIZE_PX = 1000           # Search image resolution (1000 x 1000 px)
PIXEL_SIZE_REF_NM = 1.0         # 1.0 nm / pixel
PIXEL_SIZE_SEARCH_NM = 10.0     # 10.0 nm / pixel
SCALE_FACTOR = 10               # 10x physical magnification ratio
REF_FOV_IN_MASTER_NM = 1000.0   # 1000 nm FOV
SEARCH_FOV_IN_MASTER_NM = 10000.0 # 10,000 nm FOV


# ==============================================================================
# NUMPY JSON ENCODER (PREVENTS INT64 / FLOAT32 SERIALIZATION ERRORS)
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy scalar and array types cleanly."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# ==============================================================================
# PARAMETERS & DATACLASSES
# ==============================================================================

@dataclass
class PhysicsParams:
    beam_energy_keV: float
    se_alpha: float
    psf_sigma_ref_nm: float
    psf_sigma_srch_nm: float
    psf_sigma_y_nm: float
    astigmatism_angle_deg: float
    dwell_time_ref: float
    dwell_time_srch: float
    readout_noise_std: float
    speckle_noise_std: float
    salt_pepper_prob: float
    detector_gain: float
    detector_sat_threshold: float
    gamma_exponent: float
    baseline_black_level: float
    fpn_strength: float
    charging_strength: float
    vignetting_strength: float
    raster_drift_velocity_px: float
    raster_vibration_amp_px: float
    scanline_jitter_prob: float
    scan_direction: int # 0: Horizontal scanlines, 1: Vertical scanlines
    enable_elastic_warp: bool
    elastic_alpha_px: float
    elastic_sigma_px: float
    enable_barrel_distortion: bool
    barrel_k1: float


@dataclass
class ProcessVariationParams:
    ler_sigma_nm: float
    ler_correlation_length_nm: float
    ler_hurst: float
    cd_taper_pct: float
    cmp_dishing_strength: float
    opc_corner_rounding_radius: int
    etch_bias_nm: float
    enable_pattern_collapse: bool


@dataclass
class BenchmarkAmbiguityParams:
    difficulty: str
    pure_array_probability: float
    deceptive_candidate_count: int
    cell_similarity_pct: float
    repeated_defect_count: int
    max_rotation_deg: float
    scale_range_x: Tuple[float, float]
    scale_range_y: Tuple[float, float]
    occluded_target_prob: float


@dataclass
class LayoutSpec:
    architecture: str
    pitch_x_nm: int
    pitch_y_nm: int
    line_w_x_nm: int
    line_w_y_nm: int
    feature_size_nm: int
    base_gray: int
    metal_gray: int
    contact_gray: int
    macro_width_nm: int
    dram_stagger_mode: str
    dram_wave_amp_nm: int
    dram_wave_freq1: float
    dram_wave_freq2: float
    dram_wave_phase: float
    dram_pad_angle: float
    finfet_cluster_size: int


@dataclass
class CandidateMetadata:
    candidate_id: str
    candidate_type: str # 'NATURAL_PERIODIC' or 'ADVERSARIAL_TRAP'
    unwarped_center_x_search_px: float
    unwarped_center_y_search_px: float
    transformed_center_x_search_px: float
    transformed_center_y_search_px: float
    distance_from_gt_transformed_px: float
    ssim_to_target_clean: float
    ncc_to_target_clean: float


# ==============================================================================
# COORDINATE CONVERSION ENGINE
# ==============================================================================

class CoordinateTransformer:
    """Explicit conversion routines between Master NM, Master PX, Ref PX, and Search PX."""

    @staticmethod
    def master_nm_to_px(val_nm: float) -> int:
        return int(round(val_nm)) # 1.0 nm / pixel in master canvas

    @staticmethod
    def master_px_to_search_px(x_master: float, y_master: float, search_start_x: float, search_start_y: float) -> Tuple[float, float]:
        x_search = (x_master - search_start_x) / 10.0
        y_search = (y_master - search_start_y) / 10.0
        return float(x_search), float(y_search)

    @staticmethod
    def search_px_to_master_px(x_search: float, y_search: float, search_start_x: float, search_start_y: float) -> Tuple[float, float]:
        x_master = search_start_x + x_search * 10.0
        y_master = search_start_y + y_search * 10.0
        return float(x_master), float(y_master)


# ==============================================================================
# STRUCTURAL METRICS ENGINE (SSIM & NCC)
# ==============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two patches."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Normalized Cross-Correlation (NCC) between two patches."""
    i1 = img1.astype(np.float64) - np.mean(img1)
    i2 = img2.astype(np.float64) - np.mean(img2)
    denom = np.sqrt(np.sum(i1**2) * np.sum(i2**2))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(i1 * i2) / denom)


# ==============================================================================
# COMPOUND GEOMETRIC TRANSFORM ENGINE
# ==============================================================================

class CompoundTransformEngine:
    """
    Unified geometric transform engine providing continuous forward mapping F(x,y)
    for points, polygon footprints, and inverse fixed-point remap grid generation.
    """

    @staticmethod
    def build_anisotropic_affine_matrix(center: Tuple[float, float], angle_deg: float, scale_x: float, scale_y: float) -> np.ndarray:
        cx, cy = center
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        m00 = scale_x * cos_a
        m01 = -scale_y * sin_a
        m10 = scale_x * sin_a
        m11 = scale_y * cos_a

        tx = cx - (m00 * cx + m01 * cy)
        ty = cy - (m10 * cx + m11 * cy)

        return np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)

    @classmethod
    def create_compound_warp_field(
        cls, 
        width: int, 
        height: int, 
        phys: PhysicsParams, 
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        disp_fwd_x = np.zeros((height, width), dtype=np.float32)
        disp_fwd_y = np.zeros((height, width), dtype=np.float32)

        if phys.enable_elastic_warp and phys.elastic_alpha_px > 0:
            rand_x = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
            rand_y = rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32)
            
            disp_fwd_x += gaussian_filter(rand_x, phys.elastic_sigma_px).astype(np.float32) * phys.elastic_alpha_px
            disp_fwd_y += gaussian_filter(rand_y, phys.elastic_sigma_px).astype(np.float32) * phys.elastic_alpha_px

        if phys.enable_barrel_distortion and phys.barrel_k1 > 0:
            cx, cy = width / 2.0, height / 2.0
            x_norm = (grid_x - cx) / cx
            y_norm = (grid_y - cy) / cy
            r_sq = x_norm**2 + y_norm**2
            
            disp_fwd_x += ((grid_x - cx) * (phys.barrel_k1 * r_sq)).astype(np.float32)
            disp_fwd_y += ((grid_y - cy) * (phys.barrel_k1 * r_sq)).astype(np.float32)

        if phys.raster_drift_velocity_px > 0 or phys.raster_vibration_amp_px > 0:
            indices = np.arange(height if phys.scan_direction == 0 else width, dtype=np.float32)
            drift_val = phys.raster_drift_velocity_px * indices + \
                        phys.raster_vibration_amp_px * np.sin(2.0 * np.pi * indices / 40.0).astype(np.float32)
            
            if phys.scan_direction == 0:
                disp_fwd_x += np.tile(drift_val[:, None], (1, width))
            else:
                disp_fwd_y += np.tile(drift_val[None, :], (height, 1))

        if phys.scanline_jitter_prob > 0:
            dim_size = height if phys.scan_direction == 0 else width
            jitter = np.zeros(dim_size, dtype=np.float32)
            mask = rng.random(dim_size) < phys.scanline_jitter_prob
            jitter[mask] = rng.uniform(-3.0, 3.0, size=np.sum(mask)).astype(np.float32)
            
            if phys.scan_direction == 0:
                disp_fwd_x += np.tile(jitter[:, None], (1, width))
            else:
                disp_fwd_y += np.tile(jitter[None, :], (height, 1))

        map_x = grid_x.copy()
        map_y = grid_y.copy()

        for _ in range(4):
            curr_x = np.clip(map_x, 0, width - 1.001)
            curr_y = np.clip(map_y, 0, height - 1.001)

            x0 = np.floor(curr_x).astype(np.int32)
            y0 = np.floor(curr_y).astype(np.int32)
            x1 = np.minimum(x0 + 1, width - 1)
            y1 = np.minimum(y0 + 1, height - 1)

            dx = curr_x - x0
            dy = curr_y - y0

            interp_dx = (1 - dx) * (1 - dy) * disp_fwd_x[y0, x0] + \
                        dx * (1 - dy) * disp_fwd_x[y0, x1] + \
                        (1 - dx) * dy * disp_fwd_x[y1, x0] + \
                        dx * dy * disp_fwd_x[y1, x1]

            interp_dy = (1 - dx) * (1 - dy) * disp_fwd_y[y0, x0] + \
                        dx * (1 - dy) * disp_fwd_y[y0, x1] + \
                        (1 - dx) * dy * disp_fwd_y[y1, x0] + \
                        dx * dy * disp_fwd_y[y1, x1]

            map_x = grid_x - interp_dx
            map_y = grid_y - interp_dy

        return (
            disp_fwd_x.astype(np.float32), 
            disp_fwd_y.astype(np.float32), 
            map_x.astype(np.float32), 
            map_y.astype(np.float32)
        )

    @classmethod
    def forward_point(
        cls, 
        pt_initial: Tuple[float, float], 
        affine_matrix: np.ndarray, 
        disp_fwd_x: np.ndarray, 
        disp_fwd_y: np.ndarray
    ) -> Tuple[float, float]:
        pt_aff = np.array([pt_initial[0], pt_initial[1], 1.0], dtype=np.float64)
        x_aff, y_aff = affine_matrix.dot(pt_aff)[:2]

        h, w = disp_fwd_x.shape
        x_c = np.clip(x_aff, 0.0, w - 1.001)
        y_c = np.clip(y_aff, 0.0, h - 1.001)

        x0, y0 = int(np.floor(x_c)), int(np.floor(y_c))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        dx, dy = x_c - x0, y_c - y0

        dx_val = (1 - dx) * (1 - dy) * disp_fwd_x[y0, x0] + \
                 dx * (1 - dy) * disp_fwd_x[y0, x1] + \
                 (1 - dx) * dy * disp_fwd_x[y1, x0] + \
                 dx * dy * disp_fwd_x[y1, x1]

        dy_val = (1 - dx) * (1 - dy) * disp_fwd_y[y0, x0] + \
                 dx * (1 - dy) * disp_fwd_y[y0, x1] + \
                 (1 - dx) * dy * disp_fwd_y[y1, x0] + \
                 dx * dy * disp_fwd_y[y1, x1]

        return float(x_aff + dx_val), float(y_aff + dy_val)

    @classmethod
    def forward_polygon(
        cls, 
        polygon_pts: List[Tuple[float, float]], 
        affine_matrix: np.ndarray, 
        disp_fwd_x: np.ndarray, 
        disp_fwd_y: np.ndarray
    ) -> List[Tuple[float, float]]:
        return [cls.forward_point(pt, affine_matrix, disp_fwd_x, disp_fwd_y) for pt in polygon_pts]

    @classmethod
    def test_sample_residual(
        cls, 
        rot_matrix: np.ndarray, 
        disp_fwd_x: np.ndarray, 
        disp_fwd_y: np.ndarray, 
        map_x: np.ndarray, 
        map_y: np.ndarray
    ) -> float:
        test_points = [(x, y) for x in [100.0, 250.0, 500.0, 750.0, 900.0] for y in [100.0, 250.0, 500.0, 750.0, 900.0]]
        max_err = 0.0

        for pt in test_points:
            gt_x, gt_y = cls.forward_point(pt, rot_matrix, disp_fwd_x, disp_fwd_y)

            delta_img = np.zeros((1000, 1000), dtype=np.float32)
            ix, iy = int(round(pt[0])), int(round(pt[1]))
            delta_img[iy, ix] = 1.0

            warped_aff = cv2.warpAffine(delta_img, rot_matrix, (1000, 1000), flags=cv2.INTER_LINEAR)
            warped_final = cv2.remap(warped_aff, map_x, map_y, interpolation=cv2.INTER_LINEAR)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(warped_final)
            px, py = max_loc

            if 1 <= px < 999 and 1 <= py < 999:
                dx = (warped_final[py, px + 1] - warped_final[py, px - 1]) / (2.0 * (2.0 * warped_final[py, px] - warped_final[py, px + 1] - warped_final[py, px - 1] + 1e-6))
                dy = (warped_final[py + 1, px] - warped_final[py - 1, px]) / (2.0 * (2.0 * warped_final[py, px] - warped_final[py + 1, px] - warped_final[py - 1, px] + 1e-6))
                meas_x = px + np.clip(dx, -0.5, 0.5)
                meas_y = py + np.clip(dy, -0.5, 0.5)
            else:
                meas_x, meas_y = float(px), float(py)

            err = math.sqrt((meas_x - gt_x)**2 + (meas_y - gt_y)**2)
            max_err = max(max_err, err)

        return float(max_err)


# ==============================================================================
# PHYSICAL SEM RENDERING ENGINE
# ==============================================================================

class SEMPhysicsEngine:

    MATERIAL_PROPERTIES = {
        0: {"name": "OXIDE_DIELECTRIC",   "yield": 0.65, "height_nm": 0.0},
        1: {"name": "SILICON_SUBSTRATE", "yield": 1.00, "height_nm": 0.0},
        2: {"name": "POLY_SILICON",       "yield": 1.25, "height_nm": 45.0},
        3: {"name": "METAL_INTERCONNECT", "yield": 1.85, "height_nm": 55.0},
        4: {"name": "CONTACT_VIA",        "yield": 2.10, "height_nm": 70.0}
    }

    @staticmethod
    def calculate_kanaya_okayama_psf_px(beam_energy_keV: float, base_sigma_nm: float, pixel_size_nm: float) -> float:
        r_ko_um = 0.0276 * 28.085 * (beam_energy_keV ** 1.67) / (2.33 * (14.0 ** 0.899))
        r_ko_nm = r_ko_um * 1000.0
        eff_sigma_nm = base_sigma_nm + 0.015 * r_ko_nm
        return float(eff_sigma_nm / pixel_size_nm)

    @staticmethod
    def synthesize_palasantzas_ler_field(shape: Tuple[int, int], sigma_px: float, xi_px: float, hurst: float, rng: np.random.Generator) -> np.ndarray:
        if sigma_px <= 0.0:
            return np.zeros(shape, dtype=np.float32)

        h, w = shape
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        f_sq = fx**2 + fy**2

        psd = (2.0 * (sigma_px ** 2) * (xi_px**2)) / ((1.0 + (2.0 * np.pi * np.sqrt(f_sq) * xi_px) ** 2) ** (hurst + 1.0))
        psd[0, 0] = 0.0

        random_phase = np.exp(1j * rng.uniform(0, 2.0 * np.pi, (h, w)))
        spectrum = np.sqrt(np.maximum(psd, 0.0)) * random_phase

        field = np.fft.ifft2(spectrum).real.astype(np.float32)
        std_val = np.std(field)
        if std_val > 1e-6:
            field *= (sigma_px / std_val)
        return field

    @classmethod
    def apply_edge_specific_ler(
        cls, 
        canvas: np.ndarray, 
        height_map: np.ndarray, 
        proc: ProcessVariationParams, 
        pixel_size_nm: float, 
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        if proc.ler_sigma_nm <= 0.0:
            return canvas, height_map

        sigma_px = proc.ler_sigma_nm / pixel_size_nm
        xi_px = proc.ler_correlation_length_nm / pixel_size_nm

        gx = cv2.Sobel(canvas.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(canvas.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)
        edge_mask = (edge_mag > 15.0).astype(np.float32)

        ler_field_x = cls.synthesize_palasantzas_ler_field(canvas.shape, sigma_px, xi_px, proc.ler_hurst, rng)
        ler_field_y = cls.synthesize_palasantzas_ler_field(canvas.shape, sigma_px, xi_px, proc.ler_hurst, rng)

        h, w = canvas.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        map_x = (grid_x + ler_field_x * edge_mask).astype(np.float32)
        map_y = (grid_y + ler_field_y * edge_mask).astype(np.float32)

        warped_canvas = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_height = cv2.remap(height_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return warped_canvas, warped_height

    @classmethod
    def apply_seiler_topographic_yield(cls, height_map_nm: np.ndarray, mat_map: np.ndarray, alpha: float = 0.38) -> np.ndarray:
        dz_dx = cv2.Sobel(height_map_nm, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        dz_dy = cv2.Sobel(height_map_nm, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        sec_theta = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)

        base_yield = np.ones_like(height_map_nm, dtype=np.float64)
        for mat_id, prop in cls.MATERIAL_PROPERTIES.items():
            base_yield[mat_map == mat_id] = prop["yield"]

        se_yield = base_yield * (1.0 + alpha * (sec_theta - 1.0))
        return se_yield.astype(np.float32)

    @staticmethod
    def apply_anisotropic_blur(img_float: np.ndarray, sigma_x_px: float, sigma_y_px: float, angle_deg: float) -> np.ndarray:
        kx_size = max(3, int(6 * sigma_x_px) | 1)
        ky_size = max(3, int(6 * sigma_y_px) | 1)
        ksize = max(kx_size, ky_size)

        kx = cv2.getGaussianKernel(ksize, sigma_x_px)
        ky = cv2.getGaussianKernel(ksize, sigma_y_px)
        kernel = ky @ kx.T

        if abs(angle_deg) > 1e-3:
            M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle_deg, 1.0)
            kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
            kernel /= np.maximum(np.sum(kernel), 1e-6)

        return cv2.filter2D(img_float, -1, kernel)

    @classmethod
    def process_sem_response(
        cls, 
        clean_raster: np.ndarray, 
        height_map: np.ndarray, 
        mat_map: np.ndarray, 
        phys: PhysicsParams, 
        proc: ProcessVariationParams, 
        pixel_size_nm: float, 
        rng: np.random.Generator, 
        is_search: bool = False
    ) -> np.ndarray:
        ler_raster, height_ler = cls.apply_edge_specific_ler(clean_raster, height_map, proc, pixel_size_nm, rng)

        se_yield = cls.apply_seiler_topographic_yield(height_ler, mat_map, alpha=phys.se_alpha)
        img_float = (ler_raster.astype(np.float32) / 255.0) * se_yield

        if is_search and (phys.charging_strength > 0 or phys.vignetting_strength > 0):
            h, w = img_float.shape
            gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
            r_sq = ((gx - w / 2.0) ** 2 + (gy - h / 2.0) ** 2) / ((w / 2.0) ** 2 + (h / 2.0) ** 2)
            vignette = 1.0 - phys.vignetting_strength * r_sq
            
            charge_pool = 0.0
            if phys.charging_strength > 0:
                cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
                charge_pool = phys.charging_strength * np.exp(-((gx - cx)**2 + (gy - cy)**2) / (2.0 * (0.35 * w)**2))
            
            img_float = np.clip(img_float * vignette + charge_pool, 0.0, 10.0)

        base_sigma_nm = phys.psf_sigma_srch_nm if is_search else phys.psf_sigma_ref_nm
        eff_sigma_x_px = cls.calculate_kanaya_okayama_psf_px(phys.beam_energy_keV, base_sigma_nm, pixel_size_nm)
        eff_sigma_y_px = phys.psf_sigma_y_nm / pixel_size_nm
        blurred = cls.apply_anisotropic_blur(img_float, eff_sigma_x_px, eff_sigma_y_px, phys.astigmatism_angle_deg)

        dose = phys.dwell_time_srch if is_search else phys.dwell_time_ref
        electron_counts = np.maximum(blurred, 1e-6) * dose
        noisy_shot = rng.poisson(electron_counts).astype(np.float32) / dose
        
        speckle = rng.normal(1.0, phys.speckle_noise_std, size=img_float.shape).astype(np.float32)
        readout = rng.normal(0, phys.readout_noise_std, size=img_float.shape).astype(np.float32)
        combined = np.clip(noisy_shot * speckle + readout, 0.0, 10.0)

        if phys.salt_pepper_prob > 0:
            sp = rng.random(combined.shape)
            combined[sp < (phys.salt_pepper_prob / 2.0)] = 0.0
            combined[sp > (1.0 - phys.salt_pepper_prob / 2.0)] = 2.0

        norm_signal = combined / np.maximum(np.percentile(combined, 99.5), 1e-5)
        gamma_corr = np.power(np.clip(norm_signal, 0, 1), phys.gamma_exponent)
        saturated = phys.detector_gain * (gamma_corr / (1.0 + gamma_corr / max(0.1, phys.detector_sat_threshold)))

        fpn = rng.normal(1.0, phys.fpn_strength, img_float.shape).astype(np.float32)
        output = saturated * fpn * 255.0 + phys.baseline_black_level

        return np.clip(output, 0, 255).astype(np.uint8)


# ==============================================================================
# CANONICAL LAYOUT ENGINE
# ==============================================================================

class LayoutEngine:

    @staticmethod
    def get_benchmark_ambiguity_preset(difficulty: str) -> BenchmarkAmbiguityParams:
        d = difficulty.lower()
        if d == "easy":
            return BenchmarkAmbiguityParams("easy", 0.10, 1, 0.70, 0, 0.0, (1.0, 1.0), (1.0, 1.0), 0.0)
        elif d == "medium":
            return BenchmarkAmbiguityParams("medium", 0.40, 2, 0.88, 4, 1.5, (0.98, 1.02), (0.98, 1.02), 0.0)
        elif d == "hard":
            return BenchmarkAmbiguityParams("hard", 0.75, 5, 0.96, 15, 3.0, (0.96, 1.04), (0.96, 1.04), 0.10)
        elif d == "extreme_plus":
            return BenchmarkAmbiguityParams("extreme_plus", 0.99, 12, 0.998, 40, 7.5, (0.92, 1.08), (0.92, 1.08), 0.30)
        else: # extreme
            return BenchmarkAmbiguityParams("extreme", 0.95, 10, 0.995, 30, 5.0, (0.94, 1.06), (0.94, 1.06), 0.20)

    @classmethod
    def generate_random_spec(cls, arch_choice: str, difficulty: str, rng: np.random.Generator) -> Tuple[LayoutSpec, ProcessVariationParams, PhysicsParams, BenchmarkAmbiguityParams]:
        amb = cls.get_benchmark_ambiguity_preset(difficulty)
        is_hard_or_extreme = difficulty.lower() in ["hard", "extreme", "extreme_plus"]

        spec = LayoutSpec(
            architecture=arch_choice,
            pitch_x_nm=rng.integers(180, 260),
            pitch_y_nm=rng.integers(160, 240),
            line_w_x_nm=rng.integers(28, 48),
            line_w_y_nm=rng.integers(30, 50),
            feature_size_nm=rng.integers(20, 36),
            base_gray=25,
            metal_gray=175,
            contact_gray=220,
            macro_width_nm=rng.integers(200, 300),
            dram_stagger_mode=str(rng.choice(['STAGGER_50', 'HEX'])) if is_hard_or_extreme else 'STAGGER_50',
            dram_wave_amp_nm=rng.integers(4, 16) if is_hard_or_extreme else 0,
            dram_wave_freq1=rng.uniform(0.8, 1.8),
            dram_wave_freq2=rng.uniform(0.3, 0.9),
            dram_wave_phase=rng.uniform(0.0, 2.0 * np.pi),
            dram_pad_angle=rng.uniform(-30.0, 30.0) if is_hard_or_extreme else 0.0,
            finfet_cluster_size=rng.integers(2, 4)
        )

        proc = ProcessVariationParams(
            ler_sigma_nm=rng.uniform(0.5, 1.2) if difficulty == "easy" else rng.uniform(1.2, 3.0),
            ler_correlation_length_nm=rng.uniform(12.0, 25.0),
            ler_hurst=rng.uniform(0.65, 0.85),
            cd_taper_pct=rng.uniform(0.0, 0.02) if difficulty == "easy" else rng.uniform(0.02, 0.08),
            cmp_dishing_strength=rng.uniform(0.0, 0.05) if difficulty == "easy" else rng.uniform(0.05, 0.22),
            opc_corner_rounding_radius=2 if difficulty == "easy" else rng.integers(2, 6),
            etch_bias_nm=rng.uniform(-1.0, 1.0) if difficulty == "easy" else rng.uniform(-3.5, 3.5),
            enable_pattern_collapse=is_hard_or_extreme
        )

        dwell_srch = rng.uniform(120.0, 200.0) if difficulty == "easy" else (
            rng.uniform(50.0, 100.0) if difficulty == "medium" else (
                rng.uniform(15.0, 50.0) if difficulty == "hard" else rng.uniform(8.0, 25.0)
            )
        )

        phys = PhysicsParams(
            beam_energy_keV=rng.uniform(1.0, 2.0),
            se_alpha=rng.uniform(0.32, 0.42),
            psf_sigma_ref_nm=rng.uniform(0.4, 0.6),
            psf_sigma_srch_nm=rng.uniform(1.0, 1.6) if difficulty == "easy" else rng.uniform(1.4, 2.5),
            psf_sigma_y_nm=rng.uniform(0.5, 0.7) if difficulty == "easy" else rng.uniform(1.0, 2.8),
            astigmatism_angle_deg=0.0 if difficulty == "easy" else rng.uniform(0.0, 360.0),
            dwell_time_ref=rng.uniform(220.0, 320.0),
            dwell_time_srch=dwell_srch,
            readout_noise_std=0.02 if difficulty == "easy" else rng.uniform(0.03, 0.08),
            speckle_noise_std=0.01 if difficulty == "easy" else rng.uniform(0.02, 0.05),
            salt_pepper_prob=0.0 if not is_hard_or_extreme else rng.uniform(0.0005, 0.004),
            detector_gain=1.0 if difficulty == "easy" else rng.uniform(0.8, 1.5),
            detector_sat_threshold=1.0,
            gamma_exponent=1.0 if difficulty == "easy" else rng.uniform(0.7, 1.4),
            baseline_black_level=30.0 if difficulty == "easy" else rng.uniform(20.0, 60.0),
            fpn_strength=0.005 if difficulty == "easy" else rng.uniform(0.008, 0.025),
            charging_strength=0.0 if difficulty == "easy" else rng.uniform(0.05, 0.35),
            vignetting_strength=0.0 if difficulty == "easy" else rng.uniform(0.05, 0.22),
            raster_drift_velocity_px=0.0 if difficulty == "easy" else rng.uniform(0.01, 0.06),
            raster_vibration_amp_px=0.0 if difficulty == "easy" else rng.uniform(0.2, 1.2),
            scanline_jitter_prob=0.0 if difficulty == "easy" else rng.uniform(0.005, 0.03),
            scan_direction=int(rng.choice([0, 1])),
            enable_elastic_warp=is_hard_or_extreme,
            elastic_alpha_px=rng.uniform(2.0, 4.5),
            elastic_sigma_px=rng.uniform(15.0, 25.0),
            enable_barrel_distortion=is_hard_or_extreme,
            barrel_k1=rng.uniform(1e-6, 3.0e-6)
        )

        return spec, proc, phys, amb

    @classmethod
    def render_dram_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec) -> List[Tuple[int, int]]:
        h, w = sub.shape
        contact_coords = []

        for y in range(0, h, spec.pitch_y_nm):
            cv2.rectangle(sub, (0, y), (w, y + spec.line_w_y_nm), spec.metal_gray, -1)
            cv2.rectangle(sub_h, (0, y), (w, y + spec.line_w_y_nm), 55.0, -1)
            cv2.rectangle(sub_m, (0, y), (w, y + spec.line_w_y_nm), 3, -1)

        if spec.dram_wave_amp_nm > 0:
            amp = spec.dram_wave_amp_nm
            f1, ph = spec.dram_wave_freq1, spec.dram_wave_phase
            for x in range(0, w, spec.pitch_x_nm):
                pts = [(int(x + amp * np.sin(2 * np.pi * f1 * y / spec.pitch_y_nm + ph)), y) for y in range(0, h, 20)]
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(sub, [pts_arr], False, spec.metal_gray + 20, spec.line_w_x_nm)
                cv2.polylines(sub_h, [pts_arr], False, 45.0, spec.line_w_x_nm)
                cv2.polylines(sub_m, [pts_arr], False, 2, spec.line_w_x_nm)
        else:
            for x in range(0, w, spec.pitch_x_nm):
                cv2.rectangle(sub, (x, 0), (x + spec.line_w_x_nm, h), spec.metal_gray + 20, -1)
                cv2.rectangle(sub_h, (x, 0), (x + spec.line_w_x_nm, h), 45.0, -1)
                cv2.rectangle(sub_m, (x, 0), (x + spec.line_w_x_nm, h), 2, -1)

        rx = spec.feature_size_nm
        ry = max(6, int(spec.feature_size_nm * 0.6))
        for y in range(spec.pitch_y_nm // 2, h, spec.pitch_y_nm):
            for x in range(spec.pitch_x_nm // 2, w, spec.pitch_x_nm):
                if abs(spec.dram_pad_angle) > 1.0:
                    cv2.ellipse(sub, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, spec.contact_gray, -1)
                    cv2.ellipse(sub_h, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, 70.0, -1)
                    cv2.ellipse(sub_m, (x, y), (rx, ry), spec.dram_pad_angle, 0, 360, 4, -1)
                else:
                    cv2.circle(sub, (x, y), rx, spec.contact_gray, -1)
                    cv2.circle(sub_h, (x, y), rx, 70.0, -1)
                    cv2.circle(sub_m, (x, y), rx, 4, -1)
                contact_coords.append((x, y))

        return contact_coords

    @classmethod
    def render_finfet_canonical(cls, sub: np.ndarray, sub_h: np.ndarray, sub_m: np.ndarray, spec: LayoutSpec) -> List[Tuple[int, int]]:
        h, w = sub.shape
        contact_coords = []

        fin_x_positions = []
        x = 0
        while x < w:
            for c in range(spec.finfet_cluster_size):
                fx = x + c * spec.pitch_x_nm
                if fx < w:
                    cv2.rectangle(sub, (fx, 0), (fx + spec.line_w_x_nm, h), spec.metal_gray - 20, -1)
                    cv2.rectangle(sub_h, (fx, 0), (fx + spec.line_w_x_nm, h), 50.0, -1)
                    cv2.rectangle(sub_m, (fx, 0), (fx + spec.line_w_x_nm, h), 2, -1)
                    fin_x_positions.append(fx + spec.line_w_x_nm // 2)
            x += spec.finfet_cluster_size * spec.pitch_x_nm + 120

        for y in range(0, h, spec.pitch_y_nm):
            gate_w = spec.line_w_y_nm
            cv2.rectangle(sub, (0, y), (w, y + gate_w), spec.metal_gray + 35, -1)
            cv2.rectangle(sub_h, (0, y), (w, y + gate_w), 60.0, -1)
            cv2.rectangle(sub_m, (0, y), (w, y + gate_w), 3, -1)

            for fx in fin_x_positions:
                cy = y + gate_w + spec.pitch_y_nm // 4
                if cy < h:
                    cv2.rectangle(sub, (fx - 10, cy - 8), (fx + 10, cy + 8), spec.contact_gray, -1)
                    cv2.rectangle(sub_h, (fx - 10, cy - 8), (fx + 10, cy + 8), 70.0, -1)
                    cv2.rectangle(sub_m, (fx - 10, cy - 8), (fx + 10, cy + 8), 4, -1)
                    contact_coords.append((fx, cy))

        return contact_coords

    @classmethod
    def apply_process_variations(
        cls, 
        canvas: np.ndarray, 
        height_map: np.ndarray, 
        mat_map: np.ndarray, 
        proc: ProcessVariationParams
    ):
        h, w = canvas.shape

        if proc.cd_taper_pct > 0:
            gy, gx = np.ogrid[:h, :w]
            r_norm = np.sqrt((gx - w / 2.0)**2 + (gy - h / 2.0)**2) / (w / 2.0)
            taper_factor = 1.0 + proc.cd_taper_pct * (r_norm - 0.5)
            canvas[:] = np.clip(canvas.astype(np.float32) * taper_factor, 0, 255).astype(np.uint8)

        if proc.cmp_dishing_strength > 0:
            gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            dishing_mask = 1.0 - proc.cmp_dishing_strength * gy
            height_map *= dishing_mask

        if abs(proc.etch_bias_nm) >= 1.0:
            ksize = max(3, int(abs(proc.etch_bias_nm)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if proc.etch_bias_nm > 0:
                canvas[:] = cv2.dilate(canvas, kernel)
            else:
                canvas[:] = cv2.erode(canvas, kernel)

        if proc.opc_corner_rounding_radius > 1:
            r = proc.opc_corner_rounding_radius
            kernel_opc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            canvas[:] = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel_opc)

    @classmethod
    def apply_physical_pattern_collapse(cls, canvas: np.ndarray, height_map: np.ndarray, rng: np.random.Generator):
        h, w = canvas.shape
        num = rng.integers(2, 5)
        for _ in range(num):
            cx, cy = rng.integers(1000, w - 1000), rng.integers(1000, h - 1000)
            rad = rng.integers(100, 250)

            gy, gx = np.ogrid[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]
            dist_sq = (gx - cx)**2 + (gy - cy)**2
            mask = np.exp(-dist_sq / (2.0 * (rad / 2.0)**2)).astype(np.float32)

            shift_x = mask * 12.0
            grid_y, grid_x = np.mgrid[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)].astype(np.float32)

            sub_c = canvas[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]
            sub_h = height_map[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)]

            canvas[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)] = cv2.remap(
                sub_c, grid_x - shift_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            height_map[max(0, cy - rad):min(h, cy + rad), max(0, cx - rad):min(w, cx + rad)] = cv2.remap(
                sub_h, grid_x - shift_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

    @classmethod
    def inject_geometry_aware_defects(
        cls, 
        canvas: np.ndarray, 
        height_map: np.ndarray, 
        mat_map: np.ndarray, 
        contact_coords: List[Tuple[int, int]], 
        count: int, 
        rng: np.random.Generator
    ):
        if count <= 0 or not contact_coords:
            return

        defects = ['MISSING_VIA', 'LINE_BRIDGING', 'LINE_CUT']
        indices = rng.choice(len(contact_coords), size=min(count, len(contact_coords)), replace=False)

        for idx in indices:
            px, py = contact_coords[idx]
            dtype = rng.choice(defects)

            if dtype == 'MISSING_VIA':
                cv2.circle(canvas, (px, py), 14, 25, -1)
                cv2.circle(height_map, (px, py), 14, 0.0, -1)
                cv2.circle(mat_map, (px, py), 14, 0, -1)
            elif dtype == 'LINE_BRIDGING':
                cv2.line(canvas, (px, py - 20), (px, py + 20), 195, 8)
                cv2.line(height_map, (px, py - 20), (px, py + 20), 55.0, 8)
                cv2.line(mat_map, (px, py - 20), (px, py + 20), 3, 8)
            else:
                cv2.line(canvas, (px - 20, py), (px + 20, py), 25, 10)
                cv2.line(height_map, (px - 20, py), (px + 20, py), 0.0, 10)
                cv2.line(mat_map, (px - 20, py), (px + 20, py), 0, 10)

    @classmethod
    def render_canvas(
        cls, 
        width: int, 
        height: int, 
        spec: LayoutSpec, 
        amb: BenchmarkAmbiguityParams, 
        proc: ProcessVariationParams, 
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        canvas = np.full((height, width), spec.base_gray, dtype=np.uint8)
        height_map = np.zeros((height, width), dtype=np.float32)
        mat_map = np.zeros((height, width), dtype=np.uint8)

        if spec.architecture == "FinFET":
            contacts = cls.render_finfet_canonical(canvas, height_map, mat_map, spec)
        else:
            contacts = cls.render_dram_canonical(canvas, height_map, mat_map, spec)

        cls.apply_process_variations(canvas, height_map, mat_map, proc)

        if proc.enable_pattern_collapse:
            cls.apply_physical_pattern_collapse(canvas, height_map, rng)

        cls.inject_geometry_aware_defects(canvas, height_map, mat_map, contacts, amb.repeated_defect_count, rng)

        return canvas, height_map, mat_map, contacts

    @classmethod
    def render_deceptive_candidates_in_search_fov(
        cls, 
        master_canvas: np.ndarray, 
        master_height: np.ndarray, 
        master_mat: np.ndarray, 
        spec: LayoutSpec, 
        amb: BenchmarkAmbiguityParams, 
        search_start_x: int, 
        search_start_y: int, 
        ref_start_x: int, 
        ref_start_y: int, 
        gt_transformed_center_search_px: Tuple[float, float],
        rot_matrix: np.ndarray,
        disp_fwd_x: np.ndarray,
        disp_fwd_y: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[int, List[CandidateMetadata]]:
        if amb.deceptive_candidate_count <= 0:
            return 0, []

        block_w = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)
        block_h = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)

        target_crop = master_canvas[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()
        target_h = master_height[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()
        target_m = master_mat[ref_start_y:ref_start_y + block_h, ref_start_x:ref_start_x + block_w].copy()

        placed_centers = []
        candidate_metas: List[CandidateMetadata] = []

        search_fov_px = CoordinateTransformer.master_nm_to_px(SEARCH_FOV_IN_MASTER_NM)
        min_x = search_start_x + 200
        max_x = search_start_x + search_fov_px - block_w - 200
        min_y = search_start_y + 200
        max_y = search_start_y + search_fov_px - block_h - 200

        attempts = 0
        max_attempts = amb.deceptive_candidate_count * 10

        while len(placed_centers) < amb.deceptive_candidate_count and attempts < max_attempts:
            attempts += 1
            cx = int(rng.integers(min_x, max_x))
            cy = int(rng.integers(min_y, max_y))

            if abs(cx - ref_start_x) < block_w and abs(cy - ref_start_y) < block_h:
                continue

            if any(abs(cx - px) < block_w and abs(cy - py) < block_h for px, py in placed_centers):
                continue

            cand_crop = target_crop.copy()
            cand_height = target_h.copy()
            cand_mat = target_m.copy()

            num_mods = max(1, int((1.0 - amb.cell_similarity_pct) * 100))
            for _ in range(num_mods):
                mx = rng.integers(50, block_w - 50)
                my = rng.integers(50, block_h - 50)
                val = spec.base_gray if rng.random() > 0.5 else spec.contact_gray
                
                cv2.rectangle(cand_crop, (mx - 8, my - 8), (mx + 8, my + 8), val, -1)
                cv2.rectangle(cand_height, (mx - 8, my - 8), (mx + 8, my + 8), 70.0 if val > 100 else 0.0, -1)
                cv2.rectangle(cand_mat, (mx - 8, my - 8), (mx + 8, my + 8), 4 if val > 100 else 0, -1)

            # Metrics calculated on clean layout patches before non-linear SEM response
            ssim_score = compute_ssim(target_crop, cand_crop)
            ncc_score = compute_ncc(target_crop, cand_crop)

            master_canvas[cy:cy + block_h, cx:cx + block_w] = cand_crop
            master_height[cy:cy + block_h, cx:cx + block_w] = cand_height
            master_mat[cy:cy + block_h, cx:cx + block_w] = cand_mat

            placed_centers.append((cx, cy))

            unwarped_px_x, unwarped_px_y = CoordinateTransformer.master_px_to_search_px(
                cx + block_w / 2.0, cy + block_h / 2.0, search_start_x, search_start_y
            )

            trans_px_x, trans_px_y = CompoundTransformEngine.forward_point(
                (unwarped_px_x, unwarped_px_y), rot_matrix, disp_fwd_x, disp_fwd_y
            )

            dist_gt = math.sqrt((trans_px_x - gt_transformed_center_search_px[0])**2 + (trans_px_y - gt_transformed_center_search_px[1])**2)

            cand_meta = CandidateMetadata(
                candidate_id=f"cand_{len(placed_centers):02d}",
                candidate_type="ADVERSARIAL_TRAP",
                unwarped_center_x_search_px=float(round(unwarped_px_x, 4)),
                unwarped_center_y_search_px=float(round(unwarped_px_y, 4)),
                transformed_center_x_search_px=float(round(trans_px_x, 4)),
                transformed_center_y_search_px=float(round(trans_px_y, 4)),
                distance_from_gt_transformed_px=float(round(dist_gt, 4)),
                ssim_to_target_clean=float(round(ssim_score, 4)),
                ncc_to_target_clean=float(round(ncc_score, 4))
            )
            candidate_metas.append(cand_meta)

        return len(placed_centers), candidate_metas


# ==============================================================================
# QA & DATA INTEGRITY PIPELINE
# ==============================================================================

class DatasetQAPipeline:

    @staticmethod
    def compute_sha256(filepath: str) -> str:
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def validate_sample(cls, sample_dir: str, meta: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
        errors = []

        ref_full_path = os.path.join(sample_dir, meta["reference_path"])
        search_full_path = os.path.join(sample_dir, meta["search_path"])

        if not os.path.exists(ref_full_path) or not os.path.exists(search_full_path):
            errors.append("Reference or Search image file missing.")
            return False, errors

        ref_img = cv2.imread(ref_full_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_full_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            errors.append("Failed to load Reference or Search PNG.")
            return False, errors

        if ref_img.shape != (REFERENCE_SIZE_PX, REFERENCE_SIZE_PX):
            errors.append(f"Invalid Reference size: {ref_img.shape}")

        if search_img.shape != (SEARCH_SIZE_PX, SEARCH_SIZE_PX):
            errors.append(f"Invalid Search size: {search_img.shape}")

        gt_x, gt_y = meta["gt_center_x"], meta["gt_center_y"]
        if not (0 <= gt_x <= SEARCH_SIZE_PX and 0 <= gt_y <= SEARCH_SIZE_PX):
            errors.append(f"GT Center ({gt_x}, {gt_y}) outside Search bounds.")

        polygon = meta.get("transformed_polygon", [])
        if len(polygon) < 4:
            errors.append("Transformed GT polygon missing or malformed.")
        else:
            poly_np = np.array(polygon, dtype=np.float32)
            area = cv2.contourArea(poly_np)
            if area < 100.0:
                errors.append(f"Transformed polygon area too small: {area}")

        # Tier-aware QA residual thresholds
        diff = str(meta.get("difficulty", "medium")).lower()
        max_res_thresh = 0.20 if diff == "easy" else 1.50
        max_marker_thresh = 0.10 if diff == "easy" else 0.25

        if meta.get("max_transform_residual_px", 0.0) > max_res_thresh:
            errors.append(f"Transform residual ({meta['max_transform_residual_px']:.4f} px) exceeded threshold {max_res_thresh} px")

        if meta.get("direct_marker_verification_err_px", 0.0) > max_marker_thresh:
            errors.append(f"Direct Marker Verification Error ({meta['direct_marker_verification_err_px']:.4f} px) exceeded threshold {max_marker_thresh} px")

        for cand in meta.get("candidates", []):
            if cand["distance_from_gt_transformed_px"] < 50.0:
                errors.append(f"Candidate {cand['candidate_id']} too close to Ground Truth target.")

        is_valid = (len(errors) == 0)
        return is_valid, errors


# ==============================================================================
# EVALUATION & BENCHMARKING ENGINE
# ==============================================================================

class EvaluationEngine:

    @staticmethod
    def evaluate_predictions(gt_metadata_list: List[Dict[str, Any]], predictions_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        pred_map = {p["sample_id"]: p for p in predictions_list}
        errors = []
        confidences = []

        for gt in gt_metadata_list:
            sid = gt["sample_id"]
            if sid not in pred_map:
                continue

            pred = pred_map[sid]
            px, py = pred["predicted_x"], pred["predicted_y"]
            conf = pred.get("confidence", 1.0)
            tx, ty = gt["gt_center_x"], gt["gt_center_y"]

            err = math.sqrt((px - tx)**2 + (py - ty)**2)
            errors.append(err)
            confidences.append(conf)

        n = max(1, len(errors))
        err_arr = np.array(errors)
        conf_arr = np.array(confidences)

        ap_metrics = {}
        for tol in [1.0, 3.0, 5.0, 10.0]:
            tp = (err_arr <= tol).astype(np.float32)
            sort_idx = np.argsort(-conf_arr)
            tp_sorted = tp[sort_idx]
            
            acc_tp = np.cumsum(tp_sorted)
            recalls = acc_tp / max(1.0, np.sum(tp))
            precisions = acc_tp / (np.arange(len(tp_sorted)) + 1)
            
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                p_mask = recalls >= t
                p_max = np.max(precisions[p_mask]) if np.any(p_mask) else 0.0
                ap += p_max / 11.0
            
            ap_metrics[f"AP@{int(tol)}px"] = round(float(ap), 4)

        return {
            "num_evaluated": len(errors),
            "mae_px": float(np.mean(err_arr)),
            "median_err_px": float(np.median(err_arr)),
            "rmse_px": float(np.sqrt(np.mean(err_arr**2))),
            "p95_err_px": float(np.percentile(err_arr, 95)),
            "ap_scores": ap_metrics,
            "accuracy_pct_at_1px": round(float(np.mean(err_arr <= 1.0) * 100.0), 2),
            "accuracy_pct_at_3px": round(float(np.mean(err_arr <= 3.0) * 100.0), 2),
            "accuracy_pct_at_5px": round(float(np.mean(err_arr <= 5.0) * 100.0), 2),
            "accuracy_pct_at_10px": round(float(np.mean(err_arr <= 10.0) * 100.0), 2)
        }


# ==============================================================================
# MAIN DATASET GENERATOR CLASS
# ==============================================================================

class SEMDatasetGenerator:

    def __init__(self, output_dir: str = "./synthetic_sem_dataset", visualize: bool = False, difficulty: str = "medium", seed: int = 42, strict: bool = False):
        self.output_dir = output_dir
        self.visualize = visualize
        self.difficulty = difficulty
        self.global_seed = seed
        self.strict = strict

        self.ref_dir = os.path.join(output_dir, "reference")
        self.search_dir = os.path.join(output_dir, "search")
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.search_dir, exist_ok=True)

        if self.visualize:
            self.preview_dir = os.path.join(output_dir, "previews")
            os.makedirs(self.preview_dir, exist_ok=True)

    def generate_single_sample(self, sample_id: str, arch_choice: str, difficulty: str, sample_seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(sample_seed)
        master_w = CoordinateTransformer.master_nm_to_px(12000.0)
        master_h = CoordinateTransformer.master_nm_to_px(12000.0)

        spec, proc, phys, amb = LayoutEngine.generate_random_spec(arch_choice, difficulty, rng)
        master_canvas, master_height, master_mat, contacts = LayoutEngine.render_canvas(master_w, master_h, spec, amb, proc, rng)

        search_fov_px = CoordinateTransformer.master_nm_to_px(SEARCH_FOV_IN_MASTER_NM)
        ref_fov_px = CoordinateTransformer.master_nm_to_px(REF_FOV_IN_MASTER_NM)

        search_start_x = int(rng.integers(500, master_w - search_fov_px - 500))
        search_start_y = int(rng.integers(500, master_h - search_fov_px - 500))

        unwarped_gt_x = float(rng.uniform(200.0, 800.0))
        unwarped_gt_y = float(rng.uniform(200.0, 800.0))

        ref_start_x = int(search_start_x + unwarped_gt_x * 10.0 - ref_fov_px / 2.0)
        ref_start_y = int(search_start_y + unwarped_gt_y * 10.0 - ref_fov_px / 2.0)

        # Build Continuous Forward Geometric Transform Engine
        angle_deg = float(rng.uniform(-amb.max_rotation_deg, amb.max_rotation_deg))
        scale_x = float(rng.uniform(*amb.scale_range_x))
        scale_y = float(rng.uniform(*amb.scale_range_y))

        rot_matrix = CompoundTransformEngine.build_anisotropic_affine_matrix((500.0, 500.0), angle_deg, scale_x, scale_y)
        disp_fwd_x, disp_fwd_y, map_x, map_y = CompoundTransformEngine.create_compound_warp_field(SEARCH_SIZE_PX, SEARCH_SIZE_PX, phys, rng)

        # Transformed Target GT Center
        final_gt_x, final_gt_y = CompoundTransformEngine.forward_point((unwarped_gt_x, unwarped_gt_y), rot_matrix, disp_fwd_x, disp_fwd_y)

        # DIRECT CONTINUOUS SUB-PIXEL GAUSSIAN PULSE VERIFICATION PASS
        gy_m, gx_m = np.mgrid[0:SEARCH_SIZE_PX, 0:SEARCH_SIZE_PX].astype(np.float32)
        dist_sq_m = (gx_m - unwarped_gt_x)**2 + (gy_m - unwarped_gt_y)**2
        marker_canvas = np.exp(-dist_sq_m / (2.0 * (1.0**2))).astype(np.float32)

        marker_aff = cv2.warpAffine(marker_canvas, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR)
        marker_warped = cv2.remap(marker_aff, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        min_v, max_v, min_l, max_l = cv2.minMaxLoc(marker_warped)
        px_m, py_m = max_l
        if 1 <= px_m < SEARCH_SIZE_PX - 1 and 1 <= py_m < SEARCH_SIZE_PX - 1:
            dx_m = (marker_warped[py_m, px_m + 1] - marker_warped[py_m, px_m - 1]) / (2.0 * (2.0 * marker_warped[py_m, px_m] - marker_warped[py_m, px_m + 1] - marker_warped[py_m, px_m - 1] + 1e-6))
            dy_m = (marker_warped[py_m + 1, px_m] - marker_warped[py_m - 1, px_m]) / (2.0 * (2.0 * marker_warped[py_m, px_m] - marker_warped[py_m + 1, px_m] - marker_warped[py_m - 1, px_m] + 1e-6))
            meas_gt_x = px_m + np.clip(dx_m, -0.5, 0.5)
            meas_gt_y = py_m + np.clip(dy_m, -0.5, 0.5)
        else:
            meas_gt_x, meas_gt_y = float(px_m), float(py_m)

        direct_marker_verification_err_px = float(math.sqrt((meas_gt_x - final_gt_x)**2 + (meas_gt_y - final_gt_y)**2))

        # Render Deceptive Candidates and Forward-Transform Coordinates
        actual_cand_count, cand_metas = LayoutEngine.render_deceptive_candidates_in_search_fov(
            master_canvas, master_height, master_mat, spec, amb, 
            search_start_x, search_start_y, ref_start_x, ref_start_y,
            (final_gt_x, final_gt_y), rot_matrix, disp_fwd_x, disp_fwd_y, rng
        )

        # Crop Search region
        raw_search = master_canvas[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]
        raw_search_h = master_height[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]
        raw_search_m = master_mat[search_start_y:search_start_y + search_fov_px, search_start_x:search_start_x + search_fov_px]

        search_ds = cv2.resize(cv2.GaussianBlur(raw_search, (7, 7), 1.5), (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)
        search_h_ds = cv2.resize(raw_search_h, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)
        search_m_ds = cv2.resize(raw_search_m, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_NEAREST)

        # Crop Reference region
        raw_ref_crop = master_canvas[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()
        raw_ref_h = master_height[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()
        raw_ref_m = master_mat[ref_start_y:ref_start_y + ref_fov_px, ref_start_x:ref_start_x + ref_fov_px].copy()

        is_occluded = bool(rng.random() < amb.occluded_target_prob)
        if is_occluded:
            cv2.circle(raw_ref_crop, (500, 500), 250, 15, -1)

        search_aff = cv2.warpAffine(search_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        search_h_aff = cv2.warpAffine(search_h_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        search_m_aff = cv2.warpAffine(search_m_ds, rot_matrix, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        search_warped = cv2.remap(search_aff, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        search_h_warped = cv2.remap(search_h_aff, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        search_m_warped = cv2.remap(search_m_aff, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        ref_final = SEMPhysicsEngine.process_sem_response(raw_ref_crop, raw_ref_h, raw_ref_m, phys, proc, PIXEL_SIZE_REF_NM, rng, is_search=False)
        search_final = SEMPhysicsEngine.process_sem_response(search_warped, search_h_warped, search_m_warped, phys, proc, PIXEL_SIZE_SEARCH_NM, rng, is_search=True)

        # EDGE-SAMPLED FOOTPRINT POLYGON (40 BOUNDARY POINTS FOR NONLINEAR CURVATURE)
        half_w, half_h = 50.0, 50.0
        edge_pts = []
        for t in np.linspace(-half_w, half_w, 10): edge_pts.append((unwarped_gt_x + t, unwarped_gt_y - half_h))
        for t in np.linspace(-half_h, half_h, 10): edge_pts.append((unwarped_gt_x + half_w, unwarped_gt_y + t))
        for t in np.linspace(half_w, -half_w, 10): edge_pts.append((unwarped_gt_x + t, unwarped_gt_y + half_h))
        for t in np.linspace(half_h, -half_h, 10): edge_pts.append((unwarped_gt_x - half_w, unwarped_gt_y + t))

        transformed_poly = CompoundTransformEngine.forward_polygon(edge_pts, rot_matrix, disp_fwd_x, disp_fwd_y)

        poly_np = np.array(transformed_poly, dtype=np.float32)
        x, y, w_box, h_box = cv2.boundingRect(poly_np)
        transformed_bbox = [round(float(x), 4), round(float(y), 4), round(float(x + w_box), 4), round(float(y + h_box), 4)]

        # SAMPLE-SPECIFIC SPATIAL RESIDUAL TEST
        residual_err = CompoundTransformEngine.test_sample_residual(rot_matrix, disp_fwd_x, disp_fwd_y, map_x, map_y)

        ref_filename = f"{sample_id}.png"
        search_filename = f"{sample_id}.png"

        ref_path_full = os.path.join(self.ref_dir, ref_filename)
        search_path_full = os.path.join(self.search_dir, search_filename)

        cv2.imwrite(ref_path_full, ref_final)
        cv2.imwrite(search_path_full, search_final)

        if self.visualize:
            preview_img = self.create_visual_preview(ref_final, search_final, transformed_bbox, transformed_poly, (final_gt_x, final_gt_y), cand_metas, sample_id, spec.architecture, difficulty)
            cv2.imwrite(os.path.join(self.preview_dir, f"{sample_id}_preview.png"), preview_img)

        # Force native Python types for JSON compatibility
        return {
            "sample_id": str(sample_id),
            "generator_version": str(GENERATOR_VERSION),
            "schema_version": str(SCHEMA_VERSION),
            "sample_seed": int(sample_seed),
            "architecture": str(spec.architecture),
            "difficulty": str(difficulty),
            "reference_path": f"reference/{ref_filename}",
            "search_path": f"search/{search_filename}",
            "reference_sha256": str(DatasetQAPipeline.compute_sha256(ref_path_full)),
            "search_sha256": str(DatasetQAPipeline.compute_sha256(search_path_full)),
            "gt_center_x": float(round(final_gt_x, 4)),
            "gt_center_y": float(round(final_gt_y, 4)),
            "unwarped_gt_center_x": float(round(unwarped_gt_x, 4)),
            "unwarped_gt_center_y": float(round(unwarped_gt_y, 4)),
            "transformed_polygon": [(float(round(px, 4)), float(round(py, 4))) for px, py in transformed_poly],
            "transformed_bbox": [float(b) for b in transformed_bbox],
            "max_transform_residual_px": float(round(residual_err, 6)),
            "direct_marker_verification_err_px": float(round(direct_marker_verification_err_px, 6)),
            "requested_candidate_count": int(amb.deceptive_candidate_count),
            "actual_visible_candidate_count": int(actual_cand_count),
            "candidates": [asdict(c) for c in cand_metas],
            "rotation_deg": float(round(angle_deg, 4)),
            "scale_x": float(round(scale_x, 4)),
            "scale_y": float(round(scale_y, 4)),
            "pitch_x_nm": int(spec.pitch_x_nm),
            "pitch_y_nm": int(spec.pitch_y_nm),
            "beam_energy_keV": float(round(phys.beam_energy_keV, 3)),
            "charging_strength": float(round(phys.charging_strength, 3)),
            "ler_sigma_nm": float(round(proc.ler_sigma_nm, 3)),
            "is_target_occluded": bool(is_occluded)
        }

    @staticmethod
    def create_visual_preview(
        ref_img: np.ndarray, 
        search_img: np.ndarray, 
        bbox: List[float], 
        polygon: List[Tuple[float, float]], 
        center_pt: Tuple[float, float], 
        candidates: List[CandidateMetadata], 
        sample_id: str, 
        arch: str, 
        diff: str
    ) -> np.ndarray:
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

        poly_pts = np.array([(int(px), int(py)) for px, py in polygon], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(search_rgb, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cx, cy = int(center_pt[0]), int(center_pt[1])
        cv2.drawMarker(search_rgb, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

        for cand in candidates:
            c_x, c_y = int(cand.transformed_center_x_search_px), int(cand.transformed_center_y_search_px)
            cv2.drawMarker(search_rgb, (c_x, c_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

        combined = np.hstack((ref_rgb, search_rgb))
        cv2.putText(combined, f"Ref (1000x1000, 1nm/px) - {sample_id} [{arch}]", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined, f"Search (1000x1000, 10nm/px) [{diff.upper()}] Cands: {len(candidates)}", (1030, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return combined

    def batch_generate(self, num_pairs: int = 30, arch_selection: str = "ALL") -> List[Dict[str, Any]]:
        print("=" * 80)
        print(f"DRIFT-SENSE: GENERATING {num_pairs} BENCHMARK SAMPLES (v{GENERATOR_VERSION})")
        print(f"Output Directory  : {os.path.abspath(self.output_dir)}")
        print(f"Global Base Seed  : {self.global_seed}")
        print("=" * 80)

        official_archs = ["DRAM", "FinFET"]
        manifest_data = []
        
        tier_schedule = ["easy"] * 8 + ["medium"] * 8 + ["hard"] * 8 + ["extreme"] * 6
        if len(tier_schedule) < num_pairs:
            tier_schedule.extend([self.difficulty] * (num_pairs - len(tier_schedule)))

        for i in range(1, num_pairs + 1):
            sample_id = f"sample_{i:03d}"
            arch = arch_selection.upper() if arch_selection.upper() in official_archs else official_archs[(i - 1) % len(official_archs)]
            diff = tier_schedule[i - 1] if arch_selection == "ALL" else self.difficulty
            
            sample_seed = self.global_seed + i * 1000

            meta = self.generate_single_sample(sample_id, arch, diff, sample_seed)
            
            is_valid, errors = DatasetQAPipeline.validate_sample(self.output_dir, meta, self.strict)
            meta["qa_passed"] = is_valid
            meta["qa_errors"] = errors

            if not is_valid and self.strict:
                raise ValueError(f"CRITICAL: Sample {sample_id} failed strict QA: {errors}")

            manifest_data.append(meta)
            print(f"[+] Sample {i:02d}/{num_pairs:02d} | ID: {sample_id} | Arch: {meta['architecture']:6s} | Tier: {meta['difficulty']:7s} | Cands: {meta['actual_visible_candidate_count']:2d} | Residual: {meta['max_transform_residual_px']:.4f} px | Marker Verif Err: {meta['direct_marker_verification_err_px']:.4f} px | QA: {'PASS' if is_valid else 'FAIL'}")

        csv_path = os.path.join(self.output_dir, "metadata.csv")
        fieldnames = [
            "sample_id", "generator_version", "schema_version", "sample_seed", "architecture", "difficulty",
            "reference_path", "search_path", "reference_sha256", "search_sha256", "gt_center_x", "gt_center_y",
            "unwarped_gt_center_x", "unwarped_gt_center_y", "max_transform_residual_px", "direct_marker_verification_err_px",
            "requested_candidate_count", "actual_visible_candidate_count", "rotation_deg", "scale_x", "scale_y", "pitch_x_nm", "pitch_y_nm",
            "beam_energy_keV", "charging_strength", "ler_sigma_nm", "is_target_occluded", "qa_passed"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(manifest_data)

        # JSON Dump using NumpyEncoder to prevent int64 / float32 serialization errors
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(manifest_data, f, indent=4, cls=NumpyEncoder)

        manifest_doc = {
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_seed": self.global_seed,
            "total_samples": num_pairs,
            "architecture_counts": {a: sum(1 for m in manifest_data if m["architecture"] == a) for a in official_archs},
            "difficulty_counts": {d: sum(1 for m in manifest_data if m["difficulty"] == d) for d in ["easy", "medium", "hard", "extreme", "extreme_plus"]},
            "all_samples_passed_qa": all(m["qa_passed"] for m in manifest_data),
            "samples": manifest_data
        }
        with open(os.path.join(self.output_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest_doc, f, indent=4, cls=NumpyEncoder)

        print("-" * 80)
        print(f"[SUCCESS] Generation complete. Dataset Manifest exported to {os.path.join(self.output_dir, 'dataset_manifest.json')}")
        print("=" * 80)

        return manifest_data


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Drift-Sense Physical SEM Dataset Generator v{GENERATOR_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--architecture", type=str, default="ALL", choices=["ALL", "DRAM", "FinFET"])
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./synthetic_sem_dataset")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "extreme", "extreme_plus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Enforces strict QA checks; aborts on any invalid sample.")
    parser.add_argument("--validate_6", action="store_true", help="Runs 6-sample diagnostic validation suite.")
    parser.add_argument("--validate_existing", type=str, default=None, help="Validates an existing dataset directory.")
    parser.add_argument("--evaluate_predictions", type=str, default=None, help="Path to ground truth metadata.json for evaluation.")
    parser.add_argument("--predictions_json", type=str, default=None, help="Path to predictions JSON for evaluation metrics.")

    args = parser.parse_args()

    if args.validate_existing:
        manifest_path = os.path.join(args.validate_existing, "metadata.json")
        if not os.path.exists(manifest_path):
            print(f"[ERROR] metadata.json not found in {args.validate_existing}")
            sys.exit(1)
        with open(manifest_path, 'r') as f:
            meta_list = json.load(f)
        
        all_ok = True
        for m in meta_list:
            ok, errs = DatasetQAPipeline.validate_sample(args.validate_existing, m, args.strict)
            print(f"Sample {m['sample_id']}: {'PASS' if ok else 'FAIL'} | {errs if errs else ''}")
            if not ok: all_ok = False
        print(f"Overall QA Validation Result: {'PASS' if all_ok else 'FAIL'}")
        return

    if args.evaluate_predictions and args.predictions_json:
        with open(args.evaluate_predictions, 'r') as f:
            gt_list = json.load(f)
        with open(args.predictions_json, 'r') as f:
            preds_list = json.load(f)
        
        res = EvaluationEngine.evaluate_predictions(gt_list, preds_list)
        print("=== EVALUATION BENCHMARK METRICS ===")
        print(json.dumps(res, indent=4, cls=NumpyEncoder))
        return

    if args.validate_6:
        print("=== RUNNING 6-SAMPLE DIAGNOSTIC VALIDATION SUITE ===")
        generator = SEMDatasetGenerator(
            output_dir=args.output_dir,
            visualize=True,
            difficulty="medium",
            seed=args.seed,
            strict=args.strict
        )
        generator.batch_generate(num_pairs=6, arch_selection="ALL")
        print("[SUCCESS] Diagnostic Validation Suite complete!")
        return

    generator = SEMDatasetGenerator(
        output_dir=args.output_dir,
        visualize=args.visualize,
        difficulty=args.difficulty,
        seed=args.seed,
        strict=args.strict
    )
    generator.batch_generate(num_pairs=args.num_pairs, arch_selection=args.architecture)


if __name__ == "__main__":
    main()