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