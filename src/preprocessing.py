"""Stage A — Image Conditioning and Preprocessing Engine.

Implements physics-grounded SEM image preprocessing:
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Edge-preserving bilateral / guided noise suppression
- Spatial gradient computation (Sobel/Scharr)
- High-pass / Palasantzas-aware Line-Edge Roughness (LER) residual extraction

References:
- Seiler (1983) — Secondary Electron Yield & Contrast Physics
- Palasantzas (1993) — Nanoscale Surface & Line-Edge Roughness Modeling
"""

from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in range [0.0, 1.0]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img_float = img.astype(np.float32)
    min_val, max_val = img_float.min(), img_float.max()
    if max_val > min_val:
        return (img_float - min_val) / (max_val - min_val)
    return img_float


def apply_clahe(img: np.ndarray, clip_limit: float = 2.5, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for SEM contrast normalization."""
    if img.dtype != np.uint8:
        img_u8 = (np.clip(normalize_image(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(img_u8)
    return equalized.astype(np.float32) / 255.0


def edge_preserving_filter(img: np.ndarray, d: int = 5, sigma_color: float = 0.1, sigma_space: float = 3.0) -> np.ndarray:
    """Apply bilateral filtering to suppress electron shot noise while preserving edge boundaries."""
    img_u8 = (np.clip(normalize_image(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_u8, d=d, sigmaColor=sigma_color * 255.0, sigmaSpace=sigma_space)
    return filtered.astype(np.float32) / 255.0


def compute_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial gradient magnitude and orientation angle maps using Scharr operators."""
    img_norm = normalize_image(img)
    gx = cv2.Scharr(img_norm, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_norm, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    orientation = cv2.phase(gx, gy, angleInDegrees=False)
    return magnitude, orientation, gx, gy


def extract_highpass_ler(img: np.ndarray, sigma_lowpass: float = 4.0) -> np.ndarray:
    """Extract high-frequency Line-Edge Roughness (LER) residual map by removing low-frequency envelope."""
    img_norm = normalize_image(img)
    lowpass = ndimage.gaussian_filter(img_norm, sigma=sigma_lowpass)
    highpass = img_norm - lowpass
    return highpass.astype(np.float32)


def preprocess_sem_image(
    img: np.ndarray,
    use_clahe: bool = True,
    use_denoise: bool = True,
    clip_limit: float = 2.0
) -> dict[str, np.ndarray]:
    """Comprehensive image conditioning pipeline returning preprocessed feature channels.
    
    Returns:
        dict containing:
        - 'normalized': float32 image [0, 1]
        - 'enhanced': CLAHE enhanced image
        - 'denoised': Bilateral filtered image
        - 'grad_mag': Spatial gradient magnitude
        - 'grad_orient': Spatial gradient orientation
        - 'highpass_ler': High-frequency LER residual map
    """
    norm = normalize_image(img)
    enhanced = apply_clahe(img, clip_limit=clip_limit) if use_clahe else norm
    denoised = edge_preserving_filter(enhanced) if use_denoise else enhanced
    grad_mag, grad_orient, gx, gy = compute_gradients(denoised)
    highpass_ler = extract_highpass_ler(denoised)

    return {
        'normalized': norm,
        'enhanced': enhanced,
        'denoised': denoised,
        'grad_mag': grad_mag,
        'grad_orient': grad_orient,
        'grad_x': gx,
        'grad_y': gy,
        'highpass_ler': highpass_ler
    }
