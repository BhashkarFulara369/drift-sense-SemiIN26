"""Stage B — Spectral 2D Fourier Analysis Engine.

Implements 2D Fast Fourier Transform (FFT) power spectrum analysis, windowing,
and reciprocal-space peak extraction for spatial periodicity detection.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage


def compute_fft_spectrum(img: np.ndarray, apply_window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D shift-centered FFT complex spectrum and log magnitude spectrum."""
    img_f = img.astype(np.float32)
    h, w = img_f.shape

    if apply_window:
        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window = np.outer(win_y, win_x)
        img_f = img_f * window

    fft_raw = np.fft.fft2(img_f)
    fft_shift = np.fft.fftshift(fft_raw)
    magnitude_spectrum = np.log1p(np.abs(fft_shift))

    return fft_shift, magnitude_spectrum


def find_reciprocal_peaks(
    magnitude_spectrum: np.ndarray,
    min_distance: int = 12,
    threshold_rel: float = 0.10,
    dc_radius: int = 6,
    max_peaks: int = 20
) -> list[tuple[float, float, float]]:
    """Detect dominant reciprocal-space frequency peaks in the 2D FFT magnitude spectrum."""
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2

    spectrum_masked = magnitude_spectrum.copy()
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_dc = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    spectrum_masked[dist_from_dc <= dc_radius] = 0.0

    max_val = np.max(spectrum_masked)
    if max_val <= 0:
        return []

    threshold = max_val * threshold_rel
    local_max = ndimage.maximum_filter(spectrum_masked, size=min_distance) == spectrum_masked
    detected = local_max & (spectrum_masked >= threshold)

    peak_coords = np.argwhere(detected)
    peaks = []
    for py, px in peak_coords:
        mag = float(spectrum_masked[py, px])
        peaks.append((float(py), float(px), mag))

    peaks.sort(key=lambda item: item[2], reverse=True)
    return peaks[:max_peaks]


def compute_spectral_confidence(peaks: list[tuple[float, float, float]], spectrum: np.ndarray) -> float:
    """Calculate Peak-to-Sidelobe Ratio (PSLR) confidence metric for reciprocal space lattice peaks."""
    if not peaks:
        return 0.0
    top_mag = peaks[0][2]
    mean_bg = float(np.mean(spectrum))
    std_bg = float(np.std(spectrum)) + 1e-6

    if top_mag <= mean_bg:
        return 0.0

    pslr = (top_mag - mean_bg) / std_bg
    # Normalize PSLR score to range [0.0, 1.0]
    confidence = float(np.clip(pslr / 5.0, 0.0, 1.0))
    return confidence
