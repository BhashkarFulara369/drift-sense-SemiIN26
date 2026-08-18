"""Stage B — Reciprocal Lattice Vector & Pose Synchronizer Engine.

Estimates spatial lattice vectors, lattice orientation, scale factor, and
spectral confidence from 2D reciprocal-space Fourier peaks.
"""

from __future__ import annotations
import math
import numpy as np
from src.spectral import compute_fft_spectrum, find_reciprocal_peaks, compute_spectral_confidence


def estimate_lattice_parameters(
    img: np.ndarray,
    expected_scale_ratio: float = 10.0
) -> dict[str, float]:
    """Estimate spatial lattice orientation, dominant spatial pitch, and scale factor."""
    fft_shift, spectrum = compute_fft_spectrum(img, apply_window=True)
    peaks = find_reciprocal_peaks(spectrum, min_distance=12, threshold_rel=0.15, dc_radius=8, max_peaks=16)

    confidence = compute_spectral_confidence(peaks, spectrum)
    h, w = img.shape
    cy, cx = h // 2, w // 2

    if len(peaks) < 2 or confidence < 0.05:
        return {
            'rotation_deg': 0.0,
            'pitch_x_px': 0.0,
            'pitch_y_px': 0.0,
            'confidence': confidence,
            'is_periodic': False
        }

    angles = []
    frequencies = []
    mags = []
    for py, px, mag in peaks:
        dy = py - cy
        dx = px - cx
        r = math.sqrt(dx * dx + dy * dy)
        if r > 0:
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            angles.append(angle)
            frequencies.append(r)
            mags.append(mag)

    angles_arr = np.array(angles)
    mags_arr = np.array(mags)
    rad = np.radians(angles_arr * 2.0)
    
    # Magnitude-weighted angle sum
    sin_sum = np.sum(mags_arr * np.sin(rad))
    cos_sum = np.sum(mags_arr * np.cos(rad))
    dominant_angle_deg = (np.degrees(np.atan2(sin_sum, cos_sum)) / 2.0) % 180.0
    if dominant_angle_deg > 90.0:
        dominant_angle_deg -= 180.0

    # Separate frequencies into X-aligned and Y-aligned based on dominant angle
    freqs_x = []
    freqs_y = []
    
    for ang, freq in zip(angles_arr, frequencies):
        diff = min(abs(ang - dominant_angle_deg), 180 - abs(ang - dominant_angle_deg))
        if diff < 15.0 or diff > 165.0:  # Aligned with dominant angle
            freqs_x.append(freq)
        elif abs(diff - 90.0) < 15.0:    # Orthogonal to dominant angle
            freqs_y.append(freq)

    # Use minimum fundamental frequency instead of median, to catch the primary pitch, not harmonics
    freq_x = min(freqs_x) if freqs_x else (min(frequencies) if frequencies else 0.0)
    freq_y = min(freqs_y) if freqs_y else (min(frequencies) if frequencies else 0.0)

    pitch_x = (h / freq_x) if freq_x > 0 else 0.0
    pitch_y = (h / freq_y) if freq_y > 0 else 0.0

    return {
        'rotation_deg': float(dominant_angle_deg),
        'pitch_x_px': float(pitch_x),
        'pitch_y_px': float(pitch_y),
        'confidence': confidence,
        'is_periodic': True
    }


def synchronize_spectral_pose(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    nominal_scale: float = 10.0
) -> dict[str, float]:
    """Estimate relative rotation angle and scale factor between Reference and Search images."""
    ref_lattice = estimate_lattice_parameters(ref_img, expected_scale_ratio=1.0)
    search_lattice = estimate_lattice_parameters(search_img, expected_scale_ratio=nominal_scale)

    joint_confidence = min(ref_lattice['confidence'], search_lattice['confidence'])

    if ref_lattice['is_periodic'] and search_lattice['is_periodic']:
        rel_rotation = search_lattice['rotation_deg'] - ref_lattice['rotation_deg']
        if rel_rotation > 180.0:
            rel_rotation -= 360.0
        elif rel_rotation < -180.0:
            rel_rotation += 360.0

        # Allow robust rotation search range up to +/- 45 degrees
        rel_rotation = float(np.clip(rel_rotation, -45.0, 45.0))

        # Scale estimation bounded reasonably for OOD conditions
        if search_lattice['pitch_x_px'] > 0 and ref_lattice['pitch_x_px'] > 0:
            pitch_ratio = ref_lattice['pitch_x_px'] / (search_lattice['pitch_x_px'] * nominal_scale + 1e-6)
            if 0.50 <= pitch_ratio <= 2.0:
                estimated_scale = nominal_scale * pitch_ratio
            else:
                estimated_scale = nominal_scale
        else:
            estimated_scale = nominal_scale
    else:
        rel_rotation = 0.0
        estimated_scale = nominal_scale

    estimated_scale = float(np.clip(estimated_scale, 5.0, 15.0))

    return {
        'rotation_deg': float(rel_rotation),
        'scale_factor': float(estimated_scale),
        'confidence': float(joint_confidence)
    }
