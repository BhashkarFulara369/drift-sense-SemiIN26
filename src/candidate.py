"""Stage D — Candidate Generation and ZNCC Correlation Engine.

Implements multi-scale, multi-rotation Zero-Mean Normalized Cross-Correlation (ZNCC)
template matching and 2D Local Maxima Non-Maximum Suppression (NMS) to generate Top-50 candidates.
"""

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Candidate:
    """Structure storing candidate localization hypothesis."""
    x: float
    y: float
    zncc_score: float
    scale: float
    rotation: float
    rank: int
    local_residual_score: float = 0.0
    composite_score: float = 0.0
    transform_support_count: int = 1


def prepare_reference_template(
    ref_img: np.ndarray,
    scale_factor: float = 10.0,
    rotation_deg: float = 0.0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resample and rotate high-magnification reference image to low-magnification search scale."""
    h, w = ref_img.shape
    target_w = int(round(w / scale_factor))
    target_h = int(round(h / scale_factor))
    target_w = max(16, target_w)
    target_h = max(16, target_h)

    # Apply Gaussian blur to prevent aliasing before downsampling
    sigma = scale_factor / 3.0
    ref_blur = cv2.GaussianBlur(ref_img, (0, 0), sigmaX=sigma)

    resampled = cv2.resize(ref_blur, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if abs(rotation_deg) > 0.01:
        center = (target_w / 2.0, target_h / 2.0)
        M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        rotated = cv2.warpAffine(
            resampled, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        return rotated, (target_h, target_w)

    return resampled, (target_h, target_w)


def match_zncc_single(search_img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compute Zero-Mean Normalized Cross-Correlation (ZNCC) map using OpenCV TM_CCOEFF_NORMED."""
    s_f32 = search_img.astype(np.float32)
    t_f32 = template.astype(np.float32)
    result = cv2.matchTemplate(s_f32, t_f32, cv2.TM_CCOEFF_NORMED)
    return result


def generate_top_candidates(
    search_img: np.ndarray,
    ref_img: np.ndarray,
    init_rotation_deg: float = 0.0,
    init_scale: float = 10.0,
    top_k: int = 100,
    min_dist_px: int = 15,
    rotation_sweep: list[float] | None = None,
    scale_sweep: list[float] | None = None
) -> list[Candidate]:
    """Generate Top-K distinct 2D candidate positions across scale and rotation hypothesis sweep."""
    if rotation_sweep is None:
        rotation_sweep = [-4.0, -2.0, 0.0, 2.0, 4.0]
    if scale_sweep is None:
        scale_sweep = [0.96, 0.98, 1.00, 1.02, 1.04]

    all_raw_candidates = []

    for d_rot in rotation_sweep:
        rot_angle = init_rotation_deg + d_rot
        for s_mult in scale_sweep:
            scale_val = init_scale * s_mult
            template, (tpl_h, tpl_w) = prepare_reference_template(ref_img, scale_factor=scale_val, rotation_deg=rot_angle)

            if template.shape[0] > search_img.shape[0] or template.shape[1] > search_img.shape[1]:
                continue

            zncc_map = match_zncc_single(search_img, template)
            half_h = tpl_h / 2.0
            half_w = tpl_w / 2.0

            # Proper 2D Local Maxima Peak Extraction (prevents duplicate pixel clusters)
            local_max_mask = (ndimage.maximum_filter(zncc_map, size=min_dist_px) == zncc_map)
            peak_coords = np.argwhere(local_max_mask)

            for py, px in peak_coords:
                score = float(zncc_map[py, px])
                center_x = px + half_w
                center_y = py + half_h
                all_raw_candidates.append((score, center_x, center_y, scale_val, rot_angle))

    # Sort all distinct 2D peaks across sweeps by ZNCC score descending
    all_raw_candidates.sort(key=lambda c: c[0], reverse=True)

    # Perform Non-Maximum Suppression (NMS) and accumulate transform support consensus
    selected_candidates: list[Candidate] = []
    for score, cx, cy, s_val, r_val in all_raw_candidates:
        if len(selected_candidates) >= top_k:
            break
        too_close = False
        for existing in selected_candidates:
            dist = np.hypot(cx - existing.x, cy - existing.y)
            if dist < min_dist_px:
                too_close = True
                # Consensus: This existing candidate is supported by another transform hypothesis
                existing.transform_support_count += 1
                break
        
        if not too_close:
            rank = len(selected_candidates) + 1
            cand = Candidate(
                x=float(cx),
                y=float(cy),
                zncc_score=float(score),
                scale=float(s_val),
                rotation=float(r_val),
                rank=rank,
                composite_score=float(score)
            )
            selected_candidates.append(cand)

    return selected_candidates
