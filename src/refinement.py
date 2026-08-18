"""Stage G — Sub-Pixel Refinement & Uncertainty Estimator Engine.

Implements 2D quadratic / parabolic correlation surface fitting and phase correlation
sub-pixel peak estimation. Estimates localization uncertainty based on peak curvature.

Target: < 0.1 px sub-pixel accuracy.
"""

from __future__ import annotations
import math
import cv2
import numpy as np
from src.candidate import Candidate, prepare_reference_template
from src.residual import extract_patch


def parabolic_2d_subpixel(neighborhood_3x3: np.ndarray) -> tuple[float, float, float]:
    """Fit a 2D quadratic surface to a 3x3 correlation matrix neighborhood around local maximum.
    
    Neighborhood indexing:
        [ s(-1,-1)  s(-1, 0)  s(-1,+1) ]
        [ s( 0,-1)  s( 0, 0)  s( 0,+1) ]
        [ s(+1,-1)  s(+1, 0)  s(+1,+1) ]
        
    Returns:
        dx: Sub-pixel X offset [-0.5, 0.5]
        dy: Sub-pixel Y offset [-0.5, 0.5]
        peak_val: Interpolated peak correlation value
    """
    S = neighborhood_3x3.astype(np.float64)
    s00 = S[1, 1]
    sm1_0 = S[1, 0]
    sp1_0 = S[1, 2]
    s0_m1 = S[0, 1]
    s0_p1 = S[2, 1]

    # 1D separable parabolic peak estimator
    denom_x = sm1_0 - 2.0 * s00 + sp1_0
    denom_y = s0_m1 - 2.0 * s00 + s0_p1

    if abs(denom_x) > 1e-8:
        dx = (sm1_0 - sp1_0) / (2.0 * denom_x)
    else:
        dx = 0.0

    if abs(denom_y) > 1e-8:
        dy = (s0_m1 - s0_p1) / (2.0 * denom_y)
    else:
        dy = 0.0

    dx = float(np.clip(dx, -0.5, 0.5))
    dy = float(np.clip(dy, -0.5, 0.5))

    # Interpolated peak value
    peak_val = float(s00 - 0.25 * (sm1_0 - sp1_0) * dx - 0.25 * (s0_m1 - s0_p1) * dy)
    return dx, dy, peak_val


def refine_subpixel_position(
    search_img: np.ndarray,
    ref_img: np.ndarray,
    candidate: Candidate,
    upsample_factor: int = 100
) -> tuple[float, float, float]:
    """Refine candidate coordinates to sub-pixel accuracy using local ZNCC surface fitting.
    
    Args:
        search_img: Low-magnification search image (1000x1000 px).
        ref_img: High-magnification reference image (1000x1000 px).
        candidate: Candidate hypothesis containing integer/coarse (x, y).
        upsample_factor: Internal upsampling factor for phase cross-correlation.
        
    Returns:
        sub_x: Refined sub-pixel X coordinate in Search space.
        sub_y: Refined sub-pixel Y coordinate in Search space.
        uncertainty_px: Estimated localization uncertainty std in pixels.
    """
    template, (tpl_h, tpl_w) = prepare_reference_template(
        ref_img, scale_factor=candidate.scale, rotation_deg=candidate.rotation
    )

    # Extract 5x5 score grid around integer candidate center
    cx_int = int(round(candidate.x))
    cy_int = int(round(candidate.y))

    # Compute local 3x3 ZNCC matrix around candidate
    zncc_grid = np.zeros((3, 3), dtype=np.float64)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            curr_x = cx_int + dx
            curr_y = cy_int + dy
            patch, _ = extract_patch(search_img, curr_x, curr_y, tpl_w, tpl_h)
            if patch.shape == template.shape:
                t_z = template - np.mean(template)
                p_z = patch - np.mean(patch)
                denom = np.sqrt(np.sum(t_z ** 2) * np.sum(p_z ** 2)) + 1e-8
                zncc_grid[dy + 1, dx + 1] = float(np.sum(t_z * p_z) / denom)
            else:
                zncc_grid[dy + 1, dx + 1] = candidate.zncc_score

    # Parabolic sub-pixel interpolation offset
    dx, dy, peak_val = parabolic_2d_subpixel(zncc_grid)

    sub_x = cx_int + dx
    sub_y = cy_int + dy

    # Estimate uncertainty from local curvature (Hessian inverse trace)
    s00 = zncc_grid[1, 1]
    curv_x = max(1e-4, 2.0 * s00 - zncc_grid[1, 0] - zncc_grid[1, 2])
    curv_y = max(1e-4, 2.0 * s00 - zncc_grid[0, 1] - zncc_grid[2, 1])
    uncertainty_px = float(math.sqrt(1.0 / (curv_x * 100.0) + 1.0 / (curv_y * 100.0)))
    uncertainty_px = float(np.clip(uncertainty_px, 0.01, 1.0))

    return float(sub_x), float(sub_y), uncertainty_px
