"""Stage C & E — Periodic/Aperiodic Decomposition and Residual Fingerprint Verifier.

Decomposes semiconductor images into periodic lattice components and aperiodic residuals
(Line-Edge Roughness, local defects, structural variations). Disambiguates candidate
traps using high-frequency residual fingerprint signatures.

References:
- Palasantzas (1993) — Nanoscale Surface & Line-Edge Roughness Modeling
"""

from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage
from src.candidate import Candidate, prepare_reference_template
from src.preprocessing import compute_gradients, extract_highpass_ler


def decompose_periodic_aperiodic(img: np.ndarray, blur_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Decompose image into low-frequency/periodic structural background and aperiodic residual.
    
    Args:
        img: Input float32 grayscale image.
        blur_sigma: Gaussian blur scale to isolate macro periodic envelope.
        
    Returns:
        periodic_comp: Smooth periodic component.
        aperiodic_res: High-frequency aperiodic residual map containing LER & structural micro-defects.
    """
    img_f = img.astype(np.float32)
    periodic_comp = ndimage.gaussian_filter(img_f, sigma=blur_sigma)
    aperiodic_res = img_f - periodic_comp
    return periodic_comp, aperiodic_res


def extract_patch(
    search_img: np.ndarray,
    center_x: float,
    center_y: float,
    patch_w: int,
    patch_h: int
) -> tuple[np.ndarray, bool]:
    """Extract a cropped image patch centered at (center_x, center_y).
    
    Returns:
        patch: Extracted float32 patch image.
        valid: True if patch is entirely within image bounds, False otherwise.
    """
    sh, sw = search_img.shape
    x0 = int(round(center_x - patch_w / 2.0))
    y0 = int(round(center_y - patch_h / 2.0))
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    if x0 < 0 or y0 < 0 or x1 > sw or y1 > sh:
        # Pad with border reflect
        pad_x0 = max(0, -x0)
        pad_y0 = max(0, -y0)
        pad_x1 = max(0, x1 - sw)
        pad_y1 = max(0, y1 - sh)

        crop_x0 = max(0, x0)
        crop_y0 = max(0, y0)
        crop_x1 = min(sw, x1)
        crop_y1 = min(sh, y1)

        cropped = search_img[crop_y0:crop_y1, crop_x0:crop_x1]
        padded = np.pad(cropped, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode='reflect')
        return padded.astype(np.float32), False

    return search_img[y0:y1, x0:x1].astype(np.float32), True


def compute_residual_fingerprint_score(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidate: Candidate
) -> float:
    """Compute multi-feature aperiodic residual fingerprint similarity score for a candidate.
    
    Evaluates:
    1. High-frequency LER residual correlation
    2. Gradient orientation alignment
    """
    ref_res = extract_highpass_ler(ref_img)
    search_res = extract_highpass_ler(search_img)

    # Extract corresponding search residual patch at coarse resolution
    search_h, search_w = int(round(ref_res.shape[0] / candidate.scale)), int(round(ref_res.shape[1] / candidate.scale))
    search_res_patch, _ = extract_patch(search_res, candidate.x, candidate.y, search_w, search_h)

    # Upsample search residual patch to match high-resolution reference
    search_res_patch_up = cv2.resize(search_res_patch, (ref_res.shape[1], ref_res.shape[0]), interpolation=cv2.INTER_CUBIC)

    if abs(candidate.rotation) > 0.01:
        center = (search_res_patch_up.shape[1] / 2.0, search_res_patch_up.shape[0] / 2.0)
        M = cv2.getRotationMatrix2D(center, -candidate.rotation, 1.0)
        search_res_patch_up = cv2.warpAffine(search_res_patch_up, M, (search_res_patch_up.shape[1], search_res_patch_up.shape[0]), flags=cv2.INTER_LINEAR)

    # 1. Zero-mean normalized correlation on HIGH-RES residual maps
    t_zero = ref_res - np.mean(ref_res)
    p_zero = search_res_patch_up - np.mean(search_res_patch_up)
    denom = np.sqrt(np.sum(t_zero ** 2) * np.sum(p_zero ** 2)) + 1e-8
    res_corr = float(np.sum(t_zero * p_zero) / denom)

    # 2. Gradient orientation cosine alignment on HIGH-RES maps
    _, ref_orient, _, _ = compute_gradients(ref_img)
    _, search_orient, _, _ = compute_gradients(search_img)
    
    # Extract orientation patch and upsample
    search_orient_patch, _ = extract_patch(search_orient, candidate.x, candidate.y, search_w, search_h)
    search_orient_patch_up = cv2.resize(search_orient_patch, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    if abs(candidate.rotation) > 0.01:
        search_orient_patch_up = cv2.warpAffine(search_orient_patch_up, M, (search_orient_patch_up.shape[1], search_orient_patch_up.shape[0]), flags=cv2.INTER_NEAREST)
        
    angle_diff = np.abs(ref_orient - search_orient_patch_up)
    grad_cos_sim = float(np.mean(np.cos(angle_diff)))

    # Composite residual fingerprint score
    residual_fingerprint_score = 0.6 * max(0.0, res_corr) + 0.4 * max(0.0, grad_cos_sim)
    return float(residual_fingerprint_score)


def verify_and_rerank_candidates(
    candidates: list[Candidate],
    ref_img: np.ndarray,
    search_img: np.ndarray,
    ambiguity_threshold_pct: float = 10.0,
    weight_zncc: float = 0.5,
    weight_res: float = 0.5
) -> tuple[list[Candidate], bool]:
    """Identify ambiguous candidates within score threshold and re-rank using residual fingerprints.
    
    Args:
        candidates: List of Top-50 candidate hypotheses sorted by ZNCC score.
        ref_img: Reference image (1000x1000 px).
        search_img: Search image (1000x1000 px).
        ambiguity_threshold_pct: Relative ZNCC score difference threshold (default 3.0%).
        weight_zncc: Weight of ZNCC score in composite ranking.
        weight_res: Weight of residual fingerprint score in composite ranking.
        
    Returns:
        reranked_candidates: Candidates sorted by composite score descending.
        is_ambiguous: True if ambiguity threshold was triggered across top candidates.
    """
    if not candidates:
        return [], False

    top_zncc = candidates[0].zncc_score
    ambiguous_indices = []

    for i, cand in enumerate(candidates):
        rel_diff_pct = 100.0 * (top_zncc - cand.zncc_score) / (top_zncc + 1e-8)
        if rel_diff_pct <= ambiguity_threshold_pct:
            ambiguous_indices.append(i)

    is_ambiguous = len(ambiguous_indices) > 1

    # Evaluate residual fingerprint score for ambiguous candidates (or top 5 if not ambiguous)
    eval_indices = ambiguous_indices if is_ambiguous else list(range(min(5, len(candidates))))

    for idx in eval_indices:
        cand = candidates[idx]
        res_score = compute_residual_fingerprint_score(ref_img, search_img, cand)
        cand.local_residual_score = res_score
        cand.composite_score = weight_zncc * cand.zncc_score + weight_res * res_score

    # Re-sort candidates by composite score descending
    reranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    for r, c in enumerate(reranked, start=1):
        c.rank = r

    return reranked, is_ambiguous
