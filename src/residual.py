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

    # 1. Zero-mean normalized correlation on HIGH-RES residual maps (sigma=4.0)
    t_zero = ref_res - np.mean(ref_res)
    p_zero = search_res_patch_up - np.mean(search_res_patch_up)
    denom = np.sqrt(np.sum(t_zero ** 2) * np.sum(p_zero ** 2)) + 1e-8
    res_corr = float(np.sum(t_zero * p_zero) / denom)

    # 2. Gradient orientation and magnitude on HIGH-RES maps
    ref_mag, ref_orient, _, _ = compute_gradients(ref_img)
    search_mag, search_orient, _, _ = compute_gradients(search_img)
    
    # Extract patches and upsample
    search_orient_patch, _ = extract_patch(search_orient, candidate.x, candidate.y, search_w, search_h)
    search_orient_patch_up = cv2.resize(search_orient_patch, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    search_mag_patch, _ = extract_patch(search_mag, candidate.x, candidate.y, search_w, search_h)
    search_mag_patch_up = cv2.resize(search_mag_patch, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    if abs(candidate.rotation) > 0.01:
        search_orient_patch_up = cv2.warpAffine(search_orient_patch_up, M, (search_orient_patch_up.shape[1], search_orient_patch_up.shape[0]), flags=cv2.INTER_NEAREST)
        search_mag_patch_up = cv2.warpAffine(search_mag_patch_up, M, (search_mag_patch_up.shape[1], search_mag_patch_up.shape[0]), flags=cv2.INTER_LINEAR)
        
    angle_diff = np.abs(ref_orient - search_orient_patch_up)
    grad_cos_sim = float(np.mean(np.cos(angle_diff)))
    
    m_zero = ref_mag - np.mean(ref_mag)
    sm_zero = search_mag_patch_up - np.mean(search_mag_patch_up)
    mag_denom = np.sqrt(np.sum(m_zero ** 2) * np.sum(sm_zero ** 2)) + 1e-8
    mag_corr = float(np.sum(m_zero * sm_zero) / mag_denom)
    
    # 3. Fine-scale highpass (sigma=1.5)
    ref_res_fine = extract_highpass_ler(ref_img, sigma_lowpass=1.5)
    search_res_fine = extract_highpass_ler(search_img, sigma_lowpass=1.5)
    search_resf_patch, _ = extract_patch(search_res_fine, candidate.x, candidate.y, search_w, search_h)
    search_resf_patch_up = cv2.resize(search_resf_patch, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    if abs(candidate.rotation) > 0.01:
        search_resf_patch_up = cv2.warpAffine(search_resf_patch_up, M, (search_resf_patch_up.shape[1], search_resf_patch_up.shape[0]), flags=cv2.INTER_LINEAR)
        
    tf_zero = ref_res_fine - np.mean(ref_res_fine)
    pf_zero = search_resf_patch_up - np.mean(search_resf_patch_up)
    f_denom = np.sqrt(np.sum(tf_zero ** 2) * np.sum(pf_zero ** 2)) + 1e-8
    res_fine_corr = float(np.sum(tf_zero * pf_zero) / f_denom)

    # Composite multi-channel residual fingerprint score
    residual_fingerprint_score = 0.3 * max(0.0, res_corr) + 0.3 * max(0.0, res_fine_corr) + 0.2 * max(0.0, grad_cos_sim) + 0.2 * max(0.0, mag_corr)
    
    # Store individual features on the candidate for future ML reranker
    candidate.features = {
        'res_corr': res_corr,
        'res_fine_corr': res_fine_corr,
        'grad_cos_sim': grad_cos_sim,
        'mag_corr': mag_corr,
        'support_count': candidate.transform_support_count
    }
    
    return float(residual_fingerprint_score)


def gather_parallel_evidence(
    candidates: list[Candidate],
    ref_img: np.ndarray,
    search_img: np.ndarray
) -> list[Candidate]:
    """Parallel Evidence Gathering: Computes physical residuals and context for candidates.
    
    Args:
        candidates: List of Candidate hypotheses.
        ref_img: Reference image (1000x1000 px).
        search_img: Search image (1000x1000 px).
        
    Returns:
        candidates: List of candidates with physical evidence features attached.
    """
    if not candidates:
        return []

    for cand in candidates:
        res_score = compute_residual_fingerprint_score(ref_img, search_img, cand)
        cand.local_residual_score = res_score

    return candidates
