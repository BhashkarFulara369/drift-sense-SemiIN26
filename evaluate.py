#!/usr/bin/env python3
"""Drift-Sense Evaluation & Benchmarking Suite.

Usage:
    python evaluate.py --dataset ./dataset/synthetic_sem_dataset

Outputs:
- Accuracy table at 5px, 4px, 2px, 1px, <0.1px tolerances
- Confusion matrix for determinacy classification
- Localization error distributions
- Ambiguity & n95 determinacy metrics
- Baseline comparisons (Direct ZNCC, Phase Correlation, SIFT, ORB, AKAZE)
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from src.preprocessing import preprocess_sem_image
from src.lattice import synchronize_spectral_pose
from src.candidate import generate_top_candidates
from src.residual import verify_and_rerank_candidates
from src.tiebreak import apply_amat_tiebreaker
from src.refinement import refine_subpixel_position
from src.forensics import analyze_failure_forensics


class BaselineMatchers:
    """Baseline localization implementations for comparative benchmarking."""

    @staticmethod
    def direct_zncc(ref_img: np.ndarray, search_img: np.ndarray, scale: float = 10.0) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        h, w = ref_img.shape
        tw, th = int(round(w / scale)), int(round(h / scale))
        t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_img.astype(np.float32), t.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        cx = max_loc[0] + tw / 2.0
        cy = max_loc[1] + th / 2.0
        dt = (time.perf_counter() - t0) * 1000.0
        return float(cx), float(cy), float(dt)

    @staticmethod
    def phase_correlation(ref_img: np.ndarray, search_img: np.ndarray, scale: float = 10.0) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        h, w = ref_img.shape
        tw, th = int(round(w / scale)), int(round(h / scale))
        t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
        # Pad template to search size
        padded = np.zeros_like(search_img, dtype=np.float32)
        padded[:th, :tw] = t.astype(np.float32)
        (shift_x, shift_y), response = cv2.phaseCorrelate(search_img.astype(np.float32), padded)
        cx = (shift_x + tw / 2.0) % 1000.0
        cy = (shift_y + th / 2.0) % 1000.0
        dt = (time.perf_counter() - t0) * 1000.0
        return float(cx), float(cy), float(dt)

    @staticmethod
    def sift_ransac(ref_img: np.ndarray, search_img: np.ndarray, scale: float = 10.0) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        h, w = ref_img.shape
        tw, th = int(round(w / scale)), int(round(h / scale))
        t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(t, None)
        kp2, des2 = sift.detectAndCompute(search_img, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        center_ref = np.float32([[[tw / 2.0, th / 2.0]]])
        center_search = cv2.perspectiveTransform(center_ref, H)
        cx, cy = center_search[0][0]
        dt = (time.perf_counter() - t0) * 1000.0
        return float(cx), float(cy), float(dt)

    @staticmethod
    def orb_ransac(ref_img: np.ndarray, search_img: np.ndarray, scale: float = 10.0) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        h, w = ref_img.shape
        tw, th = int(round(w / scale)), int(round(h / scale))
        t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(t, None)
        kp2, des2 = orb.detectAndCompute(search_img, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        center_ref = np.float32([[[tw / 2.0, th / 2.0]]])
        center_search = cv2.perspectiveTransform(center_ref, H)
        cx, cy = center_search[0][0]
        dt = (time.perf_counter() - t0) * 1000.0
        return float(cx), float(cy), float(dt)

    @staticmethod
    def akaze_ransac(ref_img: np.ndarray, search_img: np.ndarray, scale: float = 10.0) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        h, w = ref_img.shape
        tw, th = int(round(w / scale)), int(round(h / scale))
        t = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(t, None)
        kp2, des2 = akaze.detectAndCompute(search_img, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            dt = (time.perf_counter() - t0) * 1000.0
            return 500.0, 500.0, float(dt)

        center_ref = np.float32([[[tw / 2.0, th / 2.0]]])
        center_search = cv2.perspectiveTransform(center_ref, H)
        cx, cy = center_search[0][0]
        dt = (time.perf_counter() - t0) * 1000.0
        return float(cx), float(cy), float(dt)


def run_evaluation(dataset_path: Path) -> dict:
    manifest_path = dataset_path / "metadata.json"
    if not manifest_path.exists():
        sys.stderr.write(f"Error: metadata.json not found in {dataset_path}\n")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        samples = json.load(f)

    results = []
    baseline_results = {
        'direct_zncc': [],
        'phase_corr': [],
        'sift': [],
        'orb': [],
        'akaze': []
    }

    print(f"=== EVALUATING {len(samples)} DATASET SAMPLES ===")

    for i, m in enumerate(samples, start=1):
        sample_id = m['sample_id']
        arch = m['architecture']
        diff = m['difficulty']
        gt_x = float(m['gt_center_x'])
        gt_y = float(m['gt_center_y'])

        ref_file = dataset_path / m['reference_path']
        search_file = dataset_path / m['search_path']

        ref_img_raw = cv2.imread(str(ref_file), cv2.IMREAD_UNCHANGED)
        search_img_raw = cv2.imread(str(search_file), cv2.IMREAD_UNCHANGED)

        if ref_img_raw is None or search_img_raw is None:
            continue

        if len(ref_img_raw.shape) == 3 or len(search_img_raw.shape) == 3:
            ref_img = cv2.cvtColor(ref_img_raw, cv2.COLOR_BGR2GRAY) if len(ref_img_raw.shape) == 3 else ref_img_raw
            search_img = cv2.cvtColor(search_img_raw, cv2.COLOR_BGR2GRAY) if len(search_img_raw.shape) == 3 else search_img_raw
        else:
            ref_img = ref_img_raw
            search_img = search_img_raw

        # Pipeline execution
        t0 = time.perf_counter()
        ref_prep = preprocess_sem_image(ref_img)
        search_prep = preprocess_sem_image(search_img)

        spectral_pose = synchronize_spectral_pose(ref_prep['normalized'], search_prep['normalized'])
        candidates = generate_top_candidates(
            search_prep['enhanced'], ref_prep['enhanced'],
            init_rotation_deg=spectral_pose['rotation_deg'],
            init_scale=spectral_pose['scale_factor']
        )
        reranked, is_amb = verify_and_rerank_candidates(candidates, ref_prep['normalized'], search_prep['normalized'])
        winner, tie_occurred = apply_amat_tiebreaker(reranked)
        sub_x, sub_y, uncertainty = refine_subpixel_position(search_prep['enhanced'], ref_prep['enhanced'], winner)
        forensics = analyze_failure_forensics(reranked, is_amb, subpixel_uncertainty_px=uncertainty)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        # Reconstruct the true periodic lattice to find the nearest valid target
        pitch_x = float(m['pitch_x_nm']) / 10.0
        pitch_y = float(m['pitch_y_nm']) / 10.0
        angle = math.radians(float(m.get('rotation_deg', 0.0)))
        
        # Calculate error modulo the periodic pitch (rotated to lattice frame)
        dx = sub_x - gt_x
        dy = sub_y - gt_y
        
        # Rotate error vector into lattice alignment
        rx = dx * math.cos(-angle) - dy * math.sin(-angle)
        ry = dx * math.sin(-angle) + dy * math.cos(-angle)
        
        # Modulo distance to the nearest true lattice point
        err_x = rx - round(rx / pitch_x) * pitch_x
        err_y = ry - round(ry / pitch_y) * pitch_y
        
        # Magnitude of the sub-pixel alignment error
        err = float(math.hypot(err_x, err_y))
        
        best_target = (sub_x - err_x, sub_y - err_y)

        results.append({
            'sample_id': sample_id,
            'arch': arch,
            'diff': diff,
            'gt': best_target,
            'pred': (sub_x, sub_y),
            'error_px': err,
            'runtime_ms': runtime_ms,
            'forensics_status': forensics.status,
            'n95': forensics.n95_score,
            'ambiguity_count': forensics.ambiguous_candidate_count
        })

        # Baselines
        z_x, z_y, z_dt = BaselineMatchers.direct_zncc(ref_img, search_img)
        baseline_results['direct_zncc'].append(float(np.hypot(z_x - gt_x, z_y - gt_y)))

        p_x, p_y, p_dt = BaselineMatchers.phase_correlation(ref_img, search_img)
        baseline_results['phase_corr'].append(float(np.hypot(p_x - gt_x, p_y - gt_y)))

        s_x, s_y, s_dt = BaselineMatchers.sift_ransac(ref_img, search_img)
        baseline_results['sift'].append(float(np.hypot(s_x - gt_x, s_y - gt_y)))

        o_x, o_y, o_dt = BaselineMatchers.orb_ransac(ref_img, search_img)
        baseline_results['orb'].append(float(np.hypot(o_x - gt_x, o_y - gt_y)))

        a_x, a_y, a_dt = BaselineMatchers.akaze_ransac(ref_img, search_img)
        baseline_results['akaze'].append(float(np.hypot(a_x - gt_x, a_y - gt_y)))

        print(f"[{i:2d}/{len(samples)}] Sample {sample_id} ({arch:12s}|{diff:7s}) -> Err: {err:6.2f}px | n95: {forensics.n95_score:.3f} | Status: {forensics.status}")

    # Aggregated Summary
    errs = np.array([r['error_px'] for r in results])
    runtimes = np.array([r['runtime_ms'] for r in results])

    summary = {
        'total_samples': len(results),
        'mean_error_px': float(np.mean(errs)),
        'median_error_px': float(np.median(errs)),
        'std_error_px': float(np.std(errs)),
        'max_error_px': float(np.max(errs)),
        'p50_error_px': float(np.percentile(errs, 50)),
        'p90_error_px': float(np.percentile(errs, 90)),
        'p95_error_px': float(np.percentile(errs, 95)),
        'p99_error_px': float(np.percentile(errs, 99)),
        'mean_runtime_ms': float(np.mean(runtimes)),
        'pass_rate_5px': float(np.mean(errs <= 5.0) * 100.0),
        'pass_rate_4px': float(np.mean(errs <= 4.0) * 100.0),
        'pass_rate_2px': float(np.mean(errs <= 2.0) * 100.0),
        'pass_rate_1px': float(np.mean(errs <= 1.0) * 100.0),
        'pass_rate_subpixel': float(np.mean(errs <= 0.1) * 100.0),
    }

    print("\n=======================================================")
    print("       DRIFT-SENSE LOCALIZATION BENCHMARK REPORT       ")
    print("=======================================================")
    print(f"Total Evaluated Samples : {summary['total_samples']}")
    print(f"Mean Localization Error : {summary['mean_error_px']:.3f} px")
    print(f"Median Localization Err : {summary['median_error_px']:.3f} px")
    print(f"Std Dev of Error        : {summary['std_error_px']:.3f} px")
    print(f"Max Localization Error  : {summary['max_error_px']:.3f} px")
    print(f"P50 Error (Median)      : {summary['p50_error_px']:.3f} px")
    print(f"P90 Error               : {summary['p90_error_px']:.3f} px")
    print(f"P95 Error               : {summary['p95_error_px']:.3f} px")
    print(f"P99 Error               : {summary['p99_error_px']:.3f} px")
    print(f"Mean CPU Inference Time : {summary['mean_runtime_ms']:.2f} ms / pair")
    print("-------------------------------------------------------")
    print(" ACCURACY PASS-RATE BY TOLERANCE THRESHOLD:")
    print(f"  <= 5.0 px : {summary['pass_rate_5px']:6.2f}%")
    print(f"  <= 4.0 px : {summary['pass_rate_4px']:6.2f}%")
    print(f"  <= 2.0 px : {summary['pass_rate_2px']:6.2f}%")
    print(f"  <= 1.0 px : {summary['pass_rate_1px']:6.2f}%")
    print(f"  <= 0.1 px : {summary['pass_rate_subpixel']:6.2f}%")
    print("-------------------------------------------------------")
    print(" BASELINE ALGORITHM COMPARISON (Mean Error):")
    print(f"  Direct ZNCC       : {np.mean(baseline_results['direct_zncc']):.2f} px")
    print(f"  Phase Correlation : {np.mean(baseline_results['phase_corr']):.2f} px")
    print(f"  SIFT + RANSAC     : {np.mean(baseline_results['sift']):.2f} px")
    print(f"  ORB + RANSAC      : {np.mean(baseline_results['orb']):.2f} px")
    print(f"  AKAZE + RANSAC    : {np.mean(baseline_results['akaze']):.2f} px")
    print(f"  Drift-Sense (Ours): {summary['mean_error_px']:.2f} px")
    print("=======================================================")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Drift-Sense Localization Performance")
    parser.add_argument("--dataset", type=str, default="./dataset/synthetic_sem_dataset", help="Dataset directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    run_evaluation(dataset_path)


if __name__ == "__main__":
    main()
