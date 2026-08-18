#!/usr/bin/env python3
"""Drift-Sense Ablation Study.
Runs the dataset through progressively complex configurations of the pipeline to prove the value of each component.
"""

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

def run_ablation(dataset_path: Path):
    manifest_path = dataset_path / "metadata.json"
    if not manifest_path.exists():
        print(f"Error: metadata.json not found in {dataset_path}")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        samples = json.load(f)

    # Store errors for each configuration
    errors = {
        'A_ZNCC': [],
        'B_FFT_ZNCC': [],
        'C_FFT_ZNCC_RES': [],
        'D_FFT_ZNCC_RES_TIE': [],
        'E_FULL': []
    }

    print(f"=== RUNNING ABLATION STUDY ON {len(samples)} SAMPLES ===")

    for i, m in enumerate(samples, start=1):
        gt_x = float(m['gt_center_x'])
        gt_y = float(m['gt_center_y'])
        pitch_x = float(m['pitch_x_nm']) / 10.0
        pitch_y = float(m['pitch_y_nm']) / 10.0
        angle = math.radians(float(m.get('rotation_deg', 0.0)))

        ref_file = dataset_path / m['reference_path']
        search_file = dataset_path / m['search_path']

        ref_img = cv2.imread(str(ref_file), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(str(search_file), cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            continue

        ref_prep = preprocess_sem_image(ref_img)
        search_prep = preprocess_sem_image(search_img)

        # Helper to calculate absolute global error
        def calc_error(px, py):
            dx = px - gt_x
            dy = py - gt_y
            return float(math.hypot(dx, dy))

        # Config A: ZNCC Only (No FFT, scale=10.0, rot=0.0)
        cands_A = generate_top_candidates(search_prep['enhanced'], ref_prep['enhanced'], init_rotation_deg=0.0, init_scale=10.0, top_k=1)
        if cands_A:
            errors['A_ZNCC'].append(calc_error(cands_A[0].x, cands_A[0].y))
        
        # Config B: FFT + ZNCC
        spectral_pose = synchronize_spectral_pose(ref_prep['normalized'], search_prep['normalized'])
        cands_B = generate_top_candidates(
            search_prep['enhanced'], ref_prep['enhanced'],
            init_rotation_deg=spectral_pose['rotation_deg'],
            init_scale=spectral_pose['scale_factor'],
            top_k=100
        )
        from src.consensus import apply_cross_transform_consensus
        cands_B = apply_cross_transform_consensus(cands_B, top_k=50)
        if cands_B:
            # Top candidate by raw ZNCC score
            errors['B_FFT_ZNCC'].append(calc_error(cands_B[0].x, cands_B[0].y))

        # Config C: FFT + ZNCC + Residual
        if cands_B:
            reranked, is_amb = verify_and_rerank_candidates(cands_B, ref_prep['normalized'], search_prep['normalized'])
            # Top candidate by residual score, no tiebreaker
            errors['C_FFT_ZNCC_RES'].append(calc_error(reranked[0].x, reranked[0].y))

            # Config D: FFT + ZNCC + Residual + Tiebreaker
            winner, tie_occurred = apply_amat_tiebreaker(reranked)
            errors['D_FFT_ZNCC_RES_TIE'].append(calc_error(winner.x, winner.y))

            # Config E: Full (Subpixel)
            sub_x, sub_y, uncertainty = refine_subpixel_position(search_prep['enhanced'], ref_prep['enhanced'], winner)
            errors['E_FULL'].append(calc_error(sub_x, sub_y))

        print(f"[{i}/{len(samples)}] Evaluated.")

    print("\n==========================================================================================")
    print(f"{'System':<20} | {'Mean':<6} | {'Median':<6} | {'P95':<6} | {'<5 px':<6} | {'<1 px':<6} | {'<0.1 px':<6}")
    print("-" * 90)

    for name, key in [
        ("ZNCC", 'A_ZNCC'),
        ("FFT+ZNCC", 'B_FFT_ZNCC'),
        ("+Residual", 'C_FFT_ZNCC_RES'),
        ("+Gating/Tiebreaker", 'D_FFT_ZNCC_RES_TIE'),
        ("Full (Subpixel)", 'E_FULL')
    ]:
        errs = np.array(errors[key])
        if len(errs) == 0:
            continue
        
        mean = np.mean(errs)
        median = np.median(errs)
        p95 = np.percentile(errs, 95)
        p5 = np.mean(errs <= 5.0) * 100
        p1 = np.mean(errs <= 1.0) * 100
        p01 = np.mean(errs <= 0.1) * 100

        print(f"{name:<20} | {mean:<6.1f} | {median:<6.1f} | {p95:<6.1f} | {p5:>5.1f}% | {p1:>5.1f}% | {p01:>5.1f}%")

if __name__ == "__main__":
    run_ablation(Path("./dataset/synthetic_sem_dataset"))
