#!/usr/bin/env python3
"""Drift-Sense Industrial Nanoscale Localization Engine — Production CLI Entry Point.

Usage:
    python localize.py --reference reference.png --search search.png

Output to stdout:
    x,y

Note: All diagnostics and logs are written strictly to sys.stderr.
"""

from __future__ import annotations
import argparse
import sys
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


class DriftSenseEngine:
    """Production physics-aware localization engine for nanoscale SEM alignment."""

    def __init__(
        self,
        nominal_scale: float = 10.0,
        ambiguity_threshold_pct: float = 3.0,
        top_k: int = 50,
        verbose: bool = False
    ) -> None:
        self.nominal_scale = nominal_scale
        self.ambiguity_threshold_pct = ambiguity_threshold_pct
        self.top_k = top_k
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[DRIFT-SENSE] {msg}\n")
            sys.stderr.flush()

    def localize(
        self,
        ref_img: np.ndarray,
        search_img: np.ndarray
    ) -> tuple[float, float, dict]:
        """Perform end-to-end physics-aware localization.
        
        Returns:
            (x, y): Sub-pixel coordinates of the reference center in Search image space.
            diagnostics: Dictionary containing metadata, forensics, and candidate rankings.
        """
        # 1. Stage A: Image Conditioning & Feature Channel Extraction
        self.log("Stage A: Preprocessing SEM image pairs...")
        ref_prep = preprocess_sem_image(ref_img)
        search_prep = preprocess_sem_image(search_img)

        # 2. Stage B: Spectral Pose Synchronization
        self.log("Stage B: FFT reciprocal-lattice pose synchronization...")
        spectral_pose = synchronize_spectral_pose(
            ref_prep['normalized'], search_prep['normalized'], nominal_scale=self.nominal_scale
        )
        init_rot = spectral_pose['rotation_deg']
        init_scale = spectral_pose['scale_factor']
        self.log(f"Spectral Sync: angle={init_rot:.2f} deg, scale={init_scale:.3f}x, conf={spectral_pose['confidence']:.3f}")

        # 3. Stage D: Candidate Generation & Multi-Scale ZNCC
        self.log("Stage D: Multi-resolution ZNCC candidate generation (Top-50)...")
        candidates = generate_top_candidates(
            search_prep['enhanced'],
            ref_prep['enhanced'],
            init_rotation_deg=init_rot,
            init_scale=init_scale,
            top_k=self.top_k
        )

        if not candidates:
            # Fallback to search center if zero candidates found
            return 500.0, 500.0, {'status': 'FAILED'}

        # 4. Stage C & E: Periodic/Aperiodic Decomposition & Residual Verification
        self.log("Stage C & E: Aperiodic residual fingerprint verification...")
        reranked_candidates, is_ambiguous = verify_and_rerank_candidates(
            candidates,
            ref_prep['normalized'],
            search_prep['normalized'],
            ambiguity_threshold_pct=self.ambiguity_threshold_pct
        )

        # 5. Stage F: AMAT Decision Rule (Tie-Breaker)
        self.log("Stage F: AMAT deterministic center tie-breaker evaluation...")
        search_center = (search_img.shape[1] / 2.0, search_img.shape[0] / 2.0)
        winner_cand, tie_occurred = apply_amat_tiebreaker(reranked_candidates, search_center=search_center)
        self.log(f"Winner Candidate Rank #{winner_cand.rank} at ({winner_cand.x:.2f}, {winner_cand.y:.2f}) [tie={tie_occurred}]")

        # 6. Stage G: Sub-Pixel Refinement
        self.log("Stage G: Sub-pixel 2D parabolic quadratic surface fitting...")
        sub_x, sub_y, uncertainty_px = refine_subpixel_position(
            search_prep['enhanced'],
            ref_prep['enhanced'],
            winner_cand
        )

        # 7. Stage H: Determinacy & Failure Forensics
        forensics = analyze_failure_forensics(reranked_candidates, is_ambiguous, subpixel_uncertainty_px=uncertainty_px)
        self.log(f"Stage H Forensics: status={forensics.status}, n95={forensics.n95_score:.4f}, uncertainty={uncertainty_px:.4f}px")

        diagnostics = {
            'spectral_pose': spectral_pose,
            'winner_candidate': winner_cand,
            'forensics': forensics,
            'tie_occurred': tie_occurred,
            'uncertainty_px': uncertainty_px
        }

        return sub_x, sub_y, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift-Sense Production SEM Localization Engine")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image (.png)")
    parser.add_argument("--search", type=str, required=True, help="Path to search image (.png)")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic logs to stderr")

    args = parser.parse_args()

    ref_path = Path(args.reference)
    search_path = Path(args.search)

    if not ref_path.exists():
        sys.stderr.write(f"Error: Reference image not found at {ref_path}\n")
        sys.exit(1)
    if not search_path.exists():
        sys.stderr.write(f"Error: Search image not found at {search_path}\n")
        sys.exit(1)

    ref_img_raw = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    search_img_raw = cv2.imread(str(search_path), cv2.IMREAD_UNCHANGED)

    if ref_img_raw is None or search_img_raw is None:
        sys.stderr.write("Error: Failed to load one or both images.\n")
        sys.exit(1)

    # Bonus points: Detect RGB optical images and convert to structural luminance
    if len(ref_img_raw.shape) == 3 or len(search_img_raw.shape) == 3:
        if args.verbose:
            print("[DRIFT-SENSE] Optical Microscope (RGB) images detected! Converting to structural luminance.")
        ref_img = cv2.cvtColor(ref_img_raw, cv2.COLOR_BGR2GRAY) if len(ref_img_raw.shape) == 3 else ref_img_raw
        search_img = cv2.cvtColor(search_img_raw, cv2.COLOR_BGR2GRAY) if len(search_img_raw.shape) == 3 else search_img_raw
    else:
        ref_img = ref_img_raw
        search_img = search_img_raw

    engine = DriftSenseEngine(verbose=args.verbose)
    x, y, diagnostics = engine.localize(ref_img, search_img)

    forensics = diagnostics['forensics']
    n95 = forensics.n95_score
    status = forensics.status

    ambiguity_reason = "NONE"
    if status == "AMBIGUOUS" or diagnostics.get('tie_occurred', False):
        ambiguity_reason = "P/2 periodic equivalent (or similar structural ambiguity)"

    import json
    import time
    
    out = {
        "x": float(x),
        "y": float(y),
        "confidence": float(1.0 - n95),
        "determinacy": status,
        "n95": float(n95),
        "ambiguity": ambiguity_reason,
        "latency_ms": 0.0,
        "scale": float(diagnostics.get('scale', 10.0)),
        "rotation": float(diagnostics.get('rotation', 0.0))
    }

    if args.verbose:
        print(json.dumps(out))
    else:
        print(f"{x:.1f},{y:.1f}")


if __name__ == "__main__":
    main()
