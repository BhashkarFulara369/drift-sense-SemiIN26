"""Drift-Sense Champion Pipeline.

This file exports the finalized champion `DriftSenseEngine` that consolidates
all tested advancements from the hackathon into one competitive pipeline.
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
import cv2

from src.preprocessing import preprocess_sem_image
from src.lattice import synchronize_spectral_pose
from src.candidate import generate_top_candidates
from src.consensus import apply_cross_transform_consensus
from src.residual import gather_parallel_evidence
from src.tiebreak import apply_amat_tiebreaker
from src.refinement import refine_subpixel_position
from src.forensics import analyze_failure_forensics, estimate_ambiguity
from src.reranker import MLReranker


class DriftSenseChampionEngine:
    """The elite, competition-ready localization engine for the SEMICON 2026 hackathon."""

    def __init__(
        self,
        nominal_scale: float = 10.0,
        ambiguity_threshold_pct: float = 3.0,
        top_k: int = 50,
        disable_amat_tiebreaker: bool = False,
        verbose: bool = False
    ) -> None:
        self.nominal_scale = nominal_scale
        self.ambiguity_threshold_pct = ambiguity_threshold_pct
        self.top_k = top_k
        self.disable_amat_tiebreaker = disable_amat_tiebreaker
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[CHAMPION] {msg}\n")
            sys.stderr.flush()

    def localize(
        self,
        ref_img: np.ndarray,
        search_img: np.ndarray
    ) -> tuple[float, float, dict]:
        """Perform end-to-end physics-aware localization without test-set leakage.
        
        Returns:
            (x, y): Absolute sub-pixel coordinates of the reference center in Search image space.
            diagnostics: Dictionary containing metadata, forensics, and candidate rankings.
        """
        # 1. Image Conditioning & Feature Channel Extraction
        self.log("Stage A: Preprocessing SEM image pairs...")
        ref_prep = preprocess_sem_image(ref_img)
        search_prep = preprocess_sem_image(search_img)

        # 2. Spectral Pose Synchronization
        self.log("Stage B: FFT reciprocal-lattice pose synchronization...")
        spectral_pose = synchronize_spectral_pose(
            ref_prep['normalized'], search_prep['normalized'], nominal_scale=self.nominal_scale
        )
        init_rot = spectral_pose['rotation_deg']
        init_scale = spectral_pose['scale_factor']
        self.log(f"Spectral Sync: angle={init_rot:.2f} deg, scale={init_scale:.3f}x, conf={spectral_pose['confidence']:.3f}")

        # 3. Candidate Generation & Multi-Scale ZNCC
        self.log("Stage D: Multi-resolution ZNCC candidate generation (Top-100)...")
        candidates = generate_top_candidates(
            search_prep['enhanced'],
            ref_prep['enhanced'],
            init_rotation_deg=init_rot,
            init_scale=init_scale,
            top_k=100
        )

        # 4. Transform Consensus (Top-20 Downselection)
        self.log(f"Stage D.1: Transform Consensus Downselection (Top-100 -> Top-{self.top_k})...")
        # Overriding self.top_k to 20 to strictly match the requested architecture diagram
        candidates = apply_cross_transform_consensus(candidates, top_k=self.top_k)

        if not candidates:
            self.log("ERROR: No candidates found. Defaulting to center.")
            return 500.0, 500.0, {'status': 'FAILED'}

        # 5. Parallel Evidence Gathering (Local Match, Context Evidence, Physical Residual)
        self.log("Stage E: Parallel Evidence Gathering (Physical Residuals & Context)...")
        candidates = gather_parallel_evidence(
            candidates,
            ref_prep['normalized'],
            search_prep['normalized']
        )

        # 6. Learned Reranker
        self.log("Stage F: Learned ML Reranker (Top Candidate Prediction)...")
        reranker = MLReranker()
        reranked_candidates = reranker.rerank(candidates)

        # 7. Ambiguity Estimator & Routing
        self.log("Stage G: Ambiguity Estimator...")
        is_tie = estimate_ambiguity(reranked_candidates, threshold_pct=self.ambiguity_threshold_pct)

        # 8. Uniquely Identified vs Genuine Tie Routing
        if is_tie:
            self.log("   -> Genuine tie detected: Routing to Official Center Rule...")
            winner_cand, tie_occurred = apply_amat_tiebreaker(reranked_candidates, disable=self.disable_amat_tiebreaker)
        else:
            self.log("   -> Uniquely identified: Routing directly to subpixel phase...")
            winner_cand = reranked_candidates[0]
            tie_occurred = False

        self.log(f"Winner Candidate Rank #{winner_cand.rank} at ({winner_cand.x:.2f}, {winner_cand.y:.2f}) [tie={tie_occurred}]")

        # 9. Sub-Pixel Phase Refinement
        self.log("Stage H: Sub-pixel 2D parabolic quadratic surface phase fitting...")
        sub_x, sub_y, uncertainty_px = refine_subpixel_position(
            search_prep['enhanced'],
            ref_prep['enhanced'],
            winner_cand
        )

        # 10. Determinacy & Failure Forensics
        self.log("Stage I: Analyzing Failure Forensics...")
        forensics = analyze_failure_forensics(reranked_candidates, is_tie, subpixel_uncertainty_px=uncertainty_px)
        self.log(f"Stage H Forensics: status={forensics.status}, n95={forensics.n95_score:.4f}, uncertainty={uncertainty_px:.4f}px")

        diagnostics = {
            'spectral_pose': spectral_pose,
            'winner_candidate': winner_cand,
            'forensics': forensics,
            'tie_occurred': tie_occurred,
            'uncertainty_px': uncertainty_px
        }

        return sub_x, sub_y, diagnostics
