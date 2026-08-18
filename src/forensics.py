"""Stage H — Failure Forensics and Determinacy Metric Engine.

Implements the n95 determinacy metric and failure forensics report classification:
- Determinate: Unique candidate clearly supported by image evidence.
- Ambiguous: Highly periodic array where multiple candidates share high correlation.
- Information-limited: Low contrast or high SEM noise rendering site unidentifiable.

Mathematical Definition of n95 Metric:
- Inputs: List of top candidate match scores S = [S_1, S_2, ..., S_N] (sorted descending, N=50).
- Formula: n95 = (S_1 - P_95(S)) / S_1
  where P_95(S) is the 95th percentile score of candidate scores S.
- Interpretation:
  * n95 >= 0.12: High determinacy, strong single peak.
  * 0.03 <= n95 < 0.12: Moderate determinacy, periodic array candidate ambiguity.
  * n95 < 0.03: Low determinacy, flat score landscape / information-limited.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.candidate import Candidate


@dataclass
class ForensicsReport:
    """Structure storing failure forensics diagnostics for a single image pair."""
    status: str                         # 'DETERMINATE', 'AMBIGUOUS', or 'INFORMATION_LIMITED'
    n95_score: float                   # Calculated n95 determinacy metric [0.0, 1.0]
    top_score: float                   # Highest candidate score
    score_margin_p95: float            # Absolute difference (S_1 - P_95)
    ambiguous_candidate_count: int     # Count of candidates within 3% of top score
    residual_confidence: float         # Aperiodic residual verification confidence
    estimated_uncertainty_px: float    # Estimated sub-pixel localization uncertainty in pixels
    explanation: str                   # Natural language diagnostic explanation


def compute_n95_metric(candidate_scores: list[float]) -> tuple[float, float]:
    """Calculate the n95 determinacy metric from candidate match scores.
    
    Args:
        candidate_scores: List of float candidate scores sorted descending.
        
    Returns:
        n95_val: Normalized n95 determinacy metric.
        p95_val: 95th percentile candidate score.
    """
    if not candidate_scores:
        return 0.0, 0.0

    scores_arr = np.array(candidate_scores, dtype=np.float64)
    s1 = scores_arr[0]

    if len(scores_arr) < 5 or s1 <= 1e-6:
        return 0.0, float(s1)

    p95_val = float(np.percentile(scores_arr, 95.0))
    n95_val = float((s1 - p95_val) / (s1 + 1e-8))
    return float(np.clip(n95_val, 0.0, 1.0)), p95_val


def analyze_failure_forensics(
    candidates: list[Candidate],
    is_ambiguous: bool,
    subpixel_uncertainty_px: float = 0.05
) -> ForensicsReport:
    """Generate comprehensive failure forensics report for a localization attempt.
    
    Args:
        candidates: List of Candidate objects sorted by final rank.
        is_ambiguous: Flag from residual verifier indicating multi-candidate tie.
        subpixel_uncertainty_px: Estimated sub-pixel error bound in pixels.
        
    Returns:
        ForensicsReport object containing diagnostics and classification.
    """
    if not candidates:
        return ForensicsReport(
            status="INFORMATION_LIMITED",
            n95_score=0.0,
            top_score=0.0,
            score_margin_p95=0.0,
            ambiguous_candidate_count=0,
            residual_confidence=0.0,
            estimated_uncertainty_px=10.0,
            explanation="Zero candidate matches detected."
        )

    scores = [c.composite_score for c in candidates]
    n95_score, p95_val = compute_n95_metric(scores)
    top_score = candidates[0].composite_score

    # Count candidates within 3% of top score
    ambiguous_count = sum(
        1 for c in candidates if (top_score - c.composite_score) / (top_score + 1e-8) <= 0.03
    )

    residual_conf = candidates[0].local_residual_score

    if top_score < 0.35 or subpixel_uncertainty_px > 0.8:
        status = "INFORMATION_LIMITED"
        explanation = f"Low correlation signal (top score {top_score:.3f}) or high SEM noise."
    elif is_ambiguous or ambiguous_count > 1 or n95_score < 0.08:
        status = "AMBIGUOUS"
        explanation = f"Periodic array ambiguity: {ambiguous_count} false candidate traps within 3% of top score."
    else:
        status = "DETERMINATE"
        explanation = f"Unique landmark site determined with high confidence (n95={n95_score:.3f})."

    return ForensicsReport(
        status=status,
        n95_score=n95_score,
        top_score=top_score,
        score_margin_p95=float(top_score - p95_val),
        ambiguous_candidate_count=ambiguous_count,
        residual_confidence=residual_conf,
        estimated_uncertainty_px=subpixel_uncertainty_px,
        explanation=explanation
    )
