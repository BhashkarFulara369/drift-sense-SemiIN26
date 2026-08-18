"""Stage F — AMAT Deterministic Decision Rule & Tie-Breaker Engine.

Implements Applied Materials (AMAT) deterministic tie-break rule:
If multiple candidate locations remain statistically tied after all available image
evidence (ZNCC + residual fingerprints) has been evaluated, select the candidate
closest to the center of the Search Image (500, 500) in Euclidean distance.
"""

from __future__ import annotations
import math
from src.candidate import Candidate


def compute_distance_to_center(x: float, y: float, center: tuple[float, float] = (500.0, 500.0)) -> float:
    """Compute Euclidean distance from candidate coordinates (x, y) to Search Image center (500, 500)."""
    cx, cy = center
    return float(math.sqrt((x - cx) ** 2 + (y - cy) ** 2))


def apply_amat_tiebreaker(
    candidates: list[Candidate],
    tie_tolerance: float = 0.03,
    search_center: tuple[float, float] = (500.0, 500.0),
    disable: bool = False
) -> tuple[Candidate, bool]:
    """Select winning candidate, breaking statistical ties using AMAT center-proximity rule.
    
    Args:
        candidates: List of Candidate objects sorted by composite score descending.
        tie_tolerance: Score difference threshold (default 0.03 = 3%) below which candidates are considered tied.
        search_center: Search Image center coordinates (default (500.0, 500.0)).
        disable: If True, do not use the center-distance heuristic, just return the top candidate.
        
    Returns:
        winner: The selected Candidate object.
        tie_occurred: True if a statistical tie was detected and resolved via AMAT rule.
    """
    if not candidates:
        raise ValueError("Cannot apply AMAT tie-breaker to empty candidate list.")

    top_score = candidates[0].composite_score
    tied_candidates = []

    for cand in candidates:
        rel_diff = (top_score - cand.composite_score) / (top_score + 1e-8)
        if rel_diff <= tie_tolerance:
            tied_candidates.append(cand)

    if len(tied_candidates) <= 1:
        return candidates[0], False

    if disable:
        # Return the top candidate by score without center-bias cheating
        return tied_candidates[0], True

    # Statistical tie detected: pick candidate minimizing Euclidean distance to center (500, 500)
    tied_candidates.sort(key=lambda c: compute_distance_to_center(c.x, c.y, search_center))
    return tied_candidates[0], True
