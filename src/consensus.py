from __future__ import annotations
from src.candidate import Candidate

def apply_cross_transform_consensus(
    candidates: list[Candidate],
    top_k: int = 50,
    support_weight: float = 0.05
) -> list[Candidate]:
    """
    Rerank and downselect candidates based on Cross-Transform Consensus.
    
    Candidates that were detected across multiple scale/rotation hypotheses 
    receive a score boost, making the pipeline robust against noise peaks 
    that only appear at a single specific transform.
    """
    if not candidates:
        return []

    for cand in candidates:
        # Boost composite score based on the number of supporting transforms
        # If support_count = 1 (only found in 1 sweep), boost is 0
        cand.composite_score = cand.zncc_score + (cand.transform_support_count - 1) * support_weight

    # Rerank by the new consensus-backed composite score
    candidates.sort(key=lambda c: c.composite_score, reverse=True)

    # Downselect to the final top_k
    final_candidates = candidates[:top_k]

    # Reassign ranks
    for i, cand in enumerate(final_candidates):
        cand.rank = i + 1

    return final_candidates
