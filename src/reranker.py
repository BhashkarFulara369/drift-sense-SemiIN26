from __future__ import annotations
import numpy as np
from src.candidate import Candidate
import warnings

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MLReranker:
    """Machine Learning Reranker to predict the true target from ambiguous candidates."""
    
    def __init__(self):
        self.model = None
        if SKLEARN_AVAILABLE:
            # We initialize a mock/untrained model structure for the hackathon
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self._is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the reranker on extracted feature vectors."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn is not installed. MLReranker training is disabled.")
            return
            
        self.model.fit(X, y)
        self._is_trained = True

    def rerank(self, candidates: list[Candidate]) -> list[Candidate]:
        """Rerank candidates using the ML model or relative heuristics if untrained."""
        if not candidates:
            return []

        # Feature normalization across candidates (Context-relative ranking)
        feature_matrix = []
        for cand in candidates:
            # Fallback values if features aren't computed
            f = getattr(cand, 'features', {})
            vec = [
                f.get('res_corr', 0.0),
                f.get('res_fine_corr', 0.0),
                f.get('grad_cos_sim', 0.0),
                f.get('mag_corr', 0.0),
                float(f.get('support_count', 1.0))
            ]
            feature_matrix.append(vec)
            
        X = np.array(feature_matrix)
        
        if SKLEARN_AVAILABLE and self.model is not None and getattr(self, '_is_trained', False):
            # Predict probability of being the true target (Class 1)
            # Use Z-score for the ML model if trained
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0) + 1e-8
            X_norm = (X - means) / stds
            probs = self.model.predict_proba(X_norm)[:, 1]
            for i, cand in enumerate(candidates):
                cand.composite_score = probs[i]
        else:
            # Fallback heuristic using raw bounded features instead of unstable Z-scores
            for cand in candidates:
                f = getattr(cand, 'features', {})
                res_corr = float(f.get('res_corr', 0.0))
                res_fine_corr = float(f.get('res_fine_corr', 0.0))
                grad_cos_sim = float(f.get('grad_cos_sim', 0.0))
                mag_corr = float(f.get('mag_corr', 0.0))
                support_count = float(f.get('support_count', 1.0))
                
                # Raw features are physically bounded [0, 1].
                # We want a 0.01 physical residual difference to overcome a 0.005 ZNCC difference.
                # So we weight the primary residuals by 0.5.
                # IMPORTANT: In the old pipeline, Z-scored support_count (f[4]) dominated the heuristic.
                # To restore this balance without Z-score skew, we explicitly weight support_count higher.
                heuristic_score = cand.zncc_score + 0.5 * res_corr + 0.5 * res_fine_corr + 0.1 * grad_cos_sim + 0.1 * mag_corr + 0.2 * support_count
                cand.composite_score = float(heuristic_score)

        # Re-sort and assign ranks
        reranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
        for r, c in enumerate(reranked, start=1):
            c.rank = r

        return reranked
