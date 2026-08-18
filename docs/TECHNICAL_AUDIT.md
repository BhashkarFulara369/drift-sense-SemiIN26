# Technical Audit: DRIFT-SENSE Localization Engine

## 1. Current Architecture
The current architecture is a sequential, heavily engineered pipeline (A through H) focusing on classical CV techniques without deep learning. It uses a combination of FFT-based spectral synchronization, multiscale ZNCC, residual feature verification, and a deterministic tie-breaker. It operates purely on image pairs (Reference + Search) and produces a deterministic center.

## 2. Current Algorithm
- **Stage A (Preprocessing):** CLAHE + Bilateral filtering + Scharr gradients + high-pass LER.
- **Stage B (Spectral Pose):** 2D FFT to find reciprocal lattice vectors to estimate relative rotation and scale.
- **Stage D (Candidate Generation):** Multi-scale, multi-rotation ZNCC template matching to find the top 50 local maxima (spatially NMS).
- **Stage C & E (Residual Verifier):** Computes "aperiodic residual fingerprint" (ZNCC on high-pass LER and gradient orientation) to rerank the top candidates that fall within a 3% ambiguity threshold.
- **Stage F (AMAT Tie-Breaker):** If ambiguity still exists, chooses the candidate closest to the search image center (500.0, 500.0).
- **Stage G (Sub-Pixel Refinement):** 2D quadratic parabolic surface fitting on the 3x3 ZNCC matrix around the chosen candidate.
- **Stage H (Forensics):** Evaluates `n95` determinacy metric.

## 3. Current Generator
`dataset_generator.py` is a massively complex procedural simulator generating "DRAM" and "FinFET" structures with layered noise (shot, speckle, Palasantzas LER, astigmatism blur, stage drift, elastic warping). It outputs 1000x1000 reference and search images with a 10x physical scale ratio. 

## 4. Current Evaluation Methodology
`evaluate.py` runs the pipeline over a dataset (defined by `metadata.json`). It converts the predicted sub-pixel (x, y) to error modulo the true periodic lattice pitch. This means it measures structural alignment error rather than absolute global center error (except it uses absolute error for baseline comparison). It heavily boasts about high pass rates at <0.1px on clean data.

## 5. Current Strengths
- Avoids the "black box" nature of CNNs; maintains explainability.
- Fast runtime (<50ms per pair).
- Strong awareness of periodic vs aperiodic signals (LER).
- Proper use of reciprocal lattice for rotation/scale estimation.

## 6. Current Weaknesses
- **Tie-breaker is a massive crutch (Leakage):** The AMAT tie-breaker mathematically forces the system to pick the center. If the generator typically places targets near the center, this is gross data leakage that artificially inflates performance on ambiguous arrays.
- **Lack of Multi-Transform Consensus:** It only uses local ZNCC maxima and reranks them. It doesn't cluster candidates that independently agree across different rotation/scale hypothesis transforms.
- **Single Aperiodic Channel:** It heavily relies on one LER fingerprint extraction instead of diverse context/residual features.
- **No Learned Reranker:** It uses a hardcoded weighted sum (`0.6 * max(0, res) + 0.4 * grad_cos_sim`) rather than learning which features actually disambiguate.

## 7. Mathematical Inconsistencies
- Subpixel quadratic fitting uses OpenCV TM_CCOEFF_NORMED output, but the parabola is fit locally without considering the theoretical shape of the ZNCC autocorrelation peak (which is not necessarily a perfect parabola).
- The spectral sync bounds scale within [9.4, 10.6] for a nominal 10x. But if the physical ratio is exactly 10x, variation is only due to elastic warp or generator mechanics. If the generator enforces 10x, restricting it might hide flaws.
- The evaluation modulo the periodic lattice means the algorithm can be "wrong" globally but "right" locally, which might artificially boost stats if the task is true global localization.

## 8. Potential Leakage
- **Target Center Leakage:** The Tie-Breaker explicitly rewards picking the center of the image (500, 500). If the dataset generator places the target near the center, this is a severe leakage.
- **Same-Noise Leakage:** If the LER or shot noise is generated on a shared canvas *before* being split into Reference and Search images, the algorithm might just be matching the exact same simulated noise array rather than true persistent physical identity. 
- **Fixed Generation Seed:** If the seed is static, it risks overfitting.

## 9. Current Failure Modes
- FinFET "Half-Pitch" ambiguity: The algorithm falls back to the center tie-breaker because it cannot distinguish half-pitch shifts without wider context.
- Elastic stage warping completely breaks the rigid 2D template matcher and ZNCC peak shape.

## 10. Real-SEM Failure Modes
- True variations in charging/astigmatism might create false "aperiodic" features that the residual verifier misinterprets.
- Macro structure context (which humans use to disambiguate) is currently ignored in favor of local LER signatures.

## 11. Hidden-Test Risks
- If the hidden test set does not have targets strictly clustered around (500, 500), the AMAT Tie-Breaker will catastrophically fail and send candidates to the center instead of the true location.
- True OOD parameters (e.g., higher rotation, extreme noise) will break the hardcoded 3% ambiguity threshold and hand-tuned residual weights.

## 12. What Should Be Preserved
- The hierarchical concept (Periodic -> Candidate -> Aperiodic -> Subpixel).
- Subpixel refinement (though it needs validation).
- Fast CPU-based inference.
- The `n95` determinacy metric is conceptually sound.

## 13. What Should Be Rewritten
- **Candidate Reranking:** Replace hardcoded weights with a robust Machine Learning reranker (e.g., Random Forest or Gradient Boosting) using multiple features.
- **Candidate Consensus:** Implement Cross-Transform Consensus so a candidate needs support from multiple geometric hypotheses.
- **Global Context:** Add a comparison of larger context windows, not just the local patch.
- **Tie-Breaker:** Disable the fixed-center tiebreaker during actual evaluation to measure true algorithmic performance, or replace it with a mathematically defensible context-based tiebreaker.
- **Generator Leakage:** Ensure Reference and Search noise are independently sampled from the same latent physical wafer, not identically generated.

## 14. What Should Be Tested Experimentally
- Are ORB/SIFT really that bad, or were they just tuned poorly?
- Does the Siamese network actually help on EXTREME ambiguity where classical LER fails?
- Can phase correlation outperform ZNCC for sub-pixel refinement?
