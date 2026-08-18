#!/usr/bin/env python3
"""
================================================================================
DRIFT-SENSE: Interactive Streamlit MVP Web Dashboard
Applied Materials | Semicon India Hackathon 2026
================================================================================
Applied Materials branded UI with real-time SEM localization visualization.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import sys
import os

# Add parent directory to path to import predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import DriftSenseSolver

# ============================================================
# PAGE CONFIG & CORPORATE STYLING
# ============================================================
st.set_page_config(
    page_title="Drift-Sense | Applied Materials",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

AMAT_BLUE   = "#0072CE"
AMAT_DARK   = "#0F1C36"
AMAT_LIGHT  = "#E8F4FD"
AMAT_GREEN  = "#00C896"
AMAT_RED    = "#FF4B4B"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* ── Header Banner ── */
  .amat-header {{
    background: linear-gradient(135deg, {AMAT_DARK} 0%, #1a3a6b 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border-left: 6px solid {AMAT_BLUE};
    box-shadow: 0 8px 32px rgba(0,114,206,0.25);
  }}
  .amat-header h1 {{ color: white; font-size: 2.0rem; font-weight: 800; margin: 0; }}
  .amat-header p  {{ color: #99c8f0; font-size: 0.95rem; margin: 6px 0 0; }}

  /* ── Metric Cards ── */
  .metric-card {{
    background: linear-gradient(145deg, #1e2d4a, #162340);
    border: 1px solid rgba(0,114,206,0.35);
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
  }}
  .metric-card:hover {{ transform: translateY(-3px); }}
  .metric-label {{ color: #99c8f0; font-size: 0.78rem; font-weight: 600;
                   letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px; }}
  .metric-value {{ color: white; font-size: 1.9rem; font-weight: 800; }}
  .metric-unit  {{ color: #5a9fd4; font-size: 0.75rem; font-weight: 500; margin-top: 4px; }}

  /* ── Halt badges ── */
  .halt-ok   {{ background:{AMAT_GREEN}22; color:{AMAT_GREEN}; border:1px solid {AMAT_GREEN}55;
                padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }}
  .halt-warn {{ background:{AMAT_RED}22;   color:{AMAT_RED};   border:1px solid {AMAT_RED}55;
                padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }}

  /* ── Section divider ── */
  .section-title {{
    color: {AMAT_BLUE}; font-size: 1.05rem; font-weight: 700;
    border-bottom: 2px solid {AMAT_BLUE}33; padding-bottom: 6px; margin: 18px 0 14px;
  }}

  /* ── Upload area ── */
  [data-testid="stFileUploader"] > div {{
    border: 2px dashed {AMAT_BLUE}66 !important;
    border-radius: 12px !important;
    background: {AMAT_LIGHT}11 !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{ background: {AMAT_DARK}; }}
  [data-testid="stSidebar"] * {{ color: #cfe3f5 !important; }}

  /* Hide Streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_image_bytes(uploaded) -> np.ndarray:
    """Load UploadedFile to numpy array."""
    pil_img = Image.open(uploaded)
    return np.array(pil_img)


def draw_overlay(search_np: np.ndarray, pred_x: float, pred_y: float,
                  sigma_x: float, sigma_y: float, half_box: float = 50.0) -> Image.Image:
    """
    Draws on the search image:
      - Green bounding box (predicted region)
      - Green crosshair lines
      - Red 2σ uncertainty ellipse
    """
    if len(search_np.shape) == 2:          # grayscale → RGB for drawing
        vis = cv2.cvtColor(search_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        vis = search_np.copy().astype(np.uint8)

    # --- Bounding box (green) ---
    x1 = int(pred_x - half_box)
    y1 = int(pred_y - half_box)
    x2 = int(pred_x + half_box)
    y2 = int(pred_y + half_box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 80), 2)

    # --- Crosshair (bright green) ---
    cx, cy = int(pred_x), int(pred_y)
    cross_len = 20
    cv2.line(vis, (cx - cross_len, cy), (cx + cross_len, cy), (0, 255, 80), 2)
    cv2.line(vis, (cx, cy - cross_len), (cx, cy + cross_len), (0, 255, 80), 2)
    cv2.circle(vis, (cx, cy), 4, (0, 255, 80), -1)

    # --- 2σ Uncertainty Ellipse (red) ---
    ax = max(4, int(2 * sigma_x))
    ay = max(4, int(2 * sigma_y))
    cv2.ellipse(vis, (cx, cy), (ax, ay), 0, 0, 360, (255, 80, 80), 2)

    return Image.fromarray(vis)


def metric_card_html(label: str, value: str, unit: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-unit">{unit}</div>
    </div>"""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🔬 Drift-Sense Controls")
    st.markdown("---")

    rgb_mode = st.toggle("🌈 RGB Optical Microscope Mode", value=False,
                         help="Enable for 3-channel optical images (bonus mode)")
    st.markdown("---")
    st.markdown("### ⚙️ Algorithm Parameters")
    sigma_prior = st.slider("Stage Encoder Prior σ (px)", 50, 300, 150, 25,
                            help="Gaussian prior width for Bayesian MAP spatial penalty")
    conf_thresh  = st.slider("Confidence Threshold", 0.30, 0.95, 0.70, 0.05,
                             help="Below this → Safety Halt triggered")
    unc_thresh   = st.slider("Uncertainty Threshold (px)", 1.0, 10.0, 3.0, 0.5,
                             help="Above this → Safety Halt triggered")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#5a9fd4; line-height:1.6;'>
    <b>Algorithm Pipeline</b><br>
    1. Laplacian-Sobel Hybrid Edge Map<br>
    2. 15-angle Rotation Search Pyramid<br>
    3. Bayesian MAP Spatial Prior<br>
    4. 2D Paraboloid Sub-pixel Fit<br>
    5. Inverse-Hessian Uncertainty σ<br>
    6. Industrial Safety Halt Gate
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#3a7abf;'>
    Applied Materials | Semicon India Hackathon 2026<br>
    Bhashkar Fulara & Divyanshu Kandpal
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown("""
<div class="amat-header">
  <h1>🔬 Drift-Sense</h1>
  <p>AI-Powered Navigation-Error Recovery for Wafer Inspection Tools &nbsp;|&nbsp;
     Physics-Informed Bayesian Phase-Correlation Localization &nbsp;|&nbsp;
     Applied Materials Problem Statement</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Row ──
st.markdown('<div class="section-title">📁 Image Input</div>', unsafe_allow_html=True)
col_ref, col_srch = st.columns(2)

with col_ref:
    st.caption("**Reference Image** (100× magnification — high-res SEM patch)")
    ref_file = st.file_uploader("Upload Reference Image", type=["png","jpg","tiff","bmp"],
                                 key="ref_upload", label_visibility="collapsed")

with col_srch:
    st.caption("**Search Image** (10× magnification — wide-field 1000×1000)")
    srch_file = st.file_uploader("Upload Search Image", type=["png","jpg","tiff","bmp"],
                                  key="srch_upload", label_visibility="collapsed")

# ── Action Button ──
st.markdown("")
run_btn = st.button("🚀  Run Drift-Sense Recovery Solver",
                    type="primary",
                    use_container_width=True,
                    disabled=(ref_file is None or srch_file is None))

# ── Results Area ──
if run_btn and ref_file and srch_file:
    with st.spinner("⚙️ Solving — Bayesian MAP correlation in progress…"):
        # Save uploaded files to temp paths
        import tempfile, pathlib

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ref:
            tmp_ref.write(ref_file.read())
            ref_tmp_path = tmp_ref.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_srch:
            tmp_srch.write(srch_file.read())
            srch_tmp_path = tmp_srch.name

        solver = DriftSenseSolver()
        solver.sigma_prior = sigma_prior
        result = solver.predict(ref_tmp_path, srch_tmp_path)

        os.unlink(ref_tmp_path)
        os.unlink(srch_tmp_path)

    if result.get("status") == "ERROR":
        st.error(f"❌ Solver Error: {result.get('message')}")
    else:
        pred_x    = result["x"]
        pred_y    = result["y"]
        conf      = result["confidence"]
        sigma_x   = result["sigma_x"]
        sigma_y   = result["sigma_y"]
        theta     = result["theta_deg"]
        latency   = result["latency_ms"]
        halt_flag = result["safety_halt_flag"]

        st.markdown('<div class="section-title">📊 Recovery Results</div>', unsafe_allow_html=True)

        # ── Metric Cards ──
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card_html("Predicted X", f"{pred_x:.2f}", "pixels"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("Predicted Y", f"{pred_y:.2f}", "pixels"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("Confidence", f"{conf*100:.1f}", "%"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card_html("Latency", f"{latency:.1f}", "ms"), unsafe_allow_html=True)
        with c5:
            badge_cls = "halt-warn" if halt_flag else "halt-ok"
            badge_txt = "⛔ HALT" if halt_flag else "✅ CLEAR"
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Safety Halt</div>
              <div style="margin-top:10px;">
                <span class="{badge_cls}">{badge_txt}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Visualization Row ──
        st.markdown('<div class="section-title">🖼️ Visual Localization Output</div>', unsafe_allow_html=True)
        vis_col, info_col = st.columns([3, 1])

        srch_file.seek(0)
        search_np = np.array(Image.open(srch_file))

        overlay_pil = draw_overlay(search_np, pred_x, pred_y, sigma_x, sigma_y)

        with vis_col:
            st.image(overlay_pil, caption="Search Image — Green Box: Predicted Target | Red Ellipse: 2σ Uncertainty",
                     use_container_width=True)

        with info_col:
            st.markdown(f"""
            <div style="background:{AMAT_DARK};border:1px solid #0072CE44;border-radius:12px;padding:18px;font-size:0.82rem;color:#cfe3f5;line-height:1.9;">
            <b style="color:{AMAT_BLUE};">Detection Details</b><br><br>
            🎯 <b>Center X:</b> {pred_x:.2f} px<br>
            🎯 <b>Center Y:</b> {pred_y:.2f} px<br>
            🔄 <b>Best Angle:</b> {theta:+.1f}°<br>
            📐 <b>σ<sub>x</sub>:</b> {sigma_x:.2f} px<br>
            📐 <b>σ<sub>y</sub>:</b> {sigma_y:.2f} px<br>
            💡 <b>Confidence:</b> {conf*100:.1f}%<br>
            ⏱️ <b>Latency:</b> {latency:.1f} ms<br><br>
            <b style="color:{'#FF4B4B' if halt_flag else '#00C896'};">
              {'⛔ Safety Halt Active' if halt_flag else '✅ Safe to Proceed'}
            </b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            # Download result JSON
            import json
            st.download_button(
                "⬇️ Download Result JSON",
                data=json.dumps(result, indent=2),
                file_name="drift_sense_result.json",
                mime="application/json"
            )

elif not run_btn:
    # Placeholder info
    st.info("📂 Upload both images above, then click **Run Drift-Sense Recovery Solver** to begin localization.")
    
    st.markdown('<div class="section-title">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    | Step | Operation | Purpose |
    |------|-----------|---------|
    | 1 | **Laplacian-Sobel Hybrid Edge Map** | Noise-robust feature extraction |
    | 2 | **10× Spatial Downscale (INTER_AREA)** | Match reference to search image scale |
    | 3 | **15-angle Rotation Search Pyramid** | Handle ±3.5° stage rotational drift |
    | 4 | **Bayesian MAP Spatial Prior** | Prevent pitch-hopping in periodic arrays |
    | 5 | **2D Paraboloid Sub-pixel Fit** | Achieve <1 px landing accuracy |
    | 6 | **Inverse-Hessian Uncertainty σ** | Quantify match confidence covariance |
    | 7 | **Industrial Safety Halt Gate** | Prevent tool head collisions |
    """)
