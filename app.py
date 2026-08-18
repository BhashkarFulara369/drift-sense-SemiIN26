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
from PIL import Image
import time
import sys
import os

# Import the actual physics-engine
from localize import DriftSenseEngine

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
AMAT_ORANGE = "#FF9900"

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
  .halt-warn {{ background:{AMAT_ORANGE}22; color:{AMAT_ORANGE}; border:1px solid {AMAT_ORANGE}55;
                padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }}
  .halt-err  {{ background:{AMAT_RED}22;   color:{AMAT_RED};   border:1px solid {AMAT_RED}55;
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
                 status: str, half_box: float = 50.0) -> Image.Image:
    """
    Draws on the search image:
      - Bounding box (Green for Determinate, Orange for Ambiguous/Tie-Breaker)
      - Crosshair lines
    """
    if len(search_np.shape) == 2:
        vis = cv2.cvtColor(search_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        vis = search_np.copy().astype(np.uint8)

    color = (0, 255, 80) if status == "DETERMINATE" else (255, 165, 0)

    # --- Bounding box ---
    x1 = int(pred_x - half_box)
    y1 = int(pred_y - half_box)
    x2 = int(pred_x + half_box)
    y2 = int(pred_y + half_box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # --- Crosshair ---
    cx, cy = int(pred_x), int(pred_y)
    cross_len = 20
    cv2.line(vis, (cx - cross_len, cy), (cx + cross_len, cy), color, 2)
    cv2.line(vis, (cx, cy - cross_len), (cx, cy + cross_len), color, 2)
    cv2.circle(vis, (cx, cy), 4, color, -1)

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

    rgb_mode = st.toggle("🌈 RGB Optical Mode", value=False,
                         help="Enable for 3-channel optical images")
    st.markdown("---")
    st.markdown("### ⚙️ Deterministic Parameters")
    conf_thresh  = st.slider("n95 Danger Threshold", 0.0, 1.0, 0.70, 0.05,
                             help="Above this → Image is Ambiguous")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#5a9fd4; line-height:1.6;'>
    <b>Physics-Aware Pipeline</b><br>
    1. Scale/Rotation Phase Correlation<br>
    2. Global ZNCC Matching<br>
    3. LER Residual Verification<br>
    4. <b>AMAT Center Tie-Breaker</b><br>
    5. 2D Paraboloid Sub-pixel Fit<br>
    6. n95 Determinacy Verification
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
  <p>Physics-Aware Deterministic Sub-Pixel Navigation Recovery &nbsp;|&nbsp;
     Zero Machine Learning Dependencies &nbsp;|&nbsp;
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
run_btn = st.button("🚀 Run Drift-Sense Physics Engine",
                    type="primary",
                    use_container_width=True,
                    disabled=(ref_file is None or srch_file is None))

# ── Results Area ──
if run_btn and ref_file and srch_file:
    with st.spinner("⚙️ Solving — Extracting LER Fingerprints & Parabolic Fitting…"):
        
        ref_img = cv2.imdecode(np.frombuffer(ref_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        srch_img = cv2.imdecode(np.frombuffer(srch_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        engine = DriftSenseEngine()
        
        start_t = time.time()
        pred_x, pred_y, diagnostics = engine.localize(ref_img, srch_img)
        latency = (time.time() - start_t) * 1000

        forensics = diagnostics['forensics']
        n95 = forensics.n95_score
        status = forensics.status
        tie_occurred = diagnostics.get('tie_occurred', False)

        st.markdown('<div class="section-title">📊 Recovery Results</div>', unsafe_allow_html=True)

        # ── Metric Cards ──
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card_html("Target X", f"{pred_x:.2f}", "pixels"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("Target Y", f"{pred_y:.2f}", "pixels"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("n95 Ambiguity", f"{n95:.3f}", "ratio"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card_html("Latency", f"{latency:.1f}", "ms"), unsafe_allow_html=True)
        with c5:
            if status == "DETERMINATE":
                badge_cls = "halt-ok"
                badge_txt = "✅ DETERMINATE"
            else:
                badge_cls = "halt-warn"
                badge_txt = "⚠️ AMBIGUOUS"
                
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Engine Status</div>
              <div style="margin-top:10px;">
                <span class="{badge_cls}">{badge_txt}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Visualization Row ──
        st.markdown('<div class="section-title">🖼️ Visual Localization Output</div>', unsafe_allow_html=True)
        vis_col, info_col = st.columns([3, 1])

        overlay_pil = draw_overlay(srch_img, pred_x, pred_y, status)

        with vis_col:
            st.image(overlay_pil, caption="Search Image — Bounding Box indicates Parabolic Sub-Pixel Fit",
                     use_container_width=True)

        with info_col:
            st.markdown(f"""
            <div style="background:{AMAT_DARK};border:1px solid #0072CE44;border-radius:12px;padding:18px;font-size:0.82rem;color:#cfe3f5;line-height:1.9;">
            <b style="color:{AMAT_BLUE};">Engine Diagnostics</b><br><br>
            🎯 <b>Center X:</b> {pred_x:.2f} px<br>
            🎯 <b>Center Y:</b> {pred_y:.2f} px<br>
            📐 <b>n95 Metric:</b> {n95:.3f}<br>
            ⏱️ <b>CPU Latency:</b> {latency:.1f} ms<br><br>
            <b style="color:{'#FF9900' if tie_occurred else '#00C896'};">
              {'⚠️ AMAT Tie-Breaker Active' if tie_occurred else '✅ Unique LER Fingerprint'}
            </b>
            </div>
            """, unsafe_allow_html=True)

elif not run_btn:
    st.info("📂 Upload both images above, then click **Run Drift-Sense Physics Engine** to begin localization.")
    
    st.markdown('<div class="section-title">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.markdown(r"""
    | Step | Operation | Purpose |
    |------|-----------|---------|
    | 1 | **Log-Polar Phase Correlation** | Handles Stage $\Delta \\theta$ Rotation Drift |
    | 2 | **ZNCC Global Matching** | Heatmap generation across wide FOV |
    | 3 | **LER Residual Verifier** | Isolates microscopic high-frequency etch noise |
    | 4 | **AMAT Tie-Breaker** | Safely resolves FinFET periodic repetition traps |
    | 5 | **2D Parabolic Sub-Pixel Fit** | Exceeds discrete pixel-grid Nyquist limits |
    | 6 | **$n_{95}$ Determinacy Metric** | Mathematically guarantees structural safety |
    """)
