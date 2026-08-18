import gradio as gr
import cv2
import numpy as np
import time
import os
from localize import DriftSenseEngine

import spaces

def draw_overlay(search_np, pred_x, pred_y, status, half_box=50.0):
    if len(search_np.shape) == 2:
        vis = cv2.cvtColor(search_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        vis = search_np.copy().astype(np.uint8)

    color = (0, 255, 80) if status == "DETERMINATE" else (255, 165, 0)

    x1, y1 = int(pred_x - half_box), int(pred_y - half_box)
    x2, y2 = int(pred_x + half_box), int(pred_y + half_box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    cx, cy = int(pred_x), int(pred_y)
    cross_len = 20
    cv2.line(vis, (cx - cross_len, cy), (cx + cross_len, cy), color, 2)
    cv2.line(vis, (cx, cy - cross_len), (cx, cy + cross_len), color, 2)
    cv2.circle(vis, (cx, cy), 4, color, -1)

    return vis

@spaces.GPU
def run_driftsense(ref_img, srch_img):
    if ref_img is None or srch_img is None:
        return None, "❌ Please upload both images."
    
    # Convert to grayscale if needed
    if len(ref_img.shape) == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    if len(srch_img.shape) == 3:
        srch_img = cv2.cvtColor(srch_img, cv2.COLOR_RGB2GRAY)

    engine = DriftSenseEngine()
    
    start_t = time.time()
    pred_x, pred_y, diagnostics = engine.localize(ref_img, srch_img)
    latency = (time.time() - start_t) * 1000

    forensics = diagnostics['forensics']
    n95 = forensics.n95_score
    status = forensics.status
    
    overlay = draw_overlay(srch_img, pred_x, pred_y, status)
    
    status_msg = "✅ DETERMINATE (Unique LER Fingerprint)" if status == "DETERMINATE" else "⚠️ AMBIGUOUS (AMAT Tie-Breaker Active)"
    
    stats = f"""
### 📊 Engine Diagnostics
* **Target X:** `{pred_x:.2f} px`
* **Target Y:** `{pred_y:.2f} px`
* **n95 Ambiguity Score:** `{n95:.3f}`
* **CPU Latency:** `{latency:.1f} ms`

**Engine Status:** {status_msg}
    """
    
    return overlay, stats

# ==========================================
# GRADIO UI SETUP
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    button_primary_background_fill="#0072CE",
    button_primary_background_fill_hover="#1a3a6b"
)

with gr.Blocks(title="Drift-Sense | AMAT") as demo:
    gr.Markdown(
        """
        # 🔬 Drift-Sense
        **Physics-Aware Deterministic Sub-Pixel Navigation Recovery | Applied Materials | Semicon India Hackathon 2026**
        """
    )
    
    with gr.Row():
        ref_input = gr.Image(label="Reference Image (100x SEM)", type="numpy")
        srch_input = gr.Image(label="Search Image (10x SEM)", type="numpy")
        
    btn = gr.Button("🚀 Run Drift-Sense Physics Engine", variant="primary", size="lg")
    
    with gr.Row():
        output_image = gr.Image(label="Visual Localization Output")
        output_stats = gr.Markdown(label="Diagnostics")
        
    btn.click(fn=run_driftsense, inputs=[ref_input, srch_input], outputs=[output_image, output_stats])

if __name__ == "__main__":
    # Gradio 6.0 moved 'theme' to launch()
    demo.launch(theme=custom_theme)
