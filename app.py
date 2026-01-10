import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import requests
import numpy
import torch.serialization
import time

from utils import edge_inference, cloud_inference, load_edge_model, push_result_to_cloud


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Chandrayaan-3 Hazard Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# SIDEBAR (MODE FIRST)
# ======================================================
st.sidebar.markdown("## 🚀 Mission Control")

mode = st.sidebar.radio(
    "🌐 Inference Mode",
    ["🟢 Offline (Edge)", "☁️ Online (Cloud)", "🤖 Auto (Hybrid)"],
    index=2
)

overlay_alpha = st.sidebar.slider(
    "🎨 Overlay Transparency", 0.0, 1.0, 0.45, 0.05
)

uploaded_files = st.sidebar.file_uploader(
    "📤 Upload Lunar Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ======================================================
# TITLE
# ======================================================
st.markdown("""
<div style="text-align:center; padding:2rem;">
    <h1>🛰️ Chandrayaan-3 Hazard Detection System</h1>
    <p><b>Hybrid Edge–Cloud AI for Autonomous Rover Navigation</b></p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SYSTEM STATUS PANEL (KEY FOR DEMO 🔥)
# ======================================================
st.markdown("## 🖥️ System Status")

cloud_online = False
if mode != "🟢 Offline (Edge)":
    try:
        requests.get("http://localhost:8000/docs", timeout=1)
        cloud_online = True
    except:
        cloud_online = False

c1, c2, c3 = st.columns(3)

with c1:
    st.success("🟢 Edge AI: ACTIVE")

with c2:
    if cloud_online:
        st.success("☁️ Cloud: ONLINE")
    else:
        st.error("☁️ Cloud: OFFLINE")

with c3:
    st.info(f"⚙️ Mode: {mode}")

st.divider()
if st.button("🔄 Refresh System Status"):
    st.rerun()

# ======================================================
# PYTORCH SAFE LOAD FIX
# ======================================================
torch.serialization.add_safe_globals([
    numpy.dtype,
    numpy._core.multiarray.scalar
])

# ======================================================
# LOAD EDGE MODEL
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
edge_model = load_edge_model("unet_rover_best.pth", device)

# ======================================================
# CLASS COLORS
# ======================================================
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),       # Safe
    2: (0, 255, 0),       # Rocks
    3: (255, 255, 0),     # Crater
}

def decode_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color
    return out

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_files:
    for uploaded in uploaded_files:
        st.divider()
        st.markdown(f"### 📷 Analyzing: `{uploaded.name}`")

        img = Image.open(uploaded).convert("RGB")

        # -----------------------------
        # HYBRID INFERENCE LOGIC
        # -----------------------------
        if mode == "🟢 Offline (Edge)":
            pred, used_mode, latency, data_sent = edge_inference(edge_model, img, device)



        elif mode == "☁️ Online (Cloud)":
            pred, used_mode, latency,data_sent = cloud_inference(img)

            if pred is None:
                st.error("❌ Cloud unavailable")
                continue

        else:  # 🤖 Auto Hybrid
            pred, used_mode,latency,data_sent = cloud_inference(img)
            if pred is None:
                pred, used_mode,latency,data_sent = edge_inference(edge_model, img, device)
                data_sent = 0

                st.warning("⚠️ Cloud unavailable → Switched to EDGE automatically")
            else:
                st.success("☁️ Cloud available → Using cloud inference")

        st.markdown(f"**Inference Source:** {used_mode}")
        st.markdown("### 📡 Communication & Performance")

        c1, c2 = st.columns(2)

        with c1:
            if data_sent > 0:
                st.success(f"📤 Data sent to cloud: {data_sent:.1f} KB")
            else:
                st.info("🟢 No data sent (Edge Processing)")

        with c2:
            if latency is not None:
                st.success(f"⏱️ Inference latency: {latency:.1f} ms")

        
        # -----------------------------
        # POST PROCESSING
        # -----------------------------
        base_img = np.array(img.resize((384, 384)))
        seg_mask = decode_mask(pred)
        overlay = cv2.addWeighted(
            base_img, 1-overlay_alpha,
            seg_mask, overlay_alpha, 0
        )

        total = pred.size
        safe = np.sum(pred == 1) / total * 100
        rocks = np.sum(pred == 2) / total * 100
        crater = np.sum(pred == 3) / total * 100
        hazard = rocks + crater
                # -----------------------------
        # EDGE → CLOUD SYNC
        # -----------------------------
        result_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "safe": float(safe),
            "rocks": float(rocks),
            "crater": float(crater),
            "source": used_mode
        }

        synced = push_result_to_cloud(result_payload)

        if synced:
            st.success("☁️ Synced latest result to ground station")
        else:
            st.info("📴 Cloud not reachable (running autonomously)")


        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, safe, "Safe Zone"),
            (c2, crater, "Crater"),
            (c3, rocks, "Rocks"),
            (c4, hazard, "Hazard"),
        ]

        for col, val, label in metrics:
            with col:
                st.metric(label, f"{val:.1f}%")

        # -----------------------------
        # MISSION STATUS
        # -----------------------------
        if safe > 80:
            st.success("✅ MISSION STATUS: SAFE FOR NAVIGATION")
        elif safe > 60:
            st.warning("⚠️ MISSION STATUS: PROCEED WITH CAUTION")
        else:
            st.error("🚫 MISSION STATUS: HAZARDOUS TERRAIN")

        # -----------------------------
        # VISUALS
        # -----------------------------
        t1, t2 = st.tabs(["🎨 Overlay View", "📊 Analysis"])

        with t1:
            st.image(overlay, use_container_width=True)

        with t2:
            col1, col2 = st.columns(2)
            col1.image(seg_mask, caption="Segmentation Mask", use_container_width=True)
            col2.image(base_img, caption="Original Image", use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(
                ["Safe", "Crater", "Rocks"],
                [safe, crater, rocks],
                color=["red", "yellow", "green"]
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Coverage (%)")
            ax.set_title("Terrain Distribution")
            st.pyplot(fig)
            st.markdown("### 📊 Edge vs Cloud: Latency & Bandwidth")

        labels = []
        latencies = []
        bandwidths = []

        if "Edge" in used_mode:
            labels.append("Edge")
            latencies.append(latency)
            bandwidths.append(data_sent)

        if "Cloud" in used_mode:
            labels.append("Cloud")
            latencies.append(latency)
            bandwidths.append(data_sent)

        fig, ax1 = plt.subplots()

        ax1.bar(labels, latencies, color=["green", "blue"], alpha=0.7)
        ax1.set_ylabel("Latency (ms)")

        ax2 = ax1.twinx()
        ax2.plot(labels, bandwidths, color="red", marker="o")
        ax2.set_ylabel("Data Sent (KB)")

        ax1.set_title("Edge vs Cloud Performance Comparison")
        st.pyplot(fig)


else:
    st.info("👈 Upload lunar terrain images from the sidebar to begin analysis.")
