# import streamlit as st
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import time
# import numpy
# import torch.serialization

# from utils import (
#     edge_inference,
#     load_edge_model,
#     push_result_to_cloud
# )

# # ======================================================
# # PAGE CONFIG
# # ======================================================
# st.set_page_config(
#     page_title="Chandrayaan-3 Hazard Detection",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ======================================================
# # TITLE
# # ======================================================
# st.markdown("""
# <div style="text-align:center; padding:2rem;">
#     <h1>🛰️ Chandrayaan-3 Hazard Detection System</h1>
#     <p><b>Edge-first AI with Cloud Ground Station Sync</b></p>
# </div>
# """, unsafe_allow_html=True)

# # ======================================================
# # SIDEBAR
# # ======================================================
# st.sidebar.markdown("## 🚀 ISRO Mission Control")

# overlay_alpha = st.sidebar.slider(
#     "🎨 Overlay Transparency", 0.0, 1.0, 0.45, 0.05
# )

# uploaded_files = st.sidebar.file_uploader(
#     "📤 Upload Lunar Images",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=True
# )

# # ======================================================
# # PYTORCH SAFE LOAD FIX
# # ======================================================
# torch.serialization.add_safe_globals([
#     numpy.dtype,
#     numpy._core.multiarray.scalar
# ])

# # ======================================================
# # LOAD EDGE MODEL
# # ======================================================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# edge_model = load_edge_model("unet_rover_best.pth", device)

# # ======================================================
# # ✅ CORRECT CLASS COLORS (FIXED)
# # ======================================================
# CLASS_COLORS = {
#     0: (255, 0, 0),       # Safe (RED)
#     1: (0, 255, 0),       # Rocks (GREEN)
#     2: (255, 255, 0),     # Crater (YELLOW)
#     3: (0, 0, 0),         # Background
# }

# def decode_mask(mask):
#     h, w = mask.shape
#     out = np.zeros((h, w, 3), dtype=np.uint8)
#     for cls, color in CLASS_COLORS.items():
#         out[mask == cls] = color
#     return out

# # ======================================================
# # SYSTEM STATUS
# # ======================================================
# st.markdown("## 🖥️ System Status")

# c1, c2 = st.columns(2)
# with c1:
#     st.success("🟢 Edge AI: ACTIVE")
# with c2:
#     st.info("☁️ Cloud: Used only for telemetry")

# st.divider()

# # ======================================================
# # MAIN PROCESSING
# # ======================================================
# if uploaded_files:
#     for uploaded in uploaded_files:
#         st.divider()
#         st.markdown(f"### 📷 Analyzing: `{uploaded.name}`")

#         img = Image.open(uploaded).convert("RGB")

#         # -----------------------------
#         # EDGE INFERENCE (ALWAYS)
#         # -----------------------------
#         pred, latency = edge_inference(edge_model, img, device)

#         # -----------------------------
#         # POST PROCESSING
#         # -----------------------------
#         base_img = np.array(img.resize((384, 384)))
        
#         # Debug: Check unique classes in prediction
#         unique_classes = np.unique(pred)
#         st.write(f"🔍 Debug - Classes found in prediction: {unique_classes}")
#         st.write(f"🔍 Debug - Class distribution: {[(cls, np.sum(pred == cls)) for cls in unique_classes]}")
        
#         seg_mask = decode_mask(pred)

#         overlay = cv2.addWeighted(
#             base_img, 1 - overlay_alpha,
#             seg_mask, overlay_alpha, 0
#         )

#         total = pred.size
#         safe = np.sum(pred == 0) / total * 100
#         rocks = np.sum(pred == 1) / total * 100
#         crater = np.sum(pred == 2) / total * 100
#         hazard = rocks + crater

#         # -----------------------------
#         # EDGE → CLOUD SYNC
#         # -----------------------------
#         payload = {
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "safe": float(safe),
#             "rocks": float(rocks),
#             "crater": float(crater),
#             "source": "🟢 Edge"
#         }

#         synced = push_result_to_cloud(payload)

#         if synced:
#             st.success("☁️ Data synced to Ground Station (Render)")
#         else:
#             st.info("📴 No internet – running fully on Edge")

#         st.markdown(f"⏱️ **Inference latency:** {latency:.1f} ms")

#         # -----------------------------
#         # METRICS
#         # -----------------------------
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Safe Zone", f"{safe:.1f}%")
#         c2.metric("Crater", f"{crater:.1f}%")
#         c3.metric("Rocks", f"{rocks:.1f}%")
#         c4.metric("Hazard", f"{hazard:.1f}%")

#         # -----------------------------
#         # MISSION STATUS
#         # -----------------------------
#         if safe > 80:
#             st.success("✅ MISSION STATUS: SAFE FOR NAVIGATION")
#         elif safe > 60:
#             st.warning("⚠️ MISSION STATUS: PROCEED WITH CAUTION")
#         else:
#             st.error("🚫 MISSION STATUS: HAZARDOUS TERRAIN")

#         # -----------------------------
#         # VISUALIZATION
#         # -----------------------------
#         t1, t2 = st.tabs(["🎨 Overlay View", "📊 Analysis"])

#         with t1:
#             st.image(overlay, use_container_width=True)

#         with t2:
#             col1, col2 = st.columns(2)
#             col1.image(seg_mask, caption="Segmentation Mask", use_container_width=True)
#             col2.image(base_img, caption="Original Image", use_container_width=True)

#             fig, ax = plt.subplots()
#             ax.bar(
#                 ["Safe", "Crater", "Rocks"],
#                 [safe, crater, rocks],
#                 color=["red", "yellow", "green"]
#             )
#             ax.set_ylim(0, 100)
#             ax.set_ylabel("Coverage (%)")
#             ax.set_title("Terrain Distribution")
#             st.pyplot(fig)

# else:
#     st.info("👈 Upload lunar terrain images from the sidebar to begin analysis.")
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy
import torch.serialization

from utils import (
    edge_inference,
    load_edge_model,
    push_result_to_cloud
)

# FIX: Import transforms for proper preprocessing
from torchvision import transforms

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Chandrayaan-3 Hazard Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# TITLE
# ======================================================
st.markdown("""
<div style="text-align:center; padding:2rem;">
    <h1>🛰️ Chandrayaan-3 Hazard Detection System</h1>
    <p><b>Edge-first AI with Cloud Ground Station Sync</b></p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("## 🚀 ISRO Mission Control")

overlay_alpha = st.sidebar.slider(
    "🎨 Overlay Transparency", 0.0, 1.0, 0.45, 0.05
)

uploaded_files = st.sidebar.file_uploader(
    "📤 Upload Lunar Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

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
# FIX: PROPER TRANSFORM WITH NORMALIZATION
# ======================================================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# CLASS COLORS (ORIGINAL MAPPING)
# ======================================================
CLASS_COLORS = {
    0: (0, 0, 0),         # Background
    1: (255, 0, 0),       # Safe (RED)
    2: (0, 255, 0),       # Rocks (GREEN)
    3: (255, 255, 0),     # Crater (YELLOW)
}

def decode_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color
    return out

# ======================================================
# SYSTEM STATUS
# ======================================================
st.markdown("## 🖥️ System Status")

c1, c2 = st.columns(2)
with c1:
    st.success("🟢 Edge AI: ACTIVE")
with c2:
    st.info("☁️ Cloud: Used only for telemetry")

st.divider()

# ======================================================
# MAIN PROCESSING
# ======================================================
if uploaded_files:
    for uploaded in uploaded_files:
        st.divider()
        st.markdown(f"### 📷 Analyzing: `{uploaded.name}`")

        img = Image.open(uploaded).convert("RGB")

        # -----------------------------
        # EDGE INFERENCE (ALWAYS)
        # -----------------------------
        # FIX: Apply proper transform before inference
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        import time
        start = time.time()
        with torch.no_grad():
            pred = torch.argmax(edge_model(img_tensor), dim=1).squeeze().cpu().numpy()
        latency = (time.time() - start) * 1000

        # -----------------------------
        # POST PROCESSING
        # -----------------------------
        base_img = np.array(img.resize((384, 384)))
        seg_mask = decode_mask(pred)

        overlay = cv2.addWeighted(
            base_img, 1 - overlay_alpha,
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
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "safe": float(safe),
            "rocks": float(rocks),
            "crater": float(crater),
            "source": "🟢 Edge"
        }

        synced = push_result_to_cloud(payload)

        if synced:
            st.success("☁️ Data synced to Ground Station (Render)")
        else:
            st.info("📴 No internet – buffered locally, running fully on Edge")

        st.markdown(f"⏱️ **Inference latency:** {latency:.1f} ms")

        # -----------------------------
        # METRICS
        # -----------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Safe Zone", f"{safe:.1f}%")
        c2.metric("Crater", f"{crater:.1f}%")
        c3.metric("Rocks", f"{rocks:.1f}%")
        c4.metric("Hazard", f"{hazard:.1f}%")

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
        # VISUALIZATION
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

else:
    st.info("👈 Upload lunar terrain images from the sidebar to begin analysis.")