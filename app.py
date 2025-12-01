import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# =====================================
# 1. MODEL (same architecture)
# =====================================
class ImprovedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1 = block(512+256, 256)
        self.dec2 = block(256+128, 128)
        self.dec3 = block(128+64, 64)

        self.final = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d1 = self.dec1(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e1], dim=1))

        return self.final(d3)

# =====================================
# 2. SAFE CHECKPOINT LOADING
# =====================================
import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImprovedUNet().to(device)

checkpoint = torch.load(
    "unet_rover_best.pth",
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =====================================
# 3. PREPROCESSING
# =====================================
transform_img = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# 4. COLOR MAPS
# =====================================
CLASS_NAMES = {
    0: "Background",
    1: "Danger",
    2: "Safe",
    3: "Caution",
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (255, 255, 0),
}

def decode_mask(mask):
    h, w = mask.shape
    res = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        res[mask == cls] = color
    return res

# =====================================
# 5. STREAMLIT UI
# =====================================
st.set_page_config(page_title="Rover Segmentation", layout="wide")

st.markdown("<h1 style='text-align:center; color:#10b981;'>Hazard Detection Using Chandrayaan-3 Imagery</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📤 Upload Rover Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


tabs = st.tabs(["🖼️ Segmentation", "📊 Report & Analysis"])

# =====================================
# MULTI IMAGE PROCESSING
# =====================================
if uploaded_files:
    for uploaded in uploaded_files:

        st.divider()
        st.subheader(f"🖼️ Processing: {uploaded.name}")

        img = Image.open(uploaded).convert("RGB")
        
        timg = transform_img(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(timg)
            pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

        seg_mask = decode_mask(pred)
        overlay = cv2.addWeighted(
            np.array(img.resize((384, 384))), 0.6,
            seg_mask, 0.4,
            0
        )

        tabs = st.tabs(["🖼️ Segmentation", "📊 Report & Analysis"])

        # ---------------- TAB 1 ----------------
        with tabs[0]:
            st.image(img, caption="Uploaded Image", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(seg_mask, caption="🎨 Segmentation Mask", use_container_width=True)
            with col2:
                st.image(overlay, caption="🛰️ Overlay", use_container_width=True)

            st.markdown("""
            ### 🗺️ Legend
            - 🔴 **Red** = Danger  
            - 🟢 **Green** = Safe  
            - 🟡 **Yellow** = Caution  
            """)

        # ---------------- TAB 2 ----------------
        with tabs[1]:

            st.header("📊 Pixel Distribution")
            unique, counts = np.unique(pred, return_counts=True)
            pixel_stats = dict(zip(unique, counts))

            fig, ax = plt.subplots()
            ax.bar(
                [CLASS_NAMES[k] for k in pixel_stats.keys()],
                pixel_stats.values()
            )
            ax.set_ylabel("Pixel Count")
            ax.set_title("Class Occurrence")
            st.pyplot(fig)

            total = pred.size
            st.subheader("📈 Class Coverage (%)")

            for cls, cnt in pixel_stats.items():
                st.write(f"**{CLASS_NAMES[cls]}**: {cnt/total*100:.2f}%")

            st.subheader("📝 Auto-generated Report")

            summary = f"""
### Rover Hazard Report – {uploaded.name}

**Overall Classification Summary**
- Danger (Red): {pixel_stats.get(1,0)/total*100:.2f}%  
- Safe (Green): {pixel_stats.get(2,0)/total*100:.2f}%  
- Caution (Yellow): {pixel_stats.get(3,0)/total*100:.2f}%  

**Interpretation**
- The rover should **avoid red regions immediately**.  
- Green zones can be considered **safe traversal paths**.  
- Yellow zones indicate **partial instability / craters** and require caution.

**Inference Confidence**
This analysis is based on pixel-level segmentation using your trained UNet model.
"""

            st.markdown(summary)


