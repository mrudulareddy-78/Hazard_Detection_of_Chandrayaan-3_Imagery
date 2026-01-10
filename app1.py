import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy
import torch.serialization
from utils import edge_inference, cloud_inference, load_edge_model


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Chandrayaan-3 Hazard Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# MODERN LIGHT THEME
# ======================================================
st.markdown("""
<style>
    /* Light gradient background */
    .stApp {
        background: linear-gradient(to bottom, #f0f4f8, #e2e8f0, #f8fafc);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Title card with colorful gradient */
    .title-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .title-card h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .title-card p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Colorful metric cards */
    .metric-box {
        background: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border-top: 5px solid;
        position: relative;
        overflow: hidden;
    }
    
    .metric-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .metric-box.safe { 
        border-top-color: #10b981;
        background: linear-gradient(to bottom, #ecfdf5 0%, white 50%);
    }
    .metric-box.crater { 
        border-top-color: #f59e0b;
        background: linear-gradient(to bottom, #fffbeb 0%, white 50%);
    }
    .metric-box.rocks { 
        border-top-color: #ec4899;
        background: linear-gradient(to bottom, #fdf2f8 0%, white 50%);
    }
    .metric-box.hazard { 
        border-top-color: #ef4444;
        background: linear-gradient(to bottom, #fef2f2 0%, white 50%);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, currentColor 0%, currentColor 100%);
        -webkit-background-clip: text;
    }
    
    .metric-box.safe .metric-value { color: #059669; }
    .metric-box.crater .metric-value { color: #d97706; }
    .metric-box.rocks .metric-value { color: #db2777; }
    .metric-box.hazard .metric-value { color: #dc2626; }
    
    .metric-label {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        opacity: 0.3;
    }
    
    /* Status banner */
    .status-box {
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 700;
        font-size: 1.25rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        letter-spacing: 1px;
    }
    
    .status-box.safe {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .status-box.caution {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .status-box.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Info panels */
    .info-panel {
        background: white;
        padding: 1.8rem;
        border-radius: 14px;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
    }
    
    .info-panel h4 {
        color: #1e293b;
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    .info-panel ul {
        color: #475569;
        line-height: 1.9;
    }
    
    .info-panel li strong {
        color: #334155;
    }
    
    /* Sidebar styling - Light theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1e293b;
    }
    
    /* Sidebar header */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
    }
    
    .sidebar-header h3 {
        color: white;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .sidebar-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    /* Legend box */
    .legend-box {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .legend-box h4 {
        color: #1e293b;
        margin-top: 0;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .legend-item {
        color: #475569;
        margin: 0.7rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 0.95rem;
    }
    
    .legend-color {
        width: 24px;
        height: 24px;
        border-radius: 6px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        color: #64748b;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #cbd5e1;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 2rem 0;
    }
    
    /* Image captions */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Section headers */
    h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================
st.markdown("""
<div class="title-card">
    <h1>🛰️ Chandrayaan-3 Hazard Detection System</h1>
    <p>AI-Powered Lunar Terrain Analysis & Navigation Support</p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# MODEL DEFINITION
# ======================================================
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
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = block(512 + 256, 256)
        self.dec2 = block(256 + 128, 128)
        self.dec3 = block(128 + 64, 64)
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

# ======================================================
# 🔒 PYTORCH 2.6 SAFE LOAD FIX
# ======================================================
torch.serialization.add_safe_globals([
    numpy.dtype,
    numpy._core.multiarray.scalar
])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImprovedUNet().to(device)
checkpoint = torch.load(
    "unet_rover_best.pth",
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ======================================================
# PREPROCESSING
# ======================================================
transform_img = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# CLASS MAPS - Original colors
# ======================================================
CLASS_COLORS = {
    0: (0, 0, 0),         # Background
    1: (255, 0, 0),       # Safe - Red
    2: (0, 255, 0),       # Rocks - Green
    3: (255, 255, 0),     # Crater - Yellow
}

def decode_mask(mask):
    h, w = mask.shape
    res = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        res[mask == cls] = color
    return res

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3>🚀 ISRO Mission Control</h3>
    <p>Chandrayaan-3 Rover Analytics</p>
</div>
""", unsafe_allow_html=True)

overlay_alpha = st.sidebar.slider("🎨 Overlay Transparency", 0.0, 1.0, 0.45, 0.05)

uploaded_files = st.sidebar.file_uploader(
    "📤 Upload Terrain Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

st.sidebar.markdown("""
<div class="legend-box">
    <h4>🌙 Terrain Legend</h4>
    <div class="legend-item">
        <span class="legend-color" style="background: #ff0000;"></span>
        <span>Safe Zone</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background: #00ff00;"></span>
        <span>Rocks/Boulders</span>
    </div>
    <div class="legend-item">
        <span class="legend-color" style="background: #ffff00;"></span>
        <span>Craters</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# MAIN PROCESSING
# ======================================================
if uploaded_files:
    for uploaded in uploaded_files:
        st.divider()
        st.markdown(f"### 📷 Analyzing: `{uploaded.name}`")
        
        img = Image.open(uploaded).convert("RGB")
        timg = transform_img(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = torch.argmax(model(timg), dim=1).squeeze().cpu().numpy()
        
        seg_mask = decode_mask(pred)
        base_img = np.array(img.resize((384, 384)))
        overlay = cv2.addWeighted(base_img, 1-overlay_alpha, seg_mask, overlay_alpha, 0)
        
        total = pred.size
        safe = np.sum(pred == 1) / total * 100
        rocks = np.sum(pred == 2) / total * 100
        crater = np.sum(pred == 3) / total * 100
        hazard = rocks + crater
        
        c1, c2, c3, c4 = st.columns(4)
        
        metrics_data = [
            (c1, safe, "Safe Zone", "safe", "✓"),
            (c2, crater, "Crater", "crater", "⚠"),
            (c3, rocks, "Rocks", "rocks", "●"),
            (c4, hazard, "Total Hazard", "hazard", "!")
        ]
        
        for col, val, label, css_class, icon in metrics_data:
            with col:
                st.markdown(f"""
                <div class="metric-box {css_class}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{val:.1f}%</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        if safe > 80:
            status_class, status_text = "safe", "✅ SAFE FOR NAVIGATION"
        elif safe > 60:
            status_class, status_text = "caution", "⚠️ PROCEED WITH CAUTION"
        else:
            status_class, status_text = "danger", "🚫 HAZARDOUS TERRAIN DETECTED"
        
        st.markdown(f"""
        <div class="status-box {status_class}">
            MISSION STATUS: {status_text}
        </div>
        """, unsafe_allow_html=True)
        
        t1, t2 = st.tabs(["🎨 Overlay Visualization", "📊 Detailed Analysis"])
        
        with t1:
            st.image(overlay, use_container_width=True, caption="AI-Enhanced Terrain Map")
        
        with t2:
            col1, col2 = st.columns(2)
            col1.image(seg_mask, caption="Segmentation Mask", use_container_width=True)
            col2.image(base_img, caption="Original Lunar Surface", use_container_width=True)
            
            # Enhanced chart with original colors
            fig, ax = plt.subplots(figsize=(10, 5))
            colors_chart = ['#ff0000', '#ffff00', '#00ff00']  # Red, Yellow, Green
            bars = ax.bar(["Safe Zone", "Crater", "Rocks"], [safe, crater, rocks], 
                         color=colors_chart, alpha=0.85, edgecolor='white', linewidth=2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='#1e293b')
            
            ax.set_ylim(0, 100)
            ax.set_ylabel("Coverage (%)", fontsize=13, fontweight='bold', color='#475569')
            ax.set_title("Terrain Distribution Analysis", fontsize=15, fontweight='bold', 
                        pad=20, color='#1e293b')
            ax.grid(axis='y', alpha=0.2, linestyle='--', color='#cbd5e1')
            ax.set_facecolor('#f8fafc')
            fig.patch.set_facecolor('white')
            ax.tick_params(colors='#64748b')
            
            for spine in ax.spines.values():
                spine.set_color('#e2e8f0')
                spine.set_linewidth(2)
            
            st.pyplot(fig)
            
            st.markdown("""
            <div class="info-panel">
                <h4>🧭 Navigation Recommendations</h4>
                <ul>
                    <li><strong>Safe Zones:</strong> Optimal paths for rover traversal with minimal obstacles</li>
                    <li><strong>Crater Areas:</strong> High-risk zones requiring avoidance or careful maneuvering</li>
                    <li><strong>Rocky Terrain:</strong> Reduce speed and activate enhanced obstacle detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-panel" style="max-width: 700px; margin: 3rem auto; text-align: center;">
        <h3 style="color: #667eea; margin-top: 0;">🚀 Ready to Analyze Lunar Terrain</h3>
        <p style="color: #64748b; font-size: 1.1rem; line-height: 1.9;">
            Upload lunar surface images using the sidebar to begin AI-powered hazard detection and terrain analysis.
        </p>
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 2px solid #e2e8f0;">
            <p style="color: #475569; font-weight: 600;"><strong>Detection Capabilities:</strong></p>
            <p style="color: #64748b;">✓ Safe navigation zones &nbsp;&nbsp; ✓ Crater identification &nbsp;&nbsp; ✓ Boulder detection</p>
        </div>
    </div>
    """, unsafe_allow_html=True)