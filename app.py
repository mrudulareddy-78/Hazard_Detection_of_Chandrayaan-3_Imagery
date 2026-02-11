
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy
import torch.serialization
import requests
from streamlit_image_coordinates import streamlit_image_coordinates

from utils import (
    edge_inference,
    load_edge_model,
    push_result_to_cloud,
    auto_sync_buffer,
    push_path_run_to_cloud,
    CLOUD_API_URL
)

# Import path planning module
from path_planning import PathPlanner

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
# AUTO-SYNC BUFFERED DATA ON STARTUP
# ======================================================
if 'buffer_synced' not in st.session_state:
    st.session_state.buffer_synced = True
    synced_count = auto_sync_buffer()
    if synced_count > 0:
        st.toast(f"✅ Synced {synced_count} buffered record(s) to cloud!", icon="☁️")

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

# Help & Tutorial Section
with st.sidebar.expander("📖 Quick Help & Tutorial", expanded=False):
    st.markdown("""
    ### 📷 Image Analysis Guide

    **Quick Start:**
    1. Upload one or more lunar images (PNG/JPG)
    2. Tune overlay transparency if needed
    3. (Optional) Enable Path Planning
    4. Review overlay, analysis, and path results

    **Features:**
    - ✅ Per-image hazard segmentation
    - ✅ Detailed terrain statistics
    - ✅ Path planning with A* / RRT*
    - ✅ Cloud telemetry sync

    **Documentation:**
    📚 Check project folder for:
    - README.md
    - PATH_PLANNING_DOCUMENTATION.md
    - QUICKSTART_PATH_PLANNING.md
    """)

overlay_alpha = st.sidebar.slider(
    "🎨 Overlay Transparency", 0.0, 1.0, 0.45, 0.05
)

st.sidebar.markdown("---")

# ======================================================
# DB HISTORY VIEW (CLOUD API)
# ======================================================
with st.sidebar.expander("🗂️ Cloud DB History", expanded=False):
    st.caption(f"API: {CLOUD_API_URL}")
    try:
        telemetry_res = requests.get(f"{CLOUD_API_URL}/history", timeout=3)
        path_res = requests.get(f"{CLOUD_API_URL}/path_runs", timeout=3)
        telemetry_data = telemetry_res.json() if telemetry_res.ok else []
        path_data = path_res.json() if path_res.ok else []

        st.markdown("**Telemetry (latest 5)**")
        if telemetry_data:
            st.dataframe(telemetry_data[-5:], use_container_width=True)
        else:
            st.info("No telemetry records yet.")

        st.markdown("**Path Runs (latest 5)**")
        if path_data:
            st.dataframe(path_data[-5:], use_container_width=True)
        else:
            st.info("No path run records yet.")
    except Exception:
        st.warning("Cloud API not reachable. Check CLOUD_API_URL or API server.")
st.sidebar.markdown("### 📷 Image Configuration")
uploaded_files = st.sidebar.file_uploader(
    "📤 Upload Lunar Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Path Planning Options
st.sidebar.markdown("---")
st.sidebar.markdown("## 🛤️ Path Planning")
enable_path_planning = st.sidebar.checkbox("Enable Path Planning", value=True)

if enable_path_planning:
    path_algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["A*", "RRT*", "Compare (A* vs RRT*)"]
    )
    
    safety_mode = st.sidebar.checkbox("Safety Mode (Avoid Hazards)", value=True)
    
    if path_algorithm in ["RRT*", "Compare (A* vs RRT*)"]:
        rrt_iterations = st.sidebar.slider("RRT* Iterations", 500, 5000, 2000, 500)
    
    # Color Legend
    st.sidebar.markdown("### 🎨 Terrain Color Legend")
    st.sidebar.markdown("""
    - 🟥 **RED** = Safe zones (smooth)
    - 🟩 **GREEN** = Rocks (textured)
    - 🟨 **YELLOW** = Craters (hazardous)
    - ⚫ **BLACK** = Background
    """)

st.sidebar.markdown("---")

# ======================================================
# PYTORCH SAFE LOAD FIX
# ======================================================
torch.serialization.add_safe_globals([
    numpy.dtype,
    numpy._core.multiarray.scalar
])

# ======================================================
# LOAD EDGE MODEL (CACHED)
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model_cached(model_path: str, device_name: str):
    return load_edge_model(model_path, device_name)


edge_model = load_model_cached("unet_rover_best.pth", device)

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
# CLASS COLORS - TERRAIN CLASSIFICATION
# ======================================================
# RED = Safe zones (Class 1) - Preferred for navigation
# GREEN = Rocks/Highly textured (Class 2) - Traversable with caution  
# YELLOW = Craters (Class 3) - Hazardous, avoid
CLASS_COLORS = {
    0: (0, 0, 0),         # Background (Black)
    1: (255, 0, 0),       # Safe (RED) - Smooth, traversable terrain
    2: (0, 255, 0),       # Rocks (GREEN) - Highly textured, rocky terrain
    3: (255, 255, 0),     # Crater (YELLOW) - Impact craters, avoid
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
        if enable_path_planning:
            t1, t2, t3 = st.tabs(["🎨 Overlay View", "📊 Analysis", "🛤️ Path Planning"])
        else:
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
        
        # -----------------------------
        # PATH PLANNING TAB
        # -----------------------------
        if enable_path_planning:
            with t3:
                st.markdown("### 🛤️ Autonomous Navigation Path Planning")
                
                # Initialize path planner
                planner = PathPlanner(pred)

                st.markdown("#### Click Map to Set Start/Goal")
                click_col1, click_col2, click_col3 = st.columns(3)
                click_mode_key = f"click_mode_{uploaded.name}"
                start_row_key = f"start_row_{uploaded.name}"
                start_col_key = f"start_col_{uploaded.name}"
                goal_row_key = f"goal_row_{uploaded.name}"
                goal_col_key = f"goal_col_{uploaded.name}"

                default_start_row = st.session_state.get(start_row_key, 50)
                default_start_col = st.session_state.get(start_col_key, 50)
                default_goal_row = st.session_state.get(goal_row_key, pred.shape[0] - 50)
                default_goal_col = st.session_state.get(goal_col_key, pred.shape[1] - 50)

                if click_mode_key not in st.session_state:
                    st.session_state[click_mode_key] = None

                if click_col1.button("Set Start by Click", key=f"set_start_btn_{uploaded.name}"):
                    st.session_state[click_mode_key] = "start"
                if click_col2.button("Set Goal by Click", key=f"set_goal_btn_{uploaded.name}"):
                    st.session_state[click_mode_key] = "goal"
                if click_col3.button("Clear Click Mode", key=f"clear_click_btn_{uploaded.name}"):
                    st.session_state[click_mode_key] = None

                if st.session_state[click_mode_key] == "start":
                    st.info("Click on the map to set the START position.")
                elif st.session_state[click_mode_key] == "goal":
                    st.info("Click on the map to set the GOAL position.")

                overlay_with_markers = overlay.copy()
                cv2.circle(
                    overlay_with_markers,
                    (default_start_col, default_start_row),
                    8,
                    (255, 255, 255),
                    -1
                )
                cv2.circle(
                    overlay_with_markers,
                    (default_start_col, default_start_row),
                    5,
                    (0, 200, 83),
                    -1
                )
                cv2.circle(
                    overlay_with_markers,
                    (default_goal_col, default_goal_row),
                    10,
                    (255, 255, 255),
                    -1
                )
                cv2.circle(
                    overlay_with_markers,
                    (default_goal_col, default_goal_row),
                    7,
                    (213, 0, 0),
                    -1
                )

                click_result = streamlit_image_coordinates(
                    overlay_with_markers,
                    key=f"click_map_{uploaded.name}"
                )

                if click_result and st.session_state[click_mode_key] in ["start", "goal"]:
                    clicked_col = int(click_result["x"])
                    clicked_row = int(click_result["y"])

                    clicked_row = max(0, min(pred.shape[0] - 1, clicked_row))
                    clicked_col = max(0, min(pred.shape[1] - 1, clicked_col))

                    if st.session_state[click_mode_key] == "start":
                        st.session_state[start_row_key] = clicked_row
                        st.session_state[start_col_key] = clicked_col
                    else:
                        st.session_state[goal_row_key] = clicked_row
                        st.session_state[goal_col_key] = clicked_col

                    st.session_state[click_mode_key] = None
                    st.toast("✅ Point selected from map.", icon="📍")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Set Start Position")
                    start_row = st.slider(
                        "Start Row",
                        0,
                        pred.shape[0] - 1,
                        value=default_start_row,
                        key=start_row_key
                    )
                    start_col = st.slider(
                        "Start Column",
                        0,
                        pred.shape[1] - 1,
                        value=default_start_col,
                        key=start_col_key
                    )
                
                with col2:
                    st.markdown("#### Set Goal Position")
                    goal_row = st.slider(
                        "Goal Row",
                        0,
                        pred.shape[0] - 1,
                        value=default_goal_row,
                        key=goal_row_key
                    )
                    goal_col = st.slider(
                        "Goal Column",
                        0,
                        pred.shape[1] - 1,
                        value=default_goal_col,
                        key=goal_col_key
                    )
                
                start_pos = (start_row, start_col)
                goal_pos = (goal_row, goal_col)
                
                # Find safe positions automatically
                start_pos_safe = planner.find_safe_position(start_pos, search_radius=30)
                goal_pos_safe = planner.find_safe_position(goal_pos, search_radius=30)

                # Show if positions were adjusted
                if start_pos_safe != start_pos:
                    st.info(f"🔄 Start position adjusted to nearest safe zone: {start_pos_safe}")
                if goal_pos_safe != goal_pos:
                    st.info(f"🔄 Goal position adjusted to nearest safe zone: {goal_pos_safe}")

                # Use adjusted positions for planning
                start_pos = start_pos_safe
                goal_pos = goal_pos_safe

                # Plan paths
                paths = {}
                metrics = {}

                st.markdown("---")
                st.markdown("#### Planning Results")

                with st.spinner("Computing optimal paths..."):
                    # A* Method
                    if path_algorithm in ["A*", "Compare (A* vs RRT*)"]:
                        path_a, metrics_a = planner.a_star(start_pos, goal_pos, safety_mode=safety_mode)
                        if path_a:
                            path_a = planner.smooth_path(path_a, iterations=3)
                            paths["A*"] = path_a
                            metrics["A*"] = metrics_a

                    # RRT* Method
                    if path_algorithm in ["RRT*", "Compare (A* vs RRT*)"]:
                        path_rrt, metrics_rrt = planner.rrt_star(
                            start_pos, goal_pos,
                            max_iterations=rrt_iterations,
                            safety_mode=safety_mode
                        )
                        if path_rrt:
                            path_rrt = planner.smooth_path(path_rrt, iterations=3)
                            paths["RRT*"] = path_rrt
                            metrics["RRT*"] = metrics_rrt

                # If no paths found, try again with safety mode disabled (except legacy-only runs)
                if not paths:
                    st.warning("⚠️ No path found with safety mode ON. Trying with relaxed constraints...")
                    with st.spinner("Retrying with safety mode OFF..."):
                        if path_algorithm in ["A*", "Compare (A* vs RRT*)"]:
                            path_a, metrics_a = planner.a_star(start_pos, goal_pos, safety_mode=False)
                            if path_a:
                                path_a = planner.smooth_path(path_a, iterations=3)
                                paths["A* (Relaxed)"] = path_a
                                metrics["A* (Relaxed)"] = metrics_a

                        if path_algorithm in ["RRT*", "Compare (A* vs RRT*)"]:
                            path_rrt, metrics_rrt = planner.rrt_star(
                                start_pos, goal_pos,
                                max_iterations=rrt_iterations * 2,
                                safety_mode=False
                            )
                            if path_rrt:
                                path_rrt = planner.smooth_path(path_rrt, iterations=3)
                                paths["RRT* (Relaxed)"] = path_rrt
                                metrics["RRT* (Relaxed)"] = metrics_rrt

                # Display metrics comparison
                if paths:
                    if len(paths) > 1:
                        st.markdown("##### Algorithm Comparison")
                        comp_cols = st.columns(len(paths))
                        for idx, (algo, path) in enumerate(paths.items()):
                            with comp_cols[idx]:
                                st.metric(f"{algo} Path Length", f"{len(path)} nodes")
                                st.metric("Planning Time", f"{metrics[algo]['planning_time_ms']:.1f} ms")
                                st.metric("Nodes Explored", metrics[algo]['nodes_explored'])

                                risk_data = planner.calculate_path_risk(path)
                                st.metric("Safe Path %", f"{risk_data['safe_percentage']:.1f}%")
                    else:
                        algo = list(paths.keys())[0]
                        path = paths[algo]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Path Length", f"{len(path)} nodes")
                        col2.metric("Planning Time", f"{metrics[algo]['planning_time_ms']:.1f} ms")
                        col3.metric("Nodes Explored", metrics[algo]['nodes_explored'])

                        risk_data = planner.calculate_path_risk(path)
                        col4.metric("Safe Path %", f"{risk_data['safe_percentage']:.1f}%")

                    st.markdown("---")
                    st.markdown("##### Path Visualization")

                    # Create visualization
                    fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 5))
                    if len(paths) == 1:
                        axes = [axes]

                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D

                    terrain_legend = [
                        Patch(facecolor=(1.0, 0.0, 0.0), label="Safe"),
                        Patch(facecolor=(0.0, 1.0, 0.0), label="Rocks"),
                        Patch(facecolor=(1.0, 1.0, 0.0), label="Crater"),
                        Patch(facecolor=(0.0, 0.0, 0.0), label="Background"),
                    ]

                    for idx, (algo, path) in enumerate(paths.items()):
                        ax = axes[idx]

                        # Display base image + segmentation for context
                        ax.imshow(base_img, alpha=0.25)
                        ax.imshow(seg_mask, alpha=0.75)

                        # Use distinct colors for A* vs RRT*
                        if "A*" in algo:
                            path_color = '#00BCD4'
                        else:
                            path_color = '#FF6F00'
                        path_style = '-'
                        path_alpha = 0.9
                        path_label = f"{algo} Path"

                        # Draw path with outline for contrast
                        path_array = np.array(path)
                        ax.plot(
                            path_array[:, 1], path_array[:, 0],
                            color="black", linestyle=path_style, linewidth=5,
                            alpha=0.35, zorder=3
                        )
                        ax.plot(
                            path_array[:, 1], path_array[:, 0],
                            color=path_color, linestyle=path_style, linewidth=3,
                            label=path_label, alpha=path_alpha, zorder=4
                        )

                        # Mark start and goal (use the adjusted positions)
                        ax.scatter(
                            start_pos[1], start_pos[0],
                            s=120, c="#00C853", edgecolors="white", linewidths=2,
                            marker="o", label="Start", zorder=5
                        )
                        ax.scatter(
                            goal_pos[1], goal_pos[0],
                            s=160, c="#D50000", edgecolors="white", linewidths=2,
                            marker="*", label="Goal", zorder=5
                        )

                        # Calculate path metrics
                        risk_data = planner.calculate_path_risk(path)

                        ax.set_title(f"{algo} Path\n"
                                   f"Length: {len(path)} | Safe: {risk_data['safe_percentage']:.1f}%\n"
                                   f"Time: {metrics[algo]['planning_time_ms']:.1f}ms",
                                   fontsize=10)
                        path_legend = [
                            Line2D([0], [0], color=path_color, linestyle=path_style, linewidth=3, label=path_label),
                            Line2D([0], [0], marker="o", color="w", label="Start",
                                   markerfacecolor="#00C853", markeredgecolor="white", markersize=8),
                            Line2D([0], [0], marker="*", color="w", label="Goal",
                                   markerfacecolor="#D50000", markeredgecolor="white", markersize=10),
                        ]
                        ax.legend(
                            handles=path_legend + terrain_legend,
                            loc="upper right",
                            fontsize=8,
                            framealpha=0.9
                        )
                        ax.axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Detailed risk analysis
                    st.markdown("---")
                    st.markdown("##### Detailed Path Risk Analysis")

                    risk_cols = st.columns(len(paths))
                    for idx, (algo, path) in enumerate(paths.items()):
                        with risk_cols[idx]:
                            st.markdown(f"**{algo}**")
                            risk_data = planner.calculate_path_risk(path)

                            hazard_pixels = risk_data['rock_pixels'] + risk_data['crater_pixels']
                            safe_pct_display = 100.0 if hazard_pixels == 0 else risk_data['safe_percentage']
                            risk_score_display = 0.0 if hazard_pixels == 0 else risk_data['risk_score']

                            # Color mapping: RED = Safe, GREEN = Rocks, YELLOW = Craters
                            st.write(f"🟥 Safe (RED): {risk_data['safe_pixels']} pixels ({safe_pct_display:.1f}%)")
                            st.write(f"🟩 Rocks (GREEN): {risk_data['rock_pixels']} pixels")
                            st.write(f"🟨 Craters (YELLOW): {risk_data['crater_pixels']} pixels")
                            st.write(f"⚠️ Total Risk Score: {risk_score_display:.2f}")

                            if hazard_pixels == 0:
                                st.success("✅ No hazards on this path (all safe terrain)")
                                continue

                            if risk_data['safe_percentage'] > 90:
                                st.success("✅ Very Safe Path")
                            elif risk_data['safe_percentage'] > 70:
                                st.info("⚠️ Moderately Safe")
                            else:
                                st.warning("⚠️ High Risk Path")

                            # Log path run once per unique configuration
                            rrt_iter_value = rrt_iterations if "rrt_iterations" in locals() else None
                            run_key = (
                                f"{uploaded.name}|{algo}|{start_pos}|{goal_pos}"
                                f"|{safety_mode}|{rrt_iter_value}"
                            )
                            if "db_logged_runs" not in st.session_state:
                                st.session_state.db_logged_runs = set()
                            if run_key not in st.session_state.db_logged_runs:
                                payload = {
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "image_name": uploaded.name,
                                    "algorithm": algo,
                                    "safety_mode": bool(safety_mode),
                                    "start_row": int(start_pos[0]),
                                    "start_col": int(start_pos[1]),
                                    "goal_row": int(goal_pos[0]),
                                    "goal_col": int(goal_pos[1]),
                                    "planning_time_ms": float(metrics[algo]["planning_time_ms"]),
                                    "nodes_explored": int(metrics[algo]["nodes_explored"]),
                                    "path_length": int(len(path)),
                                    "total_cost": float(metrics[algo].get("total_cost", 0)),
                                    "safe_percentage": float(risk_data["safe_percentage"]),
                                    "risk_score": float(risk_data["risk_score"])
                                }
                                push_path_run_to_cloud(payload)
                                st.session_state.db_logged_runs.add(run_key)

                else:
                    st.error("❌ No valid path found even with relaxed constraints!")
                    st.markdown("#### 🔍 Troubleshooting Tips:")
                    st.markdown("""
                    - **Terrain Analysis**: The current terrain may have too many obstacles
                    - **Adjust Positions**: Try moving start/goal to different locations
                    - **Check Safe Zones**: Ensure there are sufficient safe (green) zones
                    - **Increase RRT Iterations**: More iterations may find a path in complex terrain

                    **Current Terrain Statistics:**
                    """)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Safe Zones", f"{safe:.1f}%")
                    col2.metric("Rocks", f"{rocks:.1f}%")
                    col3.metric("Craters", f"{crater:.1f}%")
                    col4.metric("Total Hazards", f"{hazard:.1f}%")

                    # Show the terrain map with start/goal
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(seg_mask)
                    ax.plot(start_pos[1], start_pos[0], 'go', markersize=15,
                           label='Start', markeredgecolor='white', markeredgewidth=2)
                    ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=20,
                           label='Goal', markeredgecolor='white', markeredgewidth=2)
                    ax.set_title("Terrain Map with Start/Goal Positions")
                    ax.legend()
                    ax.axis('off')
                    st.pyplot(fig)

else:
    st.info("👈 Upload lunar terrain images from the sidebar to begin analysis.")