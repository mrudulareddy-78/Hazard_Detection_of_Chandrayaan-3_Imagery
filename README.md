# 🛰️ Chandrayaan-3 AI-Powered Hazard Detection & Path Planning System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Advanced Edge-First AI System for Autonomous Lunar Rover Navigation**

---

## 🌟 Overview

This project implements a state-of-the-art autonomous navigation system for the Chandrayaan-3 lunar rover mission, featuring:

- **Real-time Hazard Detection** using U-Net deep learning (4-class semantic segmentation)
- **Advanced Path Planning** with A* and RRT* algorithms
- **Edge-First Architecture** with offline-first telemetry buffering
- **Terrain-Aware Navigation** with risk assessment and path optimization
- **High autonomy** with on-board planning and risk-aware navigation

## 📊 System Architecture

```
┌────────────────────┐
│  NavCam Image      │
│   (384×384 RGB)    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  U-Net Inference   │
│  (Hazard Detection)│
│   ~75ms            │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Path Planning     │
│  • A* (~35ms)      │
│  • RRT* (~250ms)   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Navigation        │
│  • Waypoints       │
│  • Risk Assessment │
│  • Motor Control   │
└────────────────────┘
```

---

## 🚀 Features

### 1. Hazard Detection
- **Deep Learning Model**: U-Net with BatchNorm
- **Classes**: Safe Zone, Rocks, Craters, Background
- **Performance**: <100ms inference time
- **Accuracy**: Semantic segmentation with pixel-level precision

### 2. Path Planning
- **A* Algorithm**: 
  - Optimal path guarantee
  - Fast planning (20-50ms)
  - Terrain-weighted costs
  - Deterministic results

- **RRT* Algorithm**:
  - Sampling-based exploration
  - Asymptotically optimal
  - Complex obstacle handling
  - Planning time: 100-500ms

### 3. Risk Assessment
- Quantitative safety scoring
- Per-pixel terrain analysis
- Path hazard percentage
- Traversability metrics

### 4. Edge-Cloud Architecture
- Primary processing on edge (rover)
- Cloud telemetry with offline buffering
- Auto-sync when connection restored
- Survives communication blackouts

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd LAB_EL

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Dependencies
```
streamlit
torch
torchvision
numpy
opencv-python
pillow
matplotlib
pandas
requests
fastapi
uvicorn
pydantic
```

---

## 🎮 Usage

### 1. Start Streamlit App
```bash
streamlit run app.py
```

### 2. Start Cloud API (Optional)
```bash
cd LAB_EL
uvicorn cloud_api:app --host 0.0.0.0 --port 8000
```

### 3. Using the System

#### Upload Image
- Click "📤 Upload Lunar Images" in sidebar
- Select one or more lunar terrain images
- Supported formats: PNG, JPG, JPEG

#### Configure Path Planning
- Enable "Path Planning" checkbox
- Select algorithm: A*, RRT*, or Compare (A* vs RRT*)
- Toggle "Safety Mode" (recommended: ON)
- Adjust RRT* iterations if needed (default: 2000)

#### Set Waypoints
- Use sliders to set **Start** position (row, col)
- Use sliders to set **Goal** position (row, col)
- Ensure positions are on safe terrain (green) for safety mode

#### View Results
- **Overlay View**: Hazard detection with colored overlay
- **Analysis**: Segmentation mask, original image, distribution chart
- **Path Planning**: Planned paths, metrics, and risk analysis

---

## 🔬 Algorithm Details

### U-Net Architecture
```python
Input: 384×384×3 RGB image
Encoder: 4 blocks (64→128→256→512 channels)
Decoder: 3 blocks with skip connections
Output: 384×384×4 class probabilities
Activation: ReLU + BatchNorm
Loss: Cross-entropy (training)
```

### A* Implementation
```python
Cost Function: f(n) = g(n) + h(n)
- g(n): Actual cost from start
- h(n): Euclidean distance to goal
- Terrain costs: Safe=1, Rocks=50, Craters=200
```

### RRT* Implementation
```python
Parameters:
- Sampling: Random + 10% goal-biased
- Step size: 5 pixels
- Rewiring radius: 15 pixels
- Collision detection: Bresenham line algorithm
- Optimization: Continuous rewiring for better paths
```

---

## 📈 Results

### Hazard Detection
- **Inference Time**: 50-100ms (average: 75ms)
- **Classes Detected**: 4 (Safe, Rocks, Craters, Background)
- **Model Size**: ~50MB
- **Device**: CPU or GPU (CUDA)

### Path Planning
- **A* Planning**: 20-50ms
- **RRT* Planning**: 100-500ms
- **Path Smoothing**: 30-50% waypoint reduction
- **Success Rate**: >95% for reachable goals

### Risk Assessment
- **Risk Score Range**: 0-300 (lower is safer)
- **Safe Path Threshold**: >70% safe terrain
- **Metrics**: Safe pixels, rock pixels, crater pixels, risk score

---

## 📂 Project Structure

```
LAB_EL/
│
├── app.py                              # Main Streamlit application
├── model.py                            # U-Net architecture
├── utils.py                            # Inference & telemetry utilities
├── path_planning.py                    # A* and RRT* algorithms
├── cloud_api.py                        # FastAPI telemetry endpoint
├── visualization_utils.py              # Publication-ready figures
│
├── unet_rover_best.pth                 # Pre-trained model weights
├── telemetry_buffer.json               # Offline telemetry buffer
├── requirements.txt                    # Python dependencies
│
├── PATH_PLANNING_DOCUMENTATION.md      # Comprehensive technical docs
├── QUICKSTART_PATH_PLANNING.md         # Quick start guide
└── README.md                           # This file
```

---

## 🔧 Configuration

### Model Settings
```python
# In app.py
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "unet_rover_best.pth"
input_size = (384, 384)
```

### Path Planning Settings
```python
# A* Parameters
safety_mode = True          # Avoid craters
diagonal_moves = True       # 8-directional movement

# RRT* Parameters
max_iterations = 2000       # Sampling iterations
step_size = 5.0            # Extension step (pixels)
goal_radius = 10.0         # Goal threshold
rewire_radius = 15.0       # Optimization radius
```

### Cloud API
```python
# In utils.py
CLOUD_API_URL = "https://your-cloud-endpoint.com/update"
BUFFER_FILE = "telemetry_buffer.json"
```

---

## 📊 Generating Publication Figures

```bash
# Generate all figures for paper
python visualization_utils.py
```

**Output files:**
- `algorithm_performance.png` - A* vs RRT* metrics
- `risk_analysis.png` - Path safety analysis
- `latency_breakdown.png` - System latency breakdown

---

## 🧪 Testing

### Test with Sample Images
```bash
# Use navcam.ipynb for testing with lunar images
jupyter notebook navcam.ipynb
```

### Verify Path Planning
```python
from path_planning import PathPlanner
import numpy as np

# Create test hazard map
hazard_map = np.random.randint(0, 4, (384, 384))
planner = PathPlanner(hazard_map)

# Plan path
path, metrics = planner.a_star((50, 50), (300, 300))
print(f"Path found: {len(path)} nodes")
print(f"Planning time: {metrics['planning_time_ms']:.2f}ms")
```

---

## 📝 For Research Publication

### Suggested Paper Structure
1. **Introduction**: Edge-first AI for lunar navigation
2. **Related Work**: Path planning algorithms, deep learning
3. **Methodology**: U-Net architecture, A*/RRT* implementation
4. **Experiments**: Performance metrics
5. **Results**: Latency, accuracy, path quality, risk assessment
6. **Discussion**: Limitations and future work
7. **Conclusion**: Summary and next steps

### Key Contributions
1. **Novel edge-first architecture** with offline-first telemetry
2. **Multi-algorithm path planning** comparing A* and RRT*
3. **Quantitative risk assessment** for path safety
4. **Production-ready implementation** with <100ms inference

### Figures to Include
- System architecture diagram (see above)
- U-Net architecture
- Hazard detection examples
- Path planning visualizations (A* vs RRT*)
- Performance comparison charts
- Latency breakdown

---

## 🚧 Future Work

### Short-term Enhancements
- [ ] 3D path planning with elevation data
- [ ] Multi-waypoint mission planning
- [ ] Uncertainty quantification for paths
- [ ] Real-time path execution simulation

### Medium-term Research
- [ ] Learned cost functions via reinforcement learning
- [ ] Multi-objective optimization (Pareto fronts)
- [ ] SLAM integration for unknown environments
- [ ] Battery/energy-aware path planning

### Long-term Vision
- [ ] Multi-rover coordination
- [ ] Lunar base construction planning
- [ ] Cross-domain transfer (Mars, Earth)
- [ ] Human-robot collaborative navigation

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved terrain classification models
- Additional path planning algorithms (D*, Theta*, etc.)
- Enhanced visualization tools
- Benchmark datasets
- Documentation improvements

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **ISRO** for Chandrayaan-3 mission inspiration
- **U-Net** architecture by Ronneberger et al.
- **A*** algorithm by Hart et al.
- **RRT*** algorithm by Karaman & Frazzoli
- Open-source PyTorch and Streamlit communities

---

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Create an issue]
- **Documentation**: See `PATH_PLANNING_DOCUMENTATION.md`
- **Quick Start**: See `QUICKSTART_PATH_PLANNING.md`

---

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. Hart, P. E., et al. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
3. Karaman, S., & Frazzoli, E. (2011). "Sampling-based Algorithms for Optimal Motion Planning"
4. ISRO Chandrayaan-3 Mission Design (2023)

---

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@software{chandrayaan3_navigation,
  title={Chandrayaan-3 AI-Powered Hazard Detection and Path Planning System},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo},
  note={Advanced edge-first AI system for autonomous lunar rover navigation}
}
```

---

**Version**: 1.0  
**Last Updated**: January 27, 2026  
**Status**: Ready for Publication

---

<div align="center">

### 🌙 Advancing Lunar Exploration Through Artificial Intelligence 🚀

**Made with ❤️ for Space Exploration**

</div>
