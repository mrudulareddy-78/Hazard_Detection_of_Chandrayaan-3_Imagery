# import torch
# import numpy as np
# import cv2
# import requests
# from torchvision import transforms
# from model import ImprovedUNet

# # ------------------------
# # Preprocessing
# # ------------------------
# transform_img = transforms.Compose([
#     transforms.Resize((384, 384)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # ------------------------
# # EDGE INFERENCE
# # ------------------------
# def load_edge_model(weight_path, device):
#     model = ImprovedUNet().to(device)
#     checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model

# import time

# import time

# def edge_inference(model, image_pil, device):
#     start = time.time()

#     timg = transform_img(image_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         pred = torch.argmax(model(timg), dim=1).squeeze().cpu().numpy()

#     latency_ms = (time.time() - start) * 1000
#     data_sent_kb = 0.0  # EDGE sends nothing

#     return pred, "🟢 Edge", latency_ms, data_sent_kb


# # ------------------------
# # CLOUD INFERENCE
# # ------------------------
# CLOUD_API_URL = "http://localhost:8000/predict"

# import time
# import cv2
# import numpy as np
# import requests

# def cloud_inference(image_pil):
#     try:
#         start = time.time()

#         # Encode image
#         img_np = np.array(image_pil)
#         _, encoded = cv2.imencode(".png", img_np)
#         img_bytes = encoded.tobytes()

#         data_sent_kb = len(img_bytes) / 1024  # KB sent

#         # Simulated Earth–Moon delay
#         time.sleep(1.2)

#         response = requests.post(
#             "http://localhost:8000/predict",
#             files={"file": img_bytes},
#             timeout=5
#         )

#         latency_ms = (time.time() - start) * 1000

#         if response.status_code == 200:
#             mask = np.array(response.json()["mask"])
#             return mask, "☁️ Cloud", latency_ms, data_sent_kb

#         return None, "❌ Cloud Error", None, 0.0

#     except:
#         return None, "❌ Cloud Unreachable", None, 0.0
# # ------------------------
# # EDGE → CLOUD SYNC
# # ------------------------
# import json
# import subprocess

# SYNC_FILE = "sync_data.json"

# def push_result_to_cloud(payload):
#     try:
#         with open("sync_data.json", "w") as f:
#             json.dump(payload, f, indent=2)

#         subprocess.run(["git", "add", "sync_data.json"], check=True)
#         subprocess.run(["git", "commit", "-m", "Telemetry update"], check=True)
#         subprocess.run(["git", "push"], check=True)

#         return True   # ✅ DATA SYNC SUCCESS
#     except Exception as e:
#         print("Cloud sync failed:", e)
#         return False


import torch
import numpy as np
import numpy
import time
import requests
from torchvision import transforms
from model import ImprovedUNet
import torch.serialization

# ======================================================
# PYTORCH 2.6+ SAFE LOAD FIX
# ======================================================
torch.serialization.add_safe_globals([
    numpy.dtype,
    numpy._core.multiarray.scalar
])

# ======================================================
# 🔗 CLOUD ENDPOINT (RENDER)
# ======================================================
CLOUD_API_URL = "https://hazard-detection-of-chandrayaan-3-imagery.onrender.com/push"

# ======================================================
# IMAGE TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

# ======================================================
# LOAD EDGE MODEL
# ======================================================
def load_edge_model(model_path, device):
    model = ImprovedUNet().to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False   # IMPORTANT for PyTorch 2.6+
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ======================================================
# EDGE INFERENCE
# ======================================================
def edge_inference(model, img, device):
    start = time.time()

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).squeeze().cpu().numpy()

    latency = (time.time() - start) * 1000  # ms
    data_sent = 0  # edge does not send raw image

    return pred, "🟢 Edge", latency, data_sent

# ======================================================
# PUSH RESULT TO CLOUD (REAL COMMUNICATION)
# ======================================================
def push_result_to_cloud(payload):
    try:
        response = requests.post(
            CLOUD_API_URL,
            json=payload,
            timeout=3
        )
        return response.status_code == 200
    except:
        return False
