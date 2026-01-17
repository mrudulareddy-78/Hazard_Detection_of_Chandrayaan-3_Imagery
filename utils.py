
import json
import os
import time
from typing import List

import numpy
import numpy as np
import requests
import torch
import torch.serialization
from torchvision import transforms

from model import ImprovedUNet

# ======================================================
# PYTORCH 2.6 SAFE LOAD
# ======================================================
torch.serialization.add_safe_globals([
    numpy.dtype,
    numpy._core.multiarray.scalar
])

# ======================================================
# CLOUD ENDPOINT (RENDER)
# ======================================================
CLOUD_API_URL = "https://hazard-detection-of-chandrayaan-3-imagery.onrender.com/update"

# Local buffer file for offline-first telemetry.
BUFFER_FILE = "telemetry_buffer.json"

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
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# ======================================================
# EDGE INFERENCE (ALWAYS WORKS)
# ======================================================
def edge_inference(model, img, device):
    start = time.time()

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1).squeeze().cpu().numpy()

    latency_ms = (time.time() - start) * 1000
    return pred, latency_ms

# ======================================================
# TELEMETRY BUFFER HELPERS
# ======================================================
def _load_buffer() -> List[dict]:
    if not os.path.exists(BUFFER_FILE):
        return []
    try:
        with open(BUFFER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt buffer: start fresh but do not block inference
        return []


def _persist_buffer(buffer: List[dict]) -> None:
    with open(BUFFER_FILE, "w", encoding="utf-8") as f:
        json.dump(buffer, f)


def _post_payload(payload: dict) -> bool:
    try:
        r = requests.post(
            CLOUD_API_URL,
            json=payload,
            timeout=3
        )
        return r.ok
    except Exception:
        return False


def _flush_buffer(buffer: List[dict]) -> bool:
    """Send buffered telemetry in-order; stop at first failure."""
    if not buffer:
        return True

    for idx, item in enumerate(buffer):
        if not _post_payload(item):
            # Preserve remaining items in order for a later retry
            _persist_buffer(buffer[idx:])
            return False

    # All sent; clear buffer
    _persist_buffer([])
    return True


# ======================================================
# PUSH RESULT TO CLOUD (OFFLINE-FIRST)
# ======================================================
def push_result_to_cloud(payload: dict) -> bool:
    """Send telemetry; buffer locally if offline.

    - First flush any older buffered records to preserve chronology.
    - If current send fails, append it to the buffer.
    - Always returns quickly to avoid blocking edge inference.
    """
    buffer = _load_buffer()

    # Try to flush older records before the newest one.
    if not _flush_buffer(buffer):
        buffer = _load_buffer()
        buffer.append(payload)
        _persist_buffer(buffer)
        return False

    # Send the latest payload after older ones are flushed.
    if _post_payload(payload):
        return True

    # Network still down: buffer the new payload.
    buffer = _load_buffer()
    buffer.append(payload)
    _persist_buffer(buffer)
    return False
