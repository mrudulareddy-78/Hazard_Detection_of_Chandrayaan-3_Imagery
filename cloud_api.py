from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import numpy as np
import cv2
from PIL import Image
from model import ImprovedUNet
from utils import transform_img

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ImprovedUNet().to(device)
checkpoint = torch.load("unet_rover_best.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    timg = transform_img(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.argmax(model(timg), dim=1).squeeze().cpu().numpy()

    return {"mask": pred.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
