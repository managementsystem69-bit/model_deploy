from fastapi import FastAPI, HTTPException
import os
import gdown
import torch
from typing import List

app = FastAPI(title="Railway PyTorch Model API")

# =====================================
# CONFIG
# =====================================
FOLDER_ID = "1yYcM8kryj_T_BQZBdpQEJJBIae0A1NcQ"
DOWNLOAD_DIR = "./model_files"
device = torch.device("cpu")

model = None  # Global model reference


# =====================================
# DOWNLOAD GOOGLE DRIVE FOLDER
# =====================================
def download_folder_from_drive():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # Only download if directory is empty
    if not any(os.scandir(DOWNLOAD_DIR)):
        print("Downloading full model folder from Google Drive...")
        gdown.download_folder(
            id=FOLDER_ID,
            output=DOWNLOAD_DIR,
            quiet=False,
            use_cookies=False
        )
        print("Download completed!")
    else:
        print("Model folder already exists. Skipping download.")


# =====================================
# FIND MODEL FILE AUTOMATICALLY
# =====================================
def find_model_file():
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            if file.endswith(".pth"):
                return os.path.join(root, file)
    return None


# =====================================
# LOAD MODEL AT STARTUP
# =====================================
@app.on_event("startup")
def load_model():
    global model

    download_folder_from_drive()

    model_path = find_model_file()

    if model_path is None:
        raise RuntimeError("No .pth model file found inside Drive folder!")

    print(f"Loading model from: {model_path}")

    model = torch.load(model_path, map_location=device)
    model.eval()

    print("Model loaded successfully!")


# =====================================
# ROUTES
# =====================================
@app.get("/")
def home():
    return {"status": "Railway PyTorch Model API Running Successfully"}


@app.post("/predict")
def predict(features: List[float]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not features:
        raise HTTPException(status_code=400, detail="Features list is empty")

    try:
        input_tensor = torch.tensor([features], dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        return {"prediction": output.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
