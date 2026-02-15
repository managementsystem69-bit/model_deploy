from fastapi import FastAPI, HTTPException
import os
import sys
import gdown
import torch
from typing import List

app = FastAPI(title="Best Model Complete API")

# =====================================
# CONFIG
# =====================================
FOLDER_ID = "1yYcM8kryj_T_BQZBdpQEJJBIae0A1NcQ"
BASE_DIR = "./model_files"
MODEL_DIR = "best_model_complete"  # This IS the model

device = torch.device("cpu")
model = None


# =====================================
# DOWNLOAD GOOGLE DRIVE FOLDER
# =====================================
def download_drive_folder():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # Download only if empty
    if not any(os.scandir(BASE_DIR)):
        print("Downloading best_model_complete from Google Drive...")
        gdown.download_folder(
            id=FOLDER_ID,
            output=BASE_DIR,
            quiet=False,
            use_cookies=False
        )
        print("Download complete.")
    else:
        print("Files already exist. Skipping download.")


# =====================================
# LOAD MODEL
# =====================================
@app.on_event("startup")
def load_model():
    global model

    download_drive_folder()

    model_path = os.path.join(BASE_DIR, MODEL_DIR)

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model directory not found at: {model_path}"
        )

    print(f"Loading model from directory: {model_path}")

    # Try normal torch.load first
    try:
        model = torch.load(model_path, map_location=device)
        print("Loaded using torch.load()")
    except Exception:
        # If it was saved as TorchScript
        model = torch.jit.load(model_path, map_location=device)
        print("Loaded using torch.jit.load()")

    model.eval()
    print("Model loaded successfully!")


# =====================================
# ROUTES
# =====================================
@app.get("/")
def home():
    return {
        "status": "Best Model Complete API Running Successfully"
    }


@app.post("/predict")
def predict(features: List[float]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not features:
        raise HTTPException(status_code=400, detail="Features list is empty")

    try:
        input_tensor = torch.tensor(
            [features],
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        return {"prediction": output.tolist()}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
