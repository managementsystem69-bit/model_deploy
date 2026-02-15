from fastapi import FastAPI, HTTPException
import os
import torch
import requests
import zipfile
from typing import List

app = FastAPI(title="Best Model Complete API")

# =====================================
# CONFIG
# =====================================
FILE_ID = "1i8-5nlAglEf2EjIutmHCXHl51HfreXw"  # Google Drive file ID of the ZIP
BASE_DIR = "./model_files"
MODEL_DIR = "best_model_complete"  # folder inside the zip
ZIP_NAME = "best_model_complete.pth.zip"

device = torch.device("cpu")
model = None


# =====================================
# DOWNLOAD & EXTRACT MODEL
# =====================================
def download_and_extract_model():
    os.makedirs(BASE_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, ZIP_NAME)
    model_path = os.path.join(BASE_DIR, MODEL_DIR)

    # Skip download if already extracted
    if os.path.exists(model_path):
        print("Model already exists. Skipping download.")
        return

    # Download zip from Google Drive
    print("Downloading model zip from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

    # Extract zip
    print("Extracting model zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Extraction complete.")


# =====================================
# LOAD MODEL
# =====================================
@app.on_event("startup")
def load_model():
    global model
    download_and_extract_model()

    model_path = os.path.join(BASE_DIR, MODEL_DIR, "best_model_complete.pth")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at: {model_path}")

    print(f"Loading model from file: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("Model loaded successfully!")


# =====================================
# ROUTES
# =====================================
@app.get("/")
def home():
    return {"status": "Best Model Complete API Running Successfully"}


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
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
