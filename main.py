from fastapi import FastAPI, HTTPException
import os
import torch
import requests
import zipfile
from typing import List

app = FastAPI(title="Best Dental Model API")

# =====================================
# CONFIG
# =====================================
FILE_ID = "1TMmlXj1uSvFW0CHWYHE74Wu4Uk3FPPPT"
BASE_DIR = "./model_files"
ZIP_NAME = "best_dental_model_512.zip"
MODEL_FOLDER_NAME = "best_dental_model_512"

device = torch.device("cpu")
model = None

# =====================================
# GOOGLE DRIVE DOWNLOAD (HANDLES LARGE FILES)
# =====================================
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # if large file, find confirm token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_and_extract():
    os.makedirs(BASE_DIR, exist_ok=True)

    zip_path = os.path.join(BASE_DIR, ZIP_NAME)
    extract_path = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)

    if os.path.exists(extract_path):
        print("Model already downloaded and extracted.")
        return

    print("Downloading model ZIP...")
    download_file_from_google_drive(FILE_ID, zip_path)
    print("Download complete.")

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Extraction complete.")


# =====================================
# LOAD MODEL
# =====================================
@app.on_event("startup")
def load_model():
    global model

    download_and_extract()

    model_path = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model folder not found at: {model_path}")

    print("Loading model from:", model_path)

    # adjust depending on your actual model format
    try:
        # Try loading as TorchScript
        model = torch.jit.load(model_path, map_location=device)
    except:
        # Try loading normal .pth file
        model_file = os.path.join(model_path, "best_model_complete.pth")
        model = torch.load(model_file, map_location=device)

    model.eval()
    print("Model loaded successfully!")


# =====================================
# ROUTES
# =====================================
@app.get("/")
def home():
    return {"status": "Best Dental Model API Running"}

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
