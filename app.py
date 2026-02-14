from fastapi import FastAPI
import os
import gdown
import torch

app = FastAPI()

FILE_ID = "1Pw4vR5Tv5VkfSsi0Ke6QhUohTG5alqNG"
MODEL_PATH = "best_model1.pth"

device = torch.device("cpu")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")

download_model()

print("Loading PyTorch model...")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()
print("Model loaded successfully!")

@app.get("/")
def home():
    return {"status": "PyTorch Model API Running Successfully"}

@app.post("/predict")
def predict(data: dict):
    """
    Example Input:
    {
      "features": [1, 2, 3, 4]
    }
    """

    features = data.get("features")

    if features is None:
        return {"error": "Missing 'features' in request body"}

    # Convert input to tensor
    input_tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to list
    prediction = output.tolist()

    return {"prediction": prediction}
