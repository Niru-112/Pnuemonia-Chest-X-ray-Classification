import mlflow.pytorch
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

model = mlflow.pytorch.load_model("models:/Chest_Xray_Pneumonia_MLflow_Project/Production")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": "Pneumonia" if prediction == 1 else "Normal"}
