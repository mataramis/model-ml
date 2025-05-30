from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Izinkan semua origin (atau ganti sesuai frontend Anda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model saat startup
model = load_model("model.h5")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).resize((224, 224)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)[0]
    label = "shoplifting" if prediction[0] > 0.5 else "normal"
    confidence = float(prediction[0]) if label == "shoplifting" else 1 - float(prediction[0])

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }
