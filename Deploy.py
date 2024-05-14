from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pydantic import Field
import pickle
from font_recognition import PyTorchClassifier
from font_recognition import Preprocessing
import torch
import cv2
import numpy as np

app = FastAPI()

# Labels
labels = ['Scheherazade New', 'Marhey', 'Lemonada', 'IBM Plex Sans Arabic']

# Load the preprocessor pipeline
with open("preprocess_pipe.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)
    
# Initialize the preprocessor
preprocessor = Preprocessing(loaded_pipeline)

# Initialize the pytorch model
pytorch_classifier = PyTorchClassifier(2981, 512, 256, 4 , learning_rate=0.00025, epoch=50)
pytorch_classifier.load_state_dict(torch.load("best_model.pth"))
pytorch_classifier.eval()

@app.get("/")                    
def read_items(): 
    return "Welcome to the Font Recognition API"

# Input Schema is an image file
class Image(BaseModel):
    file: UploadFile = File(..., title="Image file to be uploaded")

# Output Schema
class Prediction(BaseModel):
    pred : str = Field(..., title="Font Prediction", example="Lemonada")

@app.post("/predict_font", response_model=Prediction)         
async def predict_font(file: UploadFile = File(...)):
    try:
        # Read the image file using cv2 and convert to grayscale
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        # Preprocess the image
        image = preprocessor.preprocess_test_data(image)
        # Predict the font
        pred = pytorch_classifier.predict(image)
        print(pred)
        return {"pred": labels[pred[0]]}
    except Exception as e:
        return {"error": str(e)}