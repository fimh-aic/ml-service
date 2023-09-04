from app.ml.image import predictImage
from app.ml.receipt import recommend, case_folding
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import os
from dotenv import load_dotenv
import pandas as pd
from typing import Union

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

model_name_1 = "Food Image Recognition"
model_name_2 = "Receipt Recommendation"
version = "v1.0.0"

@app.get("/")
def read_root():
    return {"Welcome"}

@app.get('/info')
async def model_info():
    """ Return model information and version """
    return {
        "name": model_name,
        "version": version
    }

@app.post('/recognize')
async def recognize_image(image: UploadFile):
    """ Recognize the uploaded image """
    if "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="File must be an image")
    img = Image.open(image.file)
    predicted_class, confidence = predictImage(img)
    return {
        "name": model_name_1,
        "version": version,
        "result": predicted_class,
        "confidence": str(confidence)
    }

@app.post('/recommendation')
async def recomendation_(query: str, limit: Union[None, int] = None):
    """ Recognize the uploaded image """
    if not query:
        raise HTTPException(status_code=400, detail="Please provide an keyword to recommend")
    if not limit:
        limit = 10
    sanitized_query = case_folding(query)
    recommendation = recommend(sanitized_query, limit)
    return {
        "name": model_name_2,
        "version": version,
        "result": recommendation,
    }