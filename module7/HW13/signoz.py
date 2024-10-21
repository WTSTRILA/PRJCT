import boto3
from botocore.exceptions import ClientError
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import io
import logging
import time
import psutil
import uvicorn
from typing import List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

metrics.set_meter_provider(MeterProvider())
meter = metrics.get_meter(__name__)

total_requests_counter = meter.create_counter(
    name="total_requests",
    description="Total number of requests to the model",
    unit="1",
)

prediction_time_histogram = meter.create_histogram(
    name="prediction_time_seconds",
    description="Prediction execution time",
    unit="seconds",
)

cpu_usage_histogram = meter.create_histogram(
    name="cpu_usage_percent",
    description="CPU usage in percentage",
    unit="percent",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_URL = os.getenv['S3_URL']
ACCESS_KEY = os.getenv['ACCESS_KEY']
SECRET_KEY = os.getenv['SECRET_KEY']
BUCKET_NAME = os.getenv['BUCKET_NAME']
MODEL_KEY = os.getenv['MODEL_KEY']

session = None

def download_model_from_s3():
    global session
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    model_buffer = io.BytesIO()
    start_time = time.time()

    try:
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_buffer)
        model_buffer.seek(0)
        logger.info("Model successfully downloaded from S3.")
        session = ort.InferenceSession(model_buffer.read())
        logger.info(f"Model loaded from S3 in {time.time() - start_time:.4f} seconds")

    except ClientError as e:
        logger.error(f"Error downloading model from S3: {e}")

download_model_from_s3()

def predict(image):
    if session is None:
        raise RuntimeError("Model session is not initialized.")

    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).numpy().astype(np.float32)

    cpu_usage = psutil.cpu_percent()
    cpu_usage_histogram.record(cpu_usage)

    logger.info("Starting prediction.")
    try:
        outputs = session.run(None, {session.get_inputs()[0].name: image})
        predicted = (outputs[0].flatten() > 0.5).astype(float)
        logger.info("Prediction completed.")
        return predicted[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

@app.post("/predict/")
async def get_prediction(files: List[UploadFile] = File(...)):
    total_requests_counter.add(len(files))
    results = []

    for file in files:
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            logger.error(f"Image processing error for file {file.filename}: {e}")
            results.append({"file": file.filename, "error": "Image processing error"})
            continue

        prediction_start_time = time.time()
        prediction = predict(image)
        prediction_time = time.time() - prediction_start_time
        prediction_time_histogram.record(prediction_time)

        if prediction is None:
            results.append({"file": file.filename, "error": "Prediction error"})
            continue

        result = "Fractured" if prediction == 0 else "Not Fractured"
        results.append({
            "file": file.filename,
            "prediction": result,
            "prediction_time": prediction_time
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
