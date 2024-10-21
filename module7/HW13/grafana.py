import boto3
from botocore.exceptions import ClientError
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
import psutil
import uvicorn
from typing import List
from prometheus_client import Counter, Histogram
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

total_requests_counter = Counter(
    'total_requests', 'Total number of requests to the model'
)

prediction_time_histogram = Histogram(
    'prediction_time_seconds', 'Prediction execution time in seconds'
)

cpu_usage_histogram = Histogram(
    'cpu_usage_percent', 'CPU usage in percentage'
)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

S3_URL = os.getenv('S3_URL')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_KEY = os.getenv('MODEL_KEY')

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
    cpu_usage_histogram.observe(cpu_usage)

    logger.info("Starting prediction.")
    try:
        prediction_start_time = time.time()
        outputs = session.run(None, {session.get_inputs()[0].name: image})
        prediction_time = time.time() - prediction_start_time
        prediction_time_histogram.observe(prediction_time)

        predicted = (outputs[0].flatten() > 0.5).astype(float)
        logger.info("Prediction completed.")
        return predicted[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

@app.post("/predict/")
async def get_prediction(files: List[UploadFile] = File(...)):
    total_requests_counter.inc(len(files))
    results = []

    for file in files:
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            logger.error(f"Image processing error for file {file.filename}: {e}")
            results.append({"file": file.filename, "error": "Image processing error"})
            continue

        prediction = predict(image)
        if prediction is None:
            results.append({"file": file.filename, "error": "Prediction error"})
            continue

        result = "Fractured" if prediction == 0 else "Not Fractured"
        results.append({
            "file": file.filename,
            "prediction": result,
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
