import boto3
from botocore.exceptions import ClientError
import onnxruntime as ort
import logging
import io
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
from typing import List
import numpy as np
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
from datetime import datetime  
import os 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

S3_URL = os.getenv('S3_URL')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_KEY = os.getenv('MODEL_KEY')

ARIZE_API_KEY = os.getenv('ARIZE_API_KEY')
ARIZE_SPACE_KEY = os.getenv('ARIZE_SPACE_KEY')

arize_client = Client(space_id=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

feature_column_names = [
    "file_name",
    "prediction",
]

schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="prediction",
    prediction_score_column_name="PREDICTION_SCORE",
    actual_label_column_name=None,
    actual_score_column_name=None,
    feature_column_names=feature_column_names,
)

session = None


def download_model_from_s3():
    global session
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    model_buffer = io.BytesIO()

    try:
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_buffer)
        model_buffer.seek(0)
        logger.info("Model successfully downloaded from S3.")
        session = ort.InferenceSession(model_buffer.read())
    except ClientError as e:
        logger.error(f"Error downloading model from S3: {e}")


def ensure_model_loaded():
    if session is None:
        logger.info("Model not loaded, attempting to download...")
        download_model_from_s3()


def predict(image):
    ensure_model_loaded()

    if session is None:
        raise RuntimeError("Model session is not initialized.")

    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).numpy().astype(np.float32)

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
    results = []
    predictions_data = []

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
        prediction_score = prediction
        prediction_ts = datetime.now()

        predictions_data.append({
            "prediction_id": file.filename,
            "prediction": result,
            "PREDICTION_SCORE": prediction_score,
            "file_name": file.filename,
            "prediction_ts": prediction_ts,
        })

        results.append({"file": file.filename, "result": result})

    if predictions_data:
        import pandas as pd
        test_dataframe = pd.DataFrame(predictions_data)

        test_dataframe['prediction_ts'] = pd.to_datetime(test_dataframe['prediction_ts'])
        logger.info(f"Predictions DataFrame:\n{test_dataframe.head()}")

        try:
            response = arize_client.log(
                model_id="bone-fracture-model",
                model_version="v1",
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                dataframe=test_dataframe,
                schema=schema,
            )
            logger.info("Successfully logged predictions to Arize.")
        except Exception as e:
            logger.error(f"Failed to log prediction to Arize: {e}")

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
