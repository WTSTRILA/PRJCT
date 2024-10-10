import os
import io
import threading
import boto3
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
from kq import Job, Queue
from PIL import Image
import numpy as np
import uvicorn
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

producer = None
queue = None
model_lock = threading.Lock()
session = None

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')  
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = 'bone-model'
MODEL_KEY = 'bone_model.onnx'


def load_model():
    global session
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    model_buffer = io.BytesIO()

    try:
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_buffer)
        model_buffer.seek(0)
        logger.info("Модель успешно загружена из S3.")
        session = ort.InferenceSession(model_buffer.read())
    except ClientError as e:
        logger.error(f"Ошибка при загрузке модели из S3: {e}")
        raise


def ensure_model_loaded():
    global session
    with model_lock:
        if session is None:
            logger.info("Модель не загружена, пытаюсь загрузить...")
            load_model()


def preprocess_image(image_bytes, img_height=180, img_width=180):

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((img_height, img_width))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Ошибка при предобработке изображения: {e}")
        raise


def enqueue_prediction(image_bytes):

    try:
        processed_image = preprocess_image(image_bytes)
        job = Job(func='predict_fracture', args=[processed_image.tolist()], timeout=30) 
        queue.enqueue(job)
        logger.info("Задание на предсказание успешно отправлено в очередь Kafka.")
    except Exception as e:
        logger.error(f"Ошибка при отправке задания в очередь: {e}")
        raise


@app.on_event("startup")
def startup_event():

    global producer, queue
    ensure_model_loaded()

    try:
        producer = KafkaProducer(
            bootstrap_servers='127.0.0.1:9092',
            value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v
        )
        logger.info("KafkaProducer инициализирован.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации KafkaProducer: {e}")
        raise

    try:
        queue = Queue(topic='inference-requests', producer=producer)
        logger.info("Очередь Kafka 'inference-requests' инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации очереди Kafka: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():

    global producer
    if producer:
        producer.close()
        logger.info("Соединение с KafkaProducer закрыто.")


@app.post("/predict/")
async def predict_bone_fracture(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        if not contents:
            logger.warning("Получен пустой файл.")
            return JSONResponse({"error": "Empty file"}, status_code=400)

        enqueue_prediction(contents)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса на предсказание: {e}")
        return JSONResponse({"error": "Не удалось поставить задание на предсказание", "details": str(e)}, status_code=500)

    return JSONResponse({"message": "Запрос на инференс отправлен в очередь."})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
