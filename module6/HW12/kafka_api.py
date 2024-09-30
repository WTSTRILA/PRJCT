from confluent_kafka import Producer, Consumer, KafkaError
import threading
import io
import os
from PIL import Image
import wandb
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import nest_asyncio

app = FastAPI()

producer = Producer({'bootstrap.servers': 'localhost:9092'})
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'model-consumer',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['inference-requests'])

run = wandb.init(project="prjct_bones", job_type="predict", reinit=True)

def load_model():
    artifact = run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'bone_model.h5')
    return tf.keras.models.load_model(model_path)

model = load_model()

def preprocess_image(image, img_height=180, img_width=180):
    img = image.convert('RGB')
    img = img.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fracture(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return "Fractured" if prediction[0] > 0.5 else "Not Fractured"

def process_message(message):
    try:
        contents = message.value()
        image = Image.open(io.BytesIO(contents))
        result = predict_fracture(model, image)

        print(f'Результат предсказания: {result}')
        return result
    except Exception as e:
        print(f"Ошибка при обработке сообщения: {e}")
        return "Ошибка при обработке изображения"

def consume_messages():
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print("Ошибка Kafka:", msg.error())
                break

        print("Получено сообщение:", msg.value())
        prediction = process_message(msg)
        print(f'Prediction: {prediction}')

threading.Thread(target=consume_messages, daemon=True).start()

@app.post("/predict/")
async def predict_bone_fracture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        producer.produce('inference-requests', contents)
        producer.flush()
        print("Изображение успешно отправлено в Kafka.")
    except Exception as e:
        return JSONResponse({"error": "Invalid image format", "details": str(e)})

    return JSONResponse({"message": "Запрос на инференс отправлен в очередь."})

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        run.finish()