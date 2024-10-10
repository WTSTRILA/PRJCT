import os
import numpy as np
import tensorflow as tf
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
import uvicorn
import nest_asyncio
import wandb

app = FastAPI()

interpreter = None
input_details = None
output_details = None


def load_model():
    global interpreter, input_details, output_details
    run = wandb.init()
    artifact = run.use_artifact('stanislavbochok-/prjct_bones/quantized_bone_model:v0', type='model')
    artifact_dir = artifact.download()

    model_path = os.path.join(artifact_dir, 'quantized_model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def benchmark_predict(func):
    def wrapper(*args, **kwargs):
        num_runs = 10
        total_time = 0
        max_time = 0

        for _ in range(num_runs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            elapsed_time = end_time - start_time
            total_time += elapsed_time
            max_time = max(max_time, elapsed_time)

        average_time = total_time / num_runs

        return {
            "prediction": result,
            "average_prediction_time": average_time,
            "max_prediction_time": max_time,
            "num_runs": num_runs
        }

    return wrapper


def preprocess_image(image, img_height=180, img_width=180):
    img = image.convert('RGB')
    img = img.resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


@benchmark_predict
def predict_fracture(image: UploadFile):
    try:
        image_data = Image.open(BytesIO(image.file.read()))
        processed_image = preprocess_image(image_data)

        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        return "Not fractured" if prediction[0] > 0.5 else "Fractured"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    return predict_fracture(file)


if __name__ == "__main__":
    load_model()
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
