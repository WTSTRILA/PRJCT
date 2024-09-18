import wandb
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import uvicorn
import nest_asyncio

app = FastAPI()

def initialize_wandb():
    return wandb.init(project="prjct_bones", job_type="predict", reinit=True)

def load_model(run):
    artifact = run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'bone_model.h5')
    return tf.keras.models.load_model(model_path)

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

@app.post("/predict/")
async def predict_bone_fracture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse({"error": "Invalid image format", "details": str(e)})

    run = initialize_wandb()

    model = load_model(run)

    result = predict_fracture(model, image)

    run.finish()

    return JSONResponse({"prediction": result})

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
