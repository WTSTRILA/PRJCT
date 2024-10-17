import wandb
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import nest_asyncio

app = FastAPI()

class BoneFractureModel:
    def __init__(self, name: str, artifact_name: str, artifact_version: str):
        self.name = name
        self.artifact_name = artifact_name
        self.artifact_version = artifact_version
        self.model = None

    def load(self):
        run = wandb.init(project="prjct_bones", job_type="predict", reinit=True)
        artifact_path = f'stanislavbochok-/prjct_bones/{self.artifact_name}:{self.artifact_version}'
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, 'bone_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        run.finish()

    def preprocess(self, data):
        image = Image.open(io.BytesIO(data))
        img = image.convert('RGB')
        img = img.resize((180, 180))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, data):
        processed_image = self.preprocess(data)
        prediction = self.model.predict(processed_image)
        return prediction[0][0]

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, data):
        predictions = [model.predict(data) for model in self.models]
        avg_prediction = np.mean(predictions)
        return avg_prediction

    def postprocess(self, result):
        return JSONResponse({"prediction": "Перелом" if result > 0.5 else "Без перелома", "confidence": float(result)})

bone_model_1 = BoneFractureModel("bone-fracture-model-1", "bone-model", "v0")
bone_model_2 = BoneFractureModel("bone-fracture-model-2", "bone-model", "v1")

bone_model_1.load()
bone_model_2.load()

ensemble_model = EnsembleModel([bone_model_1, bone_model_2])

@app.post("/predict/")
async def predict_bone_fracture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = ensemble_model.predict(contents)
        return ensemble_model.postprocess(result)
    except Exception as e:
        return JSONResponse({"error": "Неверный формат изображения", "details": str(e)})

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=80)
