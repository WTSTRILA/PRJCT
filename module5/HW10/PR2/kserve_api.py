import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import json
from typing import Dict
import kserve
from kserve import Model, ModelServer


class BaseModelPredictor:
    def __init__(self, project_name="prjct_bones", artifact_name='stanislavbochok-/prjct_bones/bone-model:v1',
                 model_filename='bone_model.h5'):
        self.run = wandb.init(project=project_name, job_type="predict")
        artifact = self.run.use_artifact(artifact_name, type='model')
        artifact_dir = artifact.download()
        self.loaded_model = tf.keras.models.load_model(f"{artifact_dir}/{model_filename}")
        self.class_names = ['fractured', 'not fractured']

    def load_preprocessed_image(self, image_path, img_height=180, img_width=180):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_height, img_width))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        preprocessed_img = self.load_preprocessed_image(image_path)
        predictions = self.loaded_model.predict(preprocessed_img)
        predicted_class = (predictions > 0.5).astype("int32")
        return self.class_names[predicted_class[0][0]]


class CustomModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.base_model_predictor = BaseModelPredictor()
        self.ready = False

    def load(self, model_fp):
        self.base_model_predictor.loaded_model = tf.keras.models.load_model(model_fp, compile=False)
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        instances = request.get("instances", [])
        data = instances[0].get("image", {}).get("b64", "")
        image = base64.b64decode(data)
        with open("/tmp/temp_image.jpg", "wb") as f:
            f.write(image)

        preprocessed_image = self.base_model_predictor.load_preprocessed_image("/tmp/temp_image.jpg")
        ds = tf.data.Dataset.from_tensors(preprocessed_image).batch(1)
        y_pred = self.base_model_predictor.loaded_model.predict(ds)

        y_pred_class = (y_pred > 0.5).astype("int32")
        prediction_class = self.base_model_predictor.class_names[y_pred_class[0][0]]

        return {"predictions": [prediction_class]}


if __name__ == "__main__":
    model = CustomModel("custom-model")
    ModelServer().start([model])
