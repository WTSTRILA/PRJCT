import logging
import wandb
import numpy as np
from PIL import Image
import tensorflow as tf


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

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


class BoneConditionPredictor(BaseModelPredictor):
    def __init__(self):
        super().__init__(project_name="prjct_bones",
                         artifact_name='stanislavbochok-/prjct_bones/bone-model:v1',
                         model_filename='bone_model.h5')
        self.class_names = ['fractured', 'not fractured']

    def predict(self, image_path):
        preprocessed_img = self.load_preprocessed_image(image_path)

        logger.info(f"Running prediction on the image: {image_path}")
        predictions = self.loaded_model.predict(preprocessed_img)

        predicted_class = (predictions > 0.5).astype("int32")
        result = self.class_names[predicted_class[0][0]]

        logger.info(f"Prediction result: {result}")
        return result
