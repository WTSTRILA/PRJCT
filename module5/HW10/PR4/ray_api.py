import ray
import wandb
import numpy as np
from PIL import Image
import tensorflow as tf

ray.init(ignore_reinit_error=True)


@ray.remote
class BaseModelPredictor:
    def __init__(self, project_name="prjct_bones", artifact_name='stanislavbochok-/prjct_bones/bone-model:v1',
                 model_filename='bone_model.h5'):
        self.run = wandb.init(project=project_name, job_type="predict", reinit=True)
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


def predict_image(image_path):
    predictor = BaseModelPredictor.remote()
    result = ray.get(predictor.predict.remote(image_path))
    return result


if __name__ == "__main__":
    image_path = '3.jpg'

    prediction = predict_image(image_path)
    print(f"Prediction: {prediction}")
