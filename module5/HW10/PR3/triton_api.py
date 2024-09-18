import wandb
import numpy as np
from PIL import Image
import tensorflow as tf
import logging
from io import BytesIO
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

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

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

predictor = BaseModelPredictor()

def _infer_fn(images: np.ndarray):
    if isinstance(images, np.ndarray) and images.ndim == 1:
        images = list(images)

    images = [Image.open(BytesIO(img)) for img in images]

    logger.info(f"Received {len(images)} images for prediction")

    results = [predictor.predict(image) for image in images]

    logger.info(f"Results = {results}")
    return np.array(results, dtype=np.object_)

def main():
    with Triton() as triton:
        logger.info("Loading models.")
        triton.bind(
            model_name="image_predictor",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="images", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="descriptions", dtype=np.object_, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=4),
        )
        logger.info("Serving inference")
        triton.serve()

if __name__ == "__main__":
    main()
