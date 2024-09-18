import numpy as np
from PIL import Image
from pytriton.client import ModelClient

class TritonModelClient:
    def __init__(self, server_url="0.0.0.0:8001", model_name="image_predictor"):
        self.client = ModelClient(server_url, model_name)

    def prepare_image(self, image_path, img_height=180, img_width=180):
        img = Image.open(image_path).resize((img_height, img_width))
        return np.expand_dims(np.array(img) / 255.0, axis=0)

    def predict(self, image_paths):
        images_np = np.vstack([self.prepare_image(img_path) for img_path in image_paths])
        result_dict = self.client.infer_batch(images=images_np)
        return result_dict["descriptions"]

def main():
    client = TritonModelClient()
    image_paths = ["3.jpg"]
    results = client.predict(image_paths)
    print(f"Predictions: {results}")

if __name__ == "__main__":
    main()
