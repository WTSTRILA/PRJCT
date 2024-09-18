import gradio as gr
import wandb
import tensorflow as tf
import numpy as np
from PIL import Image
import os

def initialize_wandb():
    return wandb.init(project="prjct_bones", job_type="predict")

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

def predict_fracture(image):
    run = initialize_wandb()
    model = load_model(run)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = "Not Fractured" if prediction[0] > 0.5 else "Fractured"
    run.finish()
    return result

def main():
    iface = gr.Interface(
        fn=predict_fracture,
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Bone Fracture Prediction",
        description="This app predicts whether a bone is fractured based on X-ray image input."
    )
    iface.launch(share=True)
if __name__ == "__main__":
    main()
