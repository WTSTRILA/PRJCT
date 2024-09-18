import streamlit as st
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

def predict_fracture(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return "Fractured" if prediction[0] > 0.5 else "Not Fractured"

def main():
    st.title("Bone Fracture Prediction")
    st.write("This app predicts whether a bone is fractured based on X-ray image input.")

    run = initialize_wandb()
    model = load_model(run)

    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

        result = predict_fracture(model, image)

        st.write(f"The model predicts: {result}")

    run.finish()

if __name__ == "__main__":
    main()
