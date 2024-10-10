import os
import wandb
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow_model_optimization import tensorflow_model_optimization as tfmot

data_dir = Path('./data')

def initialize_wandb():
    return wandb.init(project="prjct_bones", job_type="predict", reinit=True)

def load_model():
    run = initialize_wandb()
    artifact = run.use_artifact('stanislavbochok-/prjct_bones/quantized_bone_model:v0', type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'quantized_model.tflite')

    original_model = tf.keras.models.load_model(model_path)

    if not isinstance(original_model, (tf.keras.models.Sequential, tf.keras.Model)):
        raise ValueError("Загруженная модель должна быть либо Sequential, либо функциональной моделью.")

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5,
        begin_step=2000, end_step=4000)

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(original_model, pruning_schedule=pruning_schedule)

    pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return pruned_model

def quantize_model(pruned_model):
    if not isinstance(pruned_model, (tf.keras.Sequential, tf.keras.Model)):
        raise ValueError("Модель должна быть Sequential или функциональной.")

    image_paths = [str(p) for p in data_dir.glob('*.jpg')]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [180, 180]) / 255.0
        return img

    dataset = dataset.map(load_and_preprocess_image).batch(1)

    def representative_dataset_gen():
        for image_batch in dataset.take(100):
            yield [image_batch]

    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    quantized_model = converter.convert()

    quantized_model_path = "quantized_model.tflite"
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_model)
    print(f"Квантованная модель сохранена по пути: {quantized_model_path}")

    return quantized_model_path

def upload_model_to_wandb(model_path):
    artifact = wandb.Artifact("quantized_bone_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Загружен {model_path} в W&B.")
    wandb.finish()

def main():
    run = initialize_wandb()

    model = load_model()

    quantized_model_path = quantize_model(model)

    upload_model_to_wandb(quantized_model_path)

    interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


if __name__ == "__main__":
    main()
