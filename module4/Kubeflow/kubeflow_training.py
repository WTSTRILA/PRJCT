from typing import NamedTuple
from kfp.dsl import component, Input, Output, Dataset, Model, pipeline
from kfp import compiler
import os

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials are not set in environment variables.")

s3_data_path = 's3://bones-pipeline/Bone_Fracture_Binary_Classification.zip'


@component(
    base_image="python:3.8",
    packages_to_install=["boto3"]
)
def download_and_extract_s3_zip(
    s3_data_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    output_train_data: Output[Dataset],
    output_test_data: Output[Dataset],
    output_val_data: Output[Dataset]
):
    import os
    import zipfile
    import boto3
    import shutil

    local_zip_path = '/tmp/Bone_Fracture_Binary_Classification.zip'
    extraction_path = '/tmp/bone_data'
    os.makedirs(extraction_path, exist_ok=True)

    s3_bucket, s3_key = s3_data_path.replace("s3://", "").split("/", 1)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        s3_client.download_file(s3_bucket, s3_key, local_zip_path)
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        raise

    try:
        shutil.copytree(os.path.join(extraction_path, 'train'), output_train_data.path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(extraction_path, 'test'), output_test_data.path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(extraction_path, 'val'), output_val_data.path, dirs_exist_ok=True)
    except Exception as e:
        print(f"Error copying data to output paths: {e}")
        raise


@component(
    base_image="python:3.8",
    packages_to_install=["tensorflow"]
)
def create_train_and_evaluate_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    test_data: Input[Dataset],
    learning_rate: float,
    model_output: Output[Model]
) -> NamedTuple('outputs', [
    ('test_accuracy', float)
]):
    import tensorflow as tf
    import os

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        train_data.path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_set = val_test_datagen.flow_from_directory(
        val_data.path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_set = val_test_datagen.flow_from_directory(
        test_data.path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_set, validation_data=val_set, epochs=10)

    test_loss, test_accuracy = model.evaluate(test_set)

    model.save(model_output.path, save_format='tf')
    model_output.metadata = {"model": "bone_fracture_model"}

    return (test_accuracy,)


@pipeline(
    name="bone-fracture-classification-pipeline",
    description="Pipeline for training a bone fracture classification model."
)
def bone_fracture_classification_pipeline(
    s3_data_path: str = s3_data_path,
    learning_rate: float = 0.001,
    aws_access_key_id: str = aws_access_key_id,
    aws_secret_access_key: str = aws_secret_access_key
):
    load_data_task = download_and_extract_s3_zip(
        s3_data_path=s3_data_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    train_and_evaluate_task = create_train_and_evaluate_model(
        train_data=load_data_task.outputs["output_train_data"],
        val_data=load_data_task.outputs["output_val_data"],
        test_data=load_data_task.outputs["output_test_data"],
        learning_rate=learning_rate
    )

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=bone_fracture_classification_pipeline,
        package_path='bone_fracture_classification_pipeline.yaml'
    )
