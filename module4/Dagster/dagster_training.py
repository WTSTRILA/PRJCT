from dagster import (
    job,
    op,
    In,
    Out,
    ResourceDefinition,
    Field,
    String,
    Float,
    get_dagster_logger,
)
import os
import zipfile
import boto3
import tensorflow as tf
from typing import NamedTuple

IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32


def aws_resource_init(init_context):
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key_id or not aws_secret_access_key:
        raise EnvironmentError("AWS credentials are not set in environment variables.")
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


aws_resource = ResourceDefinition(
    resource_fn=aws_resource_init,
    config_schema={},
)


@op(
    name="download_and_extract_s3_zip",
    out={
        "train_data": Out(str),
        "test_data": Out(str),
        "val_data": Out(str),
    },
    required_resource_keys={"aws"},
)

def download_and_extract_s3_zip(context, s3_data_path: str) -> NamedTuple:
    logger = get_dagster_logger()
    s3_client = context.resources.aws

    local_path = '/tmp/bone_data'
    os.makedirs(local_path, exist_ok=True)
    local_zip_path = os.path.join(local_path, 'Bone_Fracture_Binary_Classification.zip')

    try:
        s3_bucket, s3_key = s3_data_path.replace("s3://", "").split("/", 1)
    except ValueError as e:
        logger.error(f"Invalid path S3: {s3_data_path}")
        raise e

    try:
        s3_client.download_file(s3_bucket, s3_key, local_zip_path)
        logger.info(f"Downloaded {s3_data_path} в {local_zip_path}")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        logger.info(f"Unpacked zip file in {local_path}")
    except zipfile.BadZipFile as e:
        logger.error(f"Error unzipping zip file: {e}")
        raise

    train_path = os.path.join(local_path, 'train')
    test_path = os.path.join(local_path, 'test')
    val_path = os.path.join(local_path, 'val')

    for path in [train_path, test_path, val_path]:
        if not os.path.isdir(path):
            logger.error(f"Expected dir not found: {path}")
            raise FileNotFoundError(f"Dir not found: {path}")

    return (train_path, test_path, val_path)


class ModelEvaluation(NamedTuple):
    accuracy: float


@op(
    name="create_train_and_evaluate_model",
    ins={
        "train_data": In(str),
        "val_data": In(str),
        "test_data": In(str),
    },
    out=Out(ModelEvaluation, description="Accuracy on test data"),
    required_resource_keys={"model_storage"},
)

def create_train_and_evaluate_model(context, train_data: str, val_data: str, test_data: str) -> ModelEvaluation:
    logger = get_dagster_logger()

    learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
    epochs = int(os.getenv("EPOCHS", 12))

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        train_data,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_set = val_test_datagen.flow_from_directory(
        val_data,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_set = val_test_datagen.flow_from_directory(
        test_data,
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Start model training...")
    model.fit(training_set, validation_data=val_set, epochs=epochs)
    logger.info("Model training completed.")

    logger.info("Evaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(test_set)
    logger.info(f"Accuracy on test data: {test_accuracy}")

    model_output_path = context.resources.model_storage.save_model(model)
    logger.info(f"model saved to {model_output_path}")

    return ModelEvaluation(test_accuracy)


class ModelStorage:
    def save_model(self, model: tf.keras.Model) -> str:
        model_output_path = "/tmp/bone_model"
        os.makedirs(model_output_path, exist_ok=True)
        model.save(model_output_path)
        return model_output_path


model_storage_resource = ResourceDefinition.hardcoded_resource(ModelStorage())


@job(
    resource_defs={
        "aws": aws_resource,
        "model_storage": model_storage_resource,
    }
)
def bone_fracture_classification_job():
    s3_data_path = os.getenv('S3_DATA_PATH', 's3://bones-pipeline/Bone_Fracture_Binary_Classification.zip')
    paths = download_and_extract_s3_zip(s3_data_path=s3_data_path)
    create_train_and_evaluate_model(
        train_data=paths[0],
        val_data=paths[2],
        test_data=paths[1],
    )
