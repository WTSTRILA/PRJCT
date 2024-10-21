from dagster import job, op, Output, OutputDefinition, In, String, Nothing
import os
import boto3
import zipfile
import pandas as pd
import onnxruntime as ort
from PIL import Image
import numpy as np
from botocore.exceptions import ClientError
from airflow.hooks.base import BaseHook


aws_connection = BaseHook.get_connection('aws_default')
aws_access_key_id = aws_connection.login
aws_secret_access_key = aws_connection.password


if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials are not set in Airflow Connections.")

s3_data_path = 's3://bones-pipeline/test'
s3_model_path = 's3://bone-model/bone_model.onnx'
s3_output_path = 's3://bone-model/inference_results.csv'

@op(required_resource_keys={"s3"})
def load_data_for_inference(context) -> str:
    local_path = '/tmp/bone_data'
    os.makedirs(local_path, exist_ok=True)

    s3_bucket, s3_key = s3_data_path.replace("s3://", "").split("/", 1)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    local_zip_path = os.path.join(local_path, 'test.zip')

    try:
        s3_client.download_file(s3_bucket, s3_key, local_zip_path)
    except ClientError as e:
        context.log.error(f"Error downloading file from S3: {e}")
        raise

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
    except zipfile.BadZipFile as e:
        context.log.error(f"Error extracting zip file: {e}")
        raise
      
    return local_path

@op(required_resource_keys={"s3"})
def load_trained_model(context) -> str:
    s3_bucket, s3_key = s3_model_path.replace("s3://", "").split("/", 1)
    local_model_path = '/tmp/bone_model.onnx'
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        with open(local_model_path, 'wb') as f:
            s3_client.download_fileobj(s3_bucket, s3_key, f)
    except ClientError as e:
        context.log.error(f"Error downloading model from S3: {e}")
        raise

    return local_model_path

@op
def run_inference(context, input_data_path: str, model_path: str) -> pd.DataFrame:
    test_images_path = os.path.join(input_data_path, 'test_images')
    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images folder not found at: {test_images_path}")

    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {test_images_path}")

    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    _, channels, height, width = input_shape

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    results = []

    for image_file in image_files:
        image_path = os.path.join(test_images_path, image_file)
        with Image.open(image_path).convert('RGB') as img:
            img = img.resize((180, 180))
            image_array = np.array(img).astype(np.float32) / 255.0
            image_array = (image_array - mean) / std
            image_array = np.transpose(image_array, (2, 0, 1))
            image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        ort_inputs = {input_name: image_array}
        ort_outs = ort_session.run(None, ort_inputs)

        raw_prediction = ort_outs[0][0][0]
        binary_prediction = int(raw_prediction >= 0.5)

        results.append({
            'image_name': image_file,
            'predicted_label': binary_prediction
        })

    results_df = pd.DataFrame(results)
    return results_df

@op(required_resource_keys={"s3"})
def save_inference_results(context, inference_results: pd.DataFrame) -> None:
    s3_bucket, s3_key = s3_output_path.replace("s3://", "").split("/", 1)
    output_file = '/tmp/inference_results.csv'
    inference_results.to_csv(output_file, index=False)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        s3_client.upload_file(output_file, s3_bucket, s3_key)
    except ClientError as e:
        context.log.error(f"Error uploading results to S3: {e}")
        raise

@job
def inference_pipeline():
    input_data = load_data_for_inference()
    model_path = load_trained_model()
    inference_results = run_inference(input_data, model_path)
    save_inference_results(inference_results)

if __name__ == '__main__':
    from dagster import execute_job
    execute_job(inference_pipeline)
