from kfp.v2.dsl import component, Input, Output, Dataset, Artifact, pipeline
from kfp.v2 import compiler
import os
from botocore.exceptions import ClientError


aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


if not aws_access_key_id or not aws_secret_access_key:
    raise EnvironmentError("AWS credentials are not set in environment variables.")

s3_data_path = 's3://bones-pipeline/test'
s3_model_path = 's3://bone-model/bone_model.onnx'
s3_output_path = 's3://bone-model/inference_results.csv'


@component(
    base_image='python:3.8'
)
def load_data_for_inference(s3_data_path: str, aws_access_key_id: str, aws_secret_access_key: str, output_data: Output[Dataset]):
    import boto3
    import zipfile
    import os

    local_path = '/tmp/bone_data'
    os.makedirs(local_path, exist_ok=True)

    s3_bucket, s3_key = s3_data_path.replace("s3://", "").split("/", 1)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    local_zip_path = os.path.join(local_path, './test')

    try:
        s3_client.download_file(s3_bucket, s3_key, local_zip_path)
    except ClientError as e:
        print(f"Error downloading file from S3: {e}")
        raise

    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        raise

    output_data.path = local_path


@component(
    base_image='python:3.8' 
)
def load_trained_model(s3_model_path: str, aws_access_key_id: str, aws_secret_access_key: str, model_artifact: Output[Artifact]):
    import boto3
    import os

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
        print(f"Error downloading model from S3: {e}")
        raise

    model_artifact.path = local_model_path


@component(
    base_image='python:3.8',
)
def run_inference(input_data: Input[Dataset], model_artifact: Input[Artifact], inference_results: Output[Dataset]):
    import onnxruntime as ort
    import pandas as pd
    import os
    from PIL import Image
    import numpy as np

    input_data_path = input_data.path
    model_path = model_artifact.path

    test_images_path = os.path.join(input_data_path, 'test_images')
    if not os.path.exists(test_images_path):
        raise FileNotFoundError(f"Test images folder not found at: {test_images_path}")

    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in folder: {test_images_path}")

    try:
        ort_session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        raise

    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    _, channels, height, width = input_shape

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    results = []

    for image_file in image_files:
        image_path = os.path.join(test_images_path, image_file)
        try:
            with Image.open(image_path).convert('RGB') as img:
                img = img.resize((180, 180))  # Resize image to 180x180
                image_array = np.array(img).astype(np.float32) / 255.0
                image_array = (image_array - mean) / std  # Normalize the image
                image_array = np.transpose(image_array, (2, 0, 1))  # Change data format to CHW
                image_array = np.expand_dims(image_array, axis=0).astype(np.float32)  # Add batch dimension

            ort_inputs = {input_name: image_array}
            ort_outs = ort_session.run(None, ort_inputs)

            raw_prediction = ort_outs[0][0][0]
            binary_prediction = int(raw_prediction >= 0.5)

            results.append({
                'image_name': image_file,
                'predicted_label': binary_prediction
            })
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            raise

    results_df = pd.DataFrame(results)

    output_file = os.path.join(inference_results.path, 'inference_results.csv')
    results_df.to_csv(output_file, index=False)


@component(
    base_image='python:3.8',
)
def save_inference_results(inference_results: Input[Dataset], s3_output_path: str, aws_access_key_id: str, aws_secret_access_key: str):
    import boto3
    import os

    s3_bucket, s3_key = s3_output_path.replace("s3://", "").split("/", 1)
    output_file = os.path.join(inference_results.path, 'inference_results.csv')

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        s3_client.upload_file(output_file, s3_bucket, s3_key)
    except ClientError as e:
        print(f"Error uploading results to S3: {e}")
        raise


@pipeline(
    name='inference-pipeline'
)
def inference_pipeline():
    load_data_task = load_data_for_inference(
        s3_data_path=s3_data_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    load_model_task = load_trained_model(
        s3_model_path=s3_model_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    inference_task = run_inference(
        input_data=load_data_task.outputs['output_data'],
        model_artifact=load_model_task.outputs['model_artifact']
    )

    save_results_task = save_inference_results(
        inference_results=inference_task.outputs['inference_results'],
        s3_output_path=s3_output_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path='inference_pipeline.yaml'
    )
