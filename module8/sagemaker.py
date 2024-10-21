import boto3
from botocore.exceptions import ClientError
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
ROLE_ARN = os.getenv('ROLE_ARN')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_PATH = os.getenv('MODEL_PATH')
IMAGE = os.getenv('IMAGE')


sm_client = boto3.client('sagemaker',
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY,
                         region_name='eu-north-1')


def create_model_container():
    container = {
        "Image": IMAGE,
        "ModelDataUrl": MODEL_PATH,
        "Mode": "MultiModel"
    }
    logger.info("Model container created.")
    return container


def create_sagemaker_model(container):
    try:
        create_model_response = sm_client.create_model(
            ModelName="model-endpoint",
            ExecutionRoleArn=ROLE_ARN,
            PrimaryContainer=container
        )
        logger.info(f"Model successfully created: {create_model_response}")
    except ClientError as e:
        logger.error(f"Error creating model: {e}")


def create_endpoint_config():
    try:
        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName="model-endpoint-config",
            ProductionVariants=[
                {
                    "InstanceType": "ml.g4dn.xlarge",
                    "InitialVariantWeight": 1,
                    "InitialInstanceCount": 1,
                    "ModelName": "model-endpoint",
                    "VariantName": "AllTraffic",
                }
            ],
        )
        logger.info(f"Endpoint configuration successfully created: {create_endpoint_config_response}")
    except ClientError as e:
        logger.error(f"Error creating endpoint configuration: {e}")


def create_endpoint():
    try:
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName="model-endpoint",
            EndpointConfigName="model-endpoint-config"
        )
        logger.info(f"Endpoint successfully created: {create_endpoint_response}")
    except ClientError as e:
        logger.error(f"Error creating endpoint: {e}")


def invoke_sagemaker_endpoint(image_data):
    runtime_sm_client = boto3.client('runtime.sagemaker',
                                     aws_access_key_id=ACCESS_KEY,
                                     aws_secret_access_key=SECRET_KEY,
                                     region_name='eu-north-1')
    try:
        response = runtime_sm_client.invoke_endpoint(
            EndpointName="model-endpoint",
            ContentType="application/octet-stream",
            Body=image_data,
            TargetModel='bone_model.onnx' 
        )
        logger.info(f"Prediction successfully completed: {response['Body'].read()}")
    except ClientError as e:
        logger.error(f"Error sending prediction request: {e}")



if __name__ == "__main__":
    container = create_model_container()
    create_sagemaker_model(container)
    create_endpoint_config()
    create_endpoint()
