from google.cloud import aiplatform
import os

REGION = os.getenv('REGION')
PROJECT_ID = os.getenv('PROJECT_ID')
BUCKET = os.getenv('BUCKET')
CONTAINER_URI = os.getenv('CONTAINER_URI')
MODEL_SERVING_CONTAINER_IMAGE_URI = os.getenv('MODEL_SERVING_CONTAINER_IMAGE_URI')
DISPLAY_NAME = os.getenv('DISPLAY_NAME')

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)


model = job.run(
    model_display_name=DISPLAY_NAME,
    service_account=""
)

model = aiplatform.Model.upload(
    display_name=DISPLAY_NAME,
    artifact_uri=BUCKET,
    serving_container_image_uri=MODEL_SERVING_CONTAINER_IMAGE_URI
)

endpoint = aiplatform.Endpoint.create(
    display_name=DISPLAY_NAME,
    project=PROJECT_ID,
    location=REGION
)

deployed_model = endpoint.deploy(
    model=model,
    deployed_model_display_name=DISPLAY_NAME,
    traffic_percentage=100 
)

print(f'Model deployed to endpoint: {endpoint.resource_name}')
