import grpc
from concurrent import futures
import bone_fracture_pb2
import bone_fracture_pb2_grpc
import wandb
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import signal


def initialize_wandb():
    return wandb.init(project="prjct_bones", job_type="predict", reinit=True)


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


class BoneFractureService(bone_fracture_pb2_grpc.BoneFractureServiceServicer):
    def PredictFracture(self, request, context):
        try:
            image = Image.open(io.BytesIO(request.image))
        except Exception as e:
            context.set_details("Invalid image format")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return bone_fracture_pb2.PredictFractureResponse(prediction="Error")

        run = initialize_wandb()
        model = load_model(run)
        result = predict_fracture(model, image)
        run.finish()

        return bone_fracture_pb2.PredictFractureResponse(prediction=result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bone_fracture_pb2_grpc.add_BoneFractureServiceServicer_to_server(BoneFractureService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC сервер запущен на порту 50051")

    def handle_shutdown(signum, frame):
        print("Завершение работы сервера...")
        server.stop(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
