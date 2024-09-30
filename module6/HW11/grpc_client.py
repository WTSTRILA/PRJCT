import grpc
import bone_fracture_pb2
import bone_fracture_pb2_grpc
from PIL import Image
import io


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = bone_fracture_pb2_grpc.BoneFractureServiceStub(channel)
        image_path = '3.jpg'

        try:
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_byte_array = io.BytesIO()
                img.save(img_byte_array, format='JPEG')
                image_bytes = img_byte_array.getvalue()

            request = bone_fracture_pb2.PredictFractureRequest(image=image_bytes)
            response = stub.PredictFracture(request)
            print(f"Prediction: {response.prediction}")

        except grpc.RpcError as e:
            print(f"gRPC ошибка: {e.code()} - {e.details()}")
        except FileNotFoundError:
            print(f"Ошибка: файл изображения '{image_path}' не найден")


if __name__ == "__main__":
    run()
