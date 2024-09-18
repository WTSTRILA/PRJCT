import io
import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock, patch
from PIL import Image
import wandb
import tensorflow as tf
from fastapi.testclient import TestClient
from api_model import app, initialize_wandb, load_model, preprocess_image, predict_fracture

class TestAppFunctions(TestCase):

    @patch("api_model.initialize_wandb")
    def test_initialize_wandb(self, mock_initialize_wandb):
        mock_initialize_wandb.return_value = MagicMock()
        result = initialize_wandb()
        self.assertIsNotNone(result)
        mock_initialize_wandb.assert_called_once_with()

    @patch("mapi_model.load_model")
    @patch("api_model.initialize_wandb")
    def test_load_model(self, mock_initialize_wandb, mock_load_model):
        mock_run = MagicMock()
        mock_initialize_wandb.return_value = mock_run

        artifact = mock_run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
        artifact.download()

        model_path = artifact.file()
        mock_load_model.return_value = MagicMock()

        model = load_model(mock_run)
        self.assertIsNotNone(model)
        mock_load_model.assert_called_once()

    def test_preprocess_image(self):
        image = Image.new("RGB", (200, 200), color="white")
        result = preprocess_image(image)
        self.assertEqual(result.shape, (1, 180, 180, 3))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))

    @patch("tensorflow.keras.Model.predict")
    def test_predict_fracture(self, mock_predict):
        mock_model = MagicMock()
        mock_predict.return_value = np.array([[0.7]])
        image = Image.new("RGB", (180, 180), color="white")
        result = predict_fracture(mock_model, image)
        self.assertEqual(result, "Fractured")

        mock_predict.return_value = np.array([[0.3]])
        result = predict_fracture(mock_model, image)
        self.assertEqual(result, "Not Fractured")
        mock_predict.assert_called_once()

    @patch("api_model.initialize_wandb")
    @patch("api_model.load_model")
    @patch("api_model.predict_fracture")
    @patch("api_model.Image.open")
    @patch("api_model.io.BytesIO")
    @patch("api_model.File")
    @patch("api_model.JSONResponse")
    def test_predict_bone_fracture(self, mock_JSONResponse, mock_File, mock_BytesIO, mock_Image_open, mock_predict_fracture, mock_load_model, mock_initialize_wandb):
        # Setup mocks
        mock_run = MagicMock()
        mock_initialize_wandb.return_value = mock_run

        artifact = mock_run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
        artifact.download()
        model_path = artifact.file()
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_predict_fracture.return_value = "Fractured"

        image_data = io.BytesIO()
        image = Image.new('RGB', (180, 180), color='white')
        image.save(image_data, format='PNG')
        image_data.seek(0)

        mock_file = MagicMock()
        mock_file.read.return_value = image_data.read()
        mock_File.return_value = mock_file
        mock_Image_open.return_value = image

        client = TestClient(app)

        response = client.post("/predict/", files={"file": ("test_image.png", image_data, "image/png")})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": "Fractured"})

        mock_initialize_wandb.assert_called_once()
        mock_load_model.assert_called_once()
        mock_predict_fracture.assert_called_once_with(mock_model, image)
        mock_run.finish.assert_called_once()

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
