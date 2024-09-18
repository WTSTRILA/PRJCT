from .stream import initialize_wandb, load_model, preprocess_image, predict_fracture
import io
import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock, patch
from PIL import Image
import streamlit as st
import wandb

class TestAppFunctions(TestCase):

    @patch("stream.initialize_wandb")
    def test_initialize_wandb(self, mock_initialize_wandb):
        mock_initialize_wandb.return_value = MagicMock()
        result = initialize_wandb()
        self.assertIsNotNone(result)
        mock_initialize_wandb.assert_called_once_with()

    @patch("stream.load_model")
    @patch("stream.initialize_wandb")
    def test_load_model(self, mock_initialize_wandb, mock_load_model):
        mock_run = MagicMock()
        mock_initialize_wandb.return_value = mock_run

        artifact = mock_run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
        artifact.download()

        model_path = artifact.file()
        mock_load_model.return_value = MagicMock()

        model = load_model(model_path)
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

    @patch("streamlit.file_uploader")
    @patch("streamlit.image")
    @patch("streamlit.write")
    @patch("streamlit.title")
    @patch("streamlit.predict_fracture")
    @patch("streamlit.load_model")
    @patch("streamlit.initialize_wandb")
    def test_main(self, mock_initialize_wandb, mock_load_model, mock_predict_fracture, mock_st_title, mock_st_write,
                  mock_st_image, mock_file_uploader):
        image_data = io.BytesIO()
        image = Image.new('RGB', (180, 180), color='white')
        image.save(image_data, format='PNG')
        image_data.seek(0)

        class MockFile:
            def read(self):
                return image_data.read()

        mock_file = MockFile()
        mock_file_uploader.return_value = mock_file

        mock_run = MagicMock()
        mock_initialize_wandb.return_value = mock_run

        artifact = mock_run.use_artifact('stanislavbochok-/prjct_bones/bone-model:v1', type='model')
        artifact.download()

        model_path = artifact.file()
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_predict_fracture.return_value = "Fractured"

        with patch('streamlit.file_uploader') as mock_file_uploader:
            mock_file_uploader.return_value = MockFile()
            with patch('streamlit.image') as mock_st_image:
                with patch('streamlit.write') as mock_st_write:
                    with patch('streamlit.title') as mock_st_title:
                        main()
                        mock_st_title.assert_called_once_with("Bone Fracture Prediction")
                        mock_file_uploader.assert_called_once_with("Upload an X-ray image", type=["png", "jpg", "jpeg"])
                        mock_st_image.assert_called_once()
                        mock_predict_fracture.assert_called_once_with(mock_model, mock_file_uploader.return_value)
                        mock_st_write.assert_called_with("The model predicts: Fractured")
                        mock_run.finish.assert_called_once()

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
