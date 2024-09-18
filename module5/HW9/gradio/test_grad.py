import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import tensorflow as tf
from grad import initialize_wandb, load_model, preprocess_image, predict_fracture

class TestBoneFracturePrediction(unittest.TestCase):

    @patch('grad.initialize_wandb')
    @patch('grad.load_model')
    def test_predict_fracture(self, mock_load_model, mock_initialize_wandb):
        mock_run = MagicMock()
        mock_initialize_wandb.return_value = mock_run

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.2]])
        mock_load_model.return_value = mock_model

        test_image = Image.new('RGB', (180, 180))

        result = predict_fracture(test_image)

        self.assertEqual(result, "Fractured")

        mock_model.predict.assert_called_once()
        mock_run.finish.assert_called_once()

    def test_preprocess_image(self):
        test_image = Image.new('RGB', (200, 200))
        img_array = preprocess_image(test_image, 180, 180)

        self.assertEqual(img_array.shape, (1, 180, 180, 3))

        self.assertTrue(np.all(img_array >= 0) and np.all(img_array <= 1))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

