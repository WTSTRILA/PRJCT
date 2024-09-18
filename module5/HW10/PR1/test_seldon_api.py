import unittest
from unittest.mock import patch, MagicMock
from seldon_api import BoneConditionPredictor


class TestBoneConditionPredictor(unittest.TestCase):

    @patch('seldon_api.BaseModelPredictor.__init__', return_value=None)
    @patch('seldon_api.tf.keras.models.load_model')
    def setUp(self, mock_load_model, mock_base_init):
        self.predictor = BoneConditionPredictor()
        self.mock_model = MagicMock()
        self.predictor.loaded_model = self.mock_model
        self.predictor.class_names = ['fractured', 'not fractured']
        self.mock_model.predict.return_value = [[0.7]]

    @patch('seldon_api.BoneConditionPredictor.load_preprocessed_image')
    def test_predict(self, mock_load_preprocessed_image):
        mock_load_preprocessed_image.return_value = 'mock_image_array'
        result = self.predictor.predict('test_image.jpg')
        self.mock_model.predict.assert_called_once_with('mock_image_array')
        self.assertEqual(result, 'fractured')


if __name__ == '__main__':
    unittest.main()
