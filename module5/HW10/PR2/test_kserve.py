import unittest
import base64
import pytest
from unittest.mock import patch, MagicMock
import tensorflow as tf
from kserve_api import CustomModel


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = tf.constant([[0.9]])
    return model


@pytest.fixture
def custom_model(mock_model):
    with patch('tensorflow.keras.models.load_model', return_value=mock_model):
        model = CustomModel("custom-model")
        model.base_model_predictor.loaded_model = mock_model
        model.ready = True
        return model


def test_load(custom_model):
    custom_model.load('/mock/path/to/model')
    assert custom_model.ready
    custom_model.base_model_predictor.loaded_model.load_model.assert_called_with('/mock/path/to/model', compile=False)


def test_predict(custom_model):
     request = {
        "instances": [
            {
                "image": {
                    "b64": "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHwAAAABJRU5ErkJggg=="
                }
            }
        ]
    }
    with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
        prediction = custom_model.predict(request)
        assert prediction == {"predictions": ["fractured"]}
