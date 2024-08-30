import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

@pytest.fixture(scope="module")
def wine_quality_data() -> pd.DataFrame:
    data_path = Path("winequality-red.csv")
    return pd.read_csv(data_path, delimiter=';', quotechar='"')

def test_data_columns(wine_quality_data: pd.DataFrame):
    assert 'quality' in wine_quality_data.columns, "The dataset does not contain the 'quality' column"

def test_train_test_split(wine_quality_data: pd.DataFrame):
    X = wine_quality_data.drop('quality', axis=1)
    y = wine_quality_data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    assert X_train.shape[0] > 0, "Training set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"
    assert len(y_train) == X_train.shape[0], "Mismatch between X_train and y_train"
    assert len(y_test) == X_test.shape[0], "Mismatch between X_test and y_test"

def test_model_training(wine_quality_data: pd.DataFrame):
    X = wine_quality_data.drop('quality', axis=1)
    y = wine_quality_data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"
    assert 'accuracy' in report, "Classification report does not contain accuracy"

def test_model_output_shape(wine_quality_data: pd.DataFrame):
    X = wine_quality_data.drop('quality', axis=1)
    y = wine_quality_data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, "Output shape does not match"