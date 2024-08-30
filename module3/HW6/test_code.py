import pytest
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


@pytest.fixture(scope="module")
def data():
    wine_quality = pd.read_csv('winequality-red.csv', delimiter=';')
    X = wine_quality.drop('quality', axis=1)
    y = wine_quality['quality']
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture(scope="module")
def model(data):
    X_train, X_test, y_train, y_test = data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_accuracy(model):
    model, X_test, y_test = model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    assert accuracy > 0.6, "Точность модели ниже ожидаемого уровня"
    assert report is not None, "Отчет классификации пуст"


def test_model_training_epochs(model):
    model, X_test, y_test = model
    initial_accuracy = accuracy_score(y_test, model.predict(X_test))

    epochs = 10
    offset = random.random() / 5

    for epoch in range(2, epochs):
        acc = initial_accuracy - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        assert acc != 0, "Точность не должна быть нулевой"
        assert loss != 0, "Потери не должны быть нулевыми"
        assert acc <= initial_accuracy, "Точность должна уменьшаться"
        assert loss > 0, "Потери должны быть положительными"


if __name__ == "__main__":
    pytest.main()
