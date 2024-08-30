from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import random
import pandas as pd
import nni


def get_parameters():
    """Fetch hyperparameters from NNI."""
    params = nni.get_next_parameter()
    return params


def main():
    # Get hyperparameters from NNI
    params = get_parameters()

    # Load dataset
    wine_quality = pd.read_csv('winequality-red.csv', delimiter=';', quotechar='"')

    print("Columns in dataset:", wine_quality.columns)

    if 'quality' not in wine_quality.columns:
        raise ValueError("The dataset does not contain the 'quality' column")

    X = wine_quality.drop('quality', axis=1)
    y = wine_quality['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:\n", report)

    # Simulate epochs and losses (if needed)
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = accuracy - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        # Simulate logging if needed
        # print(f"Epoch {epoch}: Acc: {acc}, Loss: {loss}")

    # Report final result to NNI
    nni.report_final_result(accuracy)


if __name__ == "__main__":
    main()
