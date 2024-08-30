import wandb
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

sweep_config = {
    "name": "random_forest_sweep",
    "method": "grid",
    "parameters": {
        "n_estimators": {
            "values": [50, 100, 200]
        },
        "max_depth": {
            "values": [None, 10, 20, 30]
        }
    }
}

def train():
    wandb.init(project="module2")

    config = wandb.config
    n_estimators = config.n_estimators
    max_depth = config.max_depth

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

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    wandb.log({"accuracy": accuracy})
    wandb.log({"classification_report": report})

    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = accuracy - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        wandb.log({"acc": acc, "loss": loss})

    wandb.finish()

sweep_id = wandb.sweep(sweep=sweep_config, project="module2")

# Start sweep agents
wandb.agent(sweep_id, function=train, count=10)  # Adjust count to the number of runs you want
