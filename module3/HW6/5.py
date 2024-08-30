import shap
import pandas as pd
import joblib


class QualityData:
    def __init__(self, path):
        self.data = pd.read_csv(path, delimiter=';', quotechar='"')
        self.features = self.data.drop('quality', axis=1)
        self.labels = self.data['quality']

    def get_data(self):
        return self.features, self.labels


class Model:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, inputs):
        scaled_inputs = self.scaler.transform(inputs)
        return self.model.predict(scaled_inputs)

    def explain(self, inputs):
        scaled_inputs = self.scaler.transform(inputs)
        shap_values = self.explainer.shap_values(scaled_inputs)
        return shap_values


def main():
    dataset = QualityData('winequality-red.csv')
    model = Model('random_forest_model.joblib', 'scaler.joblib')

    features, labels = dataset.get_data()
    sample_data = features.sample(100)

    shap_values = model.explain(sample_data)

    shap.summary_plot(shap_values, sample_data, feature_names=sample_data.columns)


if __name__ == "__main__":
    main()