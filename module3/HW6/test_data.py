import pytest
import pandas as pd
from great_expectations.dataset.pandas_dataset import PandasDataset
from typing import Tuple


@pytest.fixture
def data() -> Tuple[PandasDataset, PandasDataset, PandasDataset]:
    df = pd.read_csv('winequality-red.csv', sep=';')

    df_train = df.iloc[:1100]
    df_val = df.iloc[1100:1300]
    df_test = df.iloc[1300:1500]

    return PandasDataset(df_train), PandasDataset(df_val), PandasDataset(df_test)


def test_data_shape(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data
    for dataset in [df_train, df_val, df_test]:
        for column in dataset.columns:
            assert dataset.expect_column_values_to_not_be_null(column=column)["success"]


def test_data_order(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data
    column_list = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    ]

    for dataset in [df_train, df_val, df_test]:
        assert dataset.expect_table_columns_to_match_ordered_list(column_list)["success"]


def test_data_content(data: Tuple[PandasDataset, PandasDataset, PandasDataset]):
    df_train, df_val, df_test = data

    for dataset in [df_train, df_val, df_test]:
        for column in dataset.columns:
            assert dataset.expect_column_values_to_not_be_null(column=column)["success"]

    for dataset in [df_train, df_val, df_test]:
        assert dataset.expect_column_values_to_be_of_type(column="fixed acidity", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="volatile acidity", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="citric acid", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="residual sugar", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="chlorides", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="free sulfur dioxide", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="total sulfur dioxide", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="density", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="pH", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="sulphates", type_="float64")["success"]
        assert dataset.expect_column_values_to_be_of_type(column="alcohol", type_="float64")["success"]
