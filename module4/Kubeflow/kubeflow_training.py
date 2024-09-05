import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output


IMAGE = "docker.io/stasbochok/winequality-training-pipeline:latest"

@dsl.component(base_image=IMAGE)
def load_data(
    train_data: Output[Dataset], test_data: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    df = pd.read_csv("winequality-red.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)


@dsl.component(base_image=IMAGE)
def train_model(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    scaler: Output[Artifact],
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from joblib import dump
    from pathlib import Path

    train_df = pd.read_csv(train_data.path)
    test_df = pd.read_csv(test_data.path)

    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"]
    X_test = test_df.drop(columns=["quality"])
    y_test = test_df["quality"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    dump(pipeline, model.path)
    dump(pipeline.named_steps["scaler"], scaler.path)


@dsl.component(base_image=IMAGE)
def save_model(
    model: Input[Model],
    scaler: Input[Artifact],
):
    import shutil
    from pathlib import Path

    model_dir = Path("/tmp/model")
    model_dir.mkdir(exist_ok=True)

    shutil.copy(model.path, model_dir / "random_forest_model.joblib")
    shutil.copy(scaler.path, model_dir / "scaler.joblib")

    print(f"Model and scaler saved to {model_dir}")


@dsl.pipeline
def training_pipeline():
    load_data_task = load_data()

    train_model_task = train_model(
        train_data=load_data_task.outputs["train_data"],
        test_data=load_data_task.outputs["test_data"],
    )

    save_model_task = save_model(
        model=train_model_task.outputs["model"],
        scaler=train_model_task.outputs["scaler"],
    )


def compile_pipeline() -> str:
    path = "/tmp/training_pipeline.yaml"
    kfp.compiler.Compiler().compile(training_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline")
    name = "winequality-red-training"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version.")
        pipeline_prev_version = client.get_pipeline(client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.pipeline_id,
        )
    else:
        pipeline = client.upload_pipeline(
            pipeline_package_path=compile_pipeline(), pipeline_name=name
        )
    print(f"pipeline {pipeline.pipeline_id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)
