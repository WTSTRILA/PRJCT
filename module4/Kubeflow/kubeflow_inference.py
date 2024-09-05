import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output

IMAGE = "docker.io/stasbochok/winequality-training-pipeline:latest"


@dsl.component(base_image=IMAGE)
def load_data(test_data: Output[Dataset]):
    import shutil
    from pathlib import Path

    from classic_example.data import load_sst2_data

    load_sst2_data(Path("/app/data"))

    shutil.move(Path("/app/data") / "test.csv", test_data.path)


@dsl.component(base_image=IMAGE)
def load_model(
    model: Output[Model],
    scaler: Output[Artifact],
):
    import shutil
    from pathlib import Path

    model_dir = Path("/tmp/model")
    model_dir.mkdir(exist_ok=True)

    shutil.copy("/tmp/model/random_forest_model.joblib", model.path)
    shutil.copy("/tmp/model/scaler.joblib", scaler.path)


@dsl.component(base_image=IMAGE)
def run_inference(
    model: Input[Model],
    scaler: Input[Artifact],
    test_data: Input[Dataset],
    pred: Output[Dataset],
):
    import pandas as pd
    from joblib import load
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from pathlib import Path

    model_path = Path("/tmp/model")
    model_path.mkdir(exist_ok=True)

    pipeline = load(model.path)
    scaler = load(scaler.path)

    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop(columns=["quality"])

    test_df["predictions"] = pipeline.predict(X_test)

    test_df.to_csv(pred.path, index=False)


@dsl.pipeline
def inference_pipeline():
    load_data_task = load_data()

    load_model_task = load_model()

    run_inference_task = run_inference(
        model=load_model_task.outputs["model"],
        scaler=load_model_task.outputs["scaler"],
        test_data=load_data_task.outputs["test_data"],
    )


def compile_pipeline() -> str:
    path = "/tmp/inference_pipeline.yaml"
    kfp.compiler.Compiler().compile(inference_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment")
    _ = client.create_experiment("inference", namespace=namespace)

    print("Uploading pipeline")
    name = "winequality-red-inference"
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
