import numpy as np
from sklearn.dummy import DummyClassifier
import concurrent.futures
from typing import Tuple
import time
from tqdm import tqdm
import typer
import ray
from dask.distributed import Client


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_train, y_train)
    return dummy_clf

def get_data(
    inference_size: int = 100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.random.rand(100)
    y_train = np.random.rand(100)
    x_test = np.random.rand(inference_size, 100)
    return x_train, y_train, x_test

def predict(model: DummyClassifier, x: np.ndarray) -> np.ndarray:
    time.sleep(0.002)
    return model.predict(x)

def run_inference(
    model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    y_pred = []
    for i in tqdm(range(0, x_test.shape[0], batch_size)):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)

def run_inference_process_pool(
    model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16
) -> np.ndarray:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = len(x_test) // max_workers
        chunks = [x_test[i : i + chunk_size] for i in range(0, len(x_test), chunk_size)]
        futures = [executor.submit(run_inference, model=model, x_test=chunk) for chunk in chunks]
        concurrent.futures.wait(futures)
        y_pred = [future.result() for future in futures]
    return np.concatenate(y_pred)

@ray.remote
def run_inference_ray(
    model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    y_pred = []
    for i in range(0, x_test.shape[0], batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)

def run_inference_ray_main(
    model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16
) -> np.ndarray:
    chunk_size = len(x_test) // max_workers
    chunks = [x_test[i : i + chunk_size] for i in range(0, len(x_test), chunk_size)]
    futures = [run_inference_ray.remote(model, chunk) for chunk in chunks]
    y_pred = ray.get(futures)
    return np.concatenate(y_pred)

def run_inference_dask(
    model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    y_pred = []
    for i in range(0, x_test.shape[0], batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)

def run_inference_dask_main(
    client, model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16
) -> np.ndarray:
    chunk_size = len(x_test) // max_workers
    chunks = [x_test[i : i + chunk_size] for i in range(0, len(x_test), chunk_size)]
    futures = [client.submit(run_inference_dask, model, chunk) for chunk in chunks]
    y_pred = client.gather(futures)
    return np.concatenate(y_pred)

def benchmark_inference(inference_size: int, max_workers: int):
    x_train, y_train, x_test = get_data(inference_size=inference_size)
    model = train_model(x_train, y_train)

    start_time = time.monotonic()
    run_inference(model=model, x_test=x_test)
    single_worker_time = time.monotonic() - start_time

    start_time = time.monotonic()
    run_inference_process_pool(model=model, x_test=x_test, max_workers=max_workers)
    pool_time = time.monotonic() - start_time

    ray.init(ignore_reinit_error=True)
    start_time = time.monotonic()
    run_inference_ray_main(model=model, x_test=x_test, max_workers=max_workers)
    ray_time = time.monotonic() - start_time
    ray.shutdown()

    client = Client()
    start_time = time.monotonic()
    run_inference_dask_main(client=client, model=model, x_test=x_test, max_workers=max_workers)
    dask_time = time.monotonic() - start_time
    client.close()

    print(f"Single worker time: {single_worker_time:.2f} seconds")
    print(f"Process pool time with {max_workers} workers: {pool_time:.2f} seconds")
    print(f"Ray time with {max_workers} workers: {ray_time:.2f} seconds")
    print(f"Dask time with {max_workers} workers: {dask_time:.2f} seconds")

def cli_app():
    app = typer.Typer()
    app.command()(lambda inference_size: benchmark_inference(int(inference_size), 16))
    app()

if __name__ == "__main__":
    cli_app()
