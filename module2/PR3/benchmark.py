import pandas as pd
import numpy as np
import time
import os
import typer
from typing import Callable

def create_dataframe(n_rows: int = 100_000, n_cols: int = 10) -> pd.DataFrame:
    np.random.seed(0)
    data = np.random.rand(n_rows, n_cols)
    columns = [f'col_{i}' for i in range(n_cols)]
    df = pd.DataFrame(data, columns=columns)
    return df

def save_csv(df: pd.DataFrame, file_name: str):
    df.to_csv(file_name, index=False)

def load_csv(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def save_parquet(df: pd.DataFrame, file_name: str):
    df.to_parquet(file_name)

def load_parquet(file_name: str) -> pd.DataFrame:
    return pd.read_parquet(file_name)

def save_pickle(df: pd.DataFrame, file_name: str):
    df.to_pickle(file_name)

def load_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle(file_name)

def save_feather(df: pd.DataFrame, file_name: str):
    df.to_feather(file_name)

def load_feather(file_name: str) -> pd.DataFrame:
    return pd.read_feather(file_name)

def save_npy(df: pd.DataFrame, file_name: str):
    np.save(file_name, df.to_numpy())

def load_npy(file_name: str) -> pd.DataFrame:
    data = np.load(file_name)
    return pd.DataFrame(data)

def save_hdf5(df: pd.DataFrame, file_name: str):
    df.to_hdf(file_name, key='df', mode='w')

def load_hdf5(file_name: str) -> pd.DataFrame:
    return pd.read_hdf(file_name, key='df')

def save_nc4(df: pd.DataFrame, file_name: str):
    from netCDF4 import Dataset
    data = df.to_numpy()
    with Dataset(file_name, 'w', format='NETCDF4') as nc_file:
        nc_file.createDimension('dim0', data.shape[0])
        nc_file.createDimension('dim1', data.shape[1])
        nc_var = nc_file.createVariable('data', 'f4', ('dim0', 'dim1'))
        nc_var[:] = data

def load_nc4(file_name: str) -> pd.DataFrame:
    from netCDF4 import Dataset
    with Dataset(file_name, 'r') as nc_file:
        data = nc_file.variables['data'][:]
    return pd.DataFrame(data)

def save_json(df: pd.DataFrame, file_name: str):
    df.to_json(file_name)

def load_json(file_name: str) -> pd.DataFrame:
    return pd.read_json(file_name)

def benchmark_format(df: pd.DataFrame, file_name: str, save_func: Callable, load_func: Callable):
    start_time = time.monotonic()
    save_func(df, file_name)
    save_time = time.monotonic() - start_time

    start_time = time.monotonic()
    df_loaded = load_func(file_name)
    load_time = time.monotonic() - start_time

    return save_time, load_time

def run_benchmark(format_type: str, file_name: str):
    df = create_dataframe()

    save_func, load_func = {
        'csv': (save_csv, load_csv),
        'parquet': (save_parquet, load_parquet),
        'pickle': (save_pickle, load_pickle),
        'feather': (save_feather, load_feather),
        'npy': (save_npy, load_npy),
        'hdf5': (save_hdf5, load_hdf5),
        'nc4': (save_nc4, load_nc4),
        'json': (save_json, load_json)
    }.get(format_type, (None, None))

    if save_func is None or load_func is None:
        raise ValueError("Не поддерживаемый формат")

    save_time, load_time = benchmark_format(df, file_name, save_func, load_func)

    print(f"Формат: {format_type}")
    print(f"  Время сохранения: {save_time:.2f} секунд")
    print(f"  Время загрузки: {load_time:.2f} секунд")

def run_all_benchmarks(inference_size: int):
    formats = ['csv', 'parquet', 'pickle', 'feather', 'npy', 'hdf5', 'nc4', 'json']

    for fmt in formats:
        file_name = f'benchmark_data.{fmt}'
        run_benchmark(fmt, file_name)
        os.remove(file_name)

def cli_app():
    app = typer.Typer()
    app.command()(run_all_benchmarks)
    app()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            inference_size = int(sys.argv[1])
        except ValueError:
            print("Неверный формат аргумента. Используйте целое число.")
            sys.exit(1)
    else:
        inference_size = 100_000  

    run_all_benchmarks(inference_size)
