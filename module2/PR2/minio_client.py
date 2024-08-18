from pathlib import Path
from minio import Minio
import os
import s3fs

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ENDPOINT = "0.0.0.0:9000"


class MinioClientNative:
    def __init__(self, bucket_name: str) -> None:
        client = Minio(
            ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
        )

        self.client = client
        self.bucket_name = bucket_name
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: Path):
        self.client.fput_object(self.bucket_name, file_path.name, file_path)

    def download_file(self, object_name: str, file_path: Path):
        self.client.fget_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=str(file_path),
        )

    def delete_file(self, object_name: str):
        self.client.remove_object(self.bucket_name, object_name)


class MinioClientS3:
    def __init__(self, bucket_name: str) -> None:
        fs = s3fs.S3FileSystem(
            key=ACCESS_KEY,
            secret=SECRET_KEY,
            use_ssl=False,
            client_kwargs={"endpoint_url": f"http://{ENDPOINT}"},
        )

        self.client = fs
        self.bucket_name = bucket_name

        if not self.client.exists(self.bucket_name):
            self.client.mkdir(self.bucket_name)

    def upload_file(self, file_path: Path):
        s3_file_path = f"s3://{self.bucket_name}/{file_path.name}"
        self.client.put(str(file_path), s3_file_path)

    def download_file(self, object_name: str, file_path: Path):
        s3_file_path = f"s3://{self.bucket_name}/{object_name}"
        self.client.download(s3_file_path, str(file_path))

    def delete_file(self, object_name: str):
        s3_file_path = f"s3://{self.bucket_name}/{object_name}"
        self.client.rm(s3_file_path)