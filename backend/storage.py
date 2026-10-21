"""Storage abstraction with local filesystem and MinIO implementations."""
import os
import shutil
from pathlib import Path
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def save(self, source_path: str, dest_key: str) -> str:
        pass

    @abstractmethod
    def get_url(self, key: str) -> str:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def get_full_path(self, key: str):
        pass


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str = "./storage"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, source_path: str, dest_key: str) -> str:
        dest = self.base_dir / dest_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        return str(dest)

    def get_url(self, key: str) -> str:
        return f"/files/{key}"

    def delete(self, key: str) -> bool:
        p = self.base_dir / key
        if p.exists():
            p.unlink()
            return True
        return False

    def get_full_path(self, key: str) -> Path:
        return self.base_dir / key


class MinIOStorage(StorageBackend):
    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str = "crack-detection", secure: bool = False):
        from minio import Minio
        self.client = Minio(endpoint, access_key=access_key,
                            secret_key=secret_key, secure=secure)
        self.bucket = bucket
        self.endpoint = endpoint
        self.secure = secure
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
        except Exception as e:
            print(f"MinIO bucket setup warning: {e}")

    def save(self, source_path: str, dest_key: str) -> str:
        self.client.fput_object(self.bucket, dest_key, source_path)
        return dest_key

    def get_url(self, key: str) -> str:
        # Always proxy through backend /files/ endpoint so the browser
        # doesn't need direct access to the MinIO internal hostname
        return f"/files/{key}"

    def delete(self, key: str) -> bool:
        try:
            self.client.remove_object(self.bucket, key)
            return True
        except Exception:
            return False

    def get_full_path(self, key: str):
        return None

    def get_object(self, key: str):
        return self.client.get_object(self.bucket, key)


def get_storage() -> StorageBackend:
    storage_type = os.getenv("STORAGE_TYPE", "local").lower()
    if storage_type == "minio":
        return MinIOStorage(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.getenv("MINIO_BUCKET", "crack-detection"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
    return LocalStorage(os.getenv("STORAGE_DIR", "./storage"))
