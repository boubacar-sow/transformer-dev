import os
from pathlib import Path
from typing import Any, Optional

import boto3

from transformer.utils.config import get_config, hash_config
from transformer.utils.logger import logger
from transformer.utils.times import get_train_validation_test_days

cfg, paths = get_config()


def copy_folder_to_s3(path: str) -> None:
    command = "aws s3 cp --recursive "
    command += f"{path} s3://{cfg.data.bucket_name}/{_shorten_path(path)}"
    logger.info(f"Running command: {command}")
    assert os.system(command) == 0  # code 0 = command ran succesfully


def download_folder_from_s3(
    bucket: Any,
    s3_folder: str,
    local_parent: Optional[str] = None,
) -> None:
    """
    Download the contents of a folder directory
    Args:
        bucket: an s3 bucket opened with s3 = resource("s3"); bucket = s3.Bucket(bucket_name)
        s3_folder: the folder path in the s3 bucket (can also be a single file name)
        local_parent: a relative or absolute directory path in the local file system where
                      the s3_folder copy will be placed

    Source: https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3
    """
    local_path = Path(local_parent)  # type: ignore
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_parent is None else local_path / s3_folder / os.path.relpath(obj.key, s3_folder)
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


def _shorten_path(path: str) -> str:
    base_directory = f"{cfg.data.local_folder}/{cfg.data.bucket_name}"
    assert path.startswith(base_directory), f"The target folder must be located in {base_directory}."
    return os.path.relpath(path=path, start=base_directory)


def download_training_data_from_s3() -> None:
    """
    Download training data from AWS S3 storage.

    The snapshots are downloaded with "aws s3 sync", this avoids re-downloading them at every training.
    """
    session = boto3.session.Session(profile_name=cfg.data.aws_profile)
    s3 = session.resource("s3")
    bucket = s3.Bucket(cfg.data.bucket_name)

    base_folder = f"{cfg.data.local_folder}/{cfg.data.bucket_name}"

    download_folder_from_s3(bucket, _shorten_path(paths.pr_data), local_parent=base_folder)
    download_folder_from_s3(bucket, _shorten_path(paths.snapshots_stats), local_parent=base_folder)
    download_folder_from_s3(bucket, _shorten_path(paths.optuna_storage), local_parent=base_folder)

    days_train, days_test = get_train_validation_test_days()
    train_months = {day.rsplit("-", 1)[0] for day in days_train + days_test}
    snapshots_folder = paths.snapshot_f.format(list(train_months)[0], "").rsplit("/", 2)[0]

    command = "aws "
    if cfg.data.aws_profile:
        command += f"--profile {cfg.data.aws_profile} "
    command += 's3 sync --only-show-errors --exclude="*" '
    for month in train_months:
        command += f'--include="{month}*" '
    command += f"s3://{cfg.data.bucket_name}/{_shorten_path(snapshots_folder)} {snapshots_folder}"
    logger.info(f"Running command: {command}")
    assert os.system(command) == 0  # code 0 = command ran succesfully

    embeddings_keys = {
        "laplacian": cfg.embeddings.laplacian_embedding,
        "node2vec": cfg.embeddings.node2vec_embedding,
        "random": cfg.embeddings.random_embedding,
        "geographic": cfg.embeddings.geographic_embedding,
        "nextdesserte_train": cfg.embeddings.nextdesserte_embedding,
        "nextdesserte_pr": cfg.embeddings.nextdesserte_embedding,
        "lignepk": cfg.embeddings.lignepk_embedding,
    }
    for key, value in embeddings_keys.items():
        emb_code = f"{key}/{hash_config(value)}"
        download_folder_from_s3(
            bucket, _shorten_path(paths.embedding_weights_f.format(emb_code)), local_parent=base_folder
        )
        download_folder_from_s3(
            bucket, _shorten_path(paths.embedding_key_to_emb_f.format(emb_code)), local_parent=base_folder
        )
