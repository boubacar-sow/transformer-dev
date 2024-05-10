from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer.tests.fixtures.mock_files import prepare_test_bucket
from transformer.utils.config import get_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import (
    load_csv,
    load_json,
    load_npy,
    load_pickle,
    load_torch,
    load_txt,
    load_txt_l,
    load_txt_part,
    load_txt_per_parts,
    read_txt_part,
    save_csv,
    save_json,
    save_npy,
    save_pickle,
    save_torch,
    save_txt,
    save_txt_l,
)

cfg, _ = get_config()

BASE_PATH = f"{cfg.data.local_folder}/{cfg.data.bucket_name}/"


def test_json(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    d = {"key1": "value1", "key2": [3, 5.0]}
    save_json(BASE_PATH + "test.json", d)
    d2 = load_json(BASE_PATH + "test.json")
    assert d == d2
    save_json(BASE_PATH + "test.json.zst", d, zstd_format=True)
    d3 = load_json(BASE_PATH + "test.json.zst", zstd_format=True)
    assert d == d3


def test_pickle(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    d = {"key1": "value1", "key2": [3, 5.0]}
    save_pickle(BASE_PATH + "test.pkl", d)
    d2 = load_pickle(BASE_PATH + "test.pkl")
    assert d == d2
    save_pickle(BASE_PATH + "test.pkl.zst", d, zstd_format=True)
    d3 = load_pickle(BASE_PATH + "test.pkl.zst", zstd_format=True)
    assert d == d3


def test_npy(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    A = np.array([[0.1, 0.3], [-1 / 3, 10]])
    save_npy(BASE_PATH + "A.npy", A)
    B = load_npy(BASE_PATH + "A.npy")
    assert isinstance(B, np.ndarray)
    assert (A == B).all()
    save_npy(BASE_PATH + "A.npy.zst", A, zstd_format=True)
    C = load_npy(BASE_PATH + "A.npy.zst", zstd_format=True)
    assert (A == C).all()


def test_txt(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    s = "test string\n!"
    save_txt(BASE_PATH + "s.txt", s)
    s2 = load_txt(BASE_PATH + "s.txt")
    assert s == s2
    save_txt(BASE_PATH + "s.txt.zst", s, zstd_format=True)
    s3 = load_txt(BASE_PATH + "s.txt.zst", zstd_format=True)
    assert s == s3


def test_txt_part(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    s = "I am a test string"
    save_txt(BASE_PATH + "s.txt", s)
    fs = FileSystem()
    buffer = fs.read_buffer(BASE_PATH + "s.txt")
    part = read_txt_part(buffer, start=2, end=11)  # type: ignore
    assert part == bytes(s[2:11], encoding="utf-8")

    text_buffer = TextIOWrapper(buffer)
    part = read_txt_part(text_buffer, start=2, end=11)
    assert part == s[2:11]

    part = load_txt_part(BASE_PATH + "s.txt", mode="r", start=0, end=6)
    assert part == s[0:6]

    parts = load_txt_per_parts(BASE_PATH + "s.txt", mode="r", chunksize=4, start=None, end=None)
    parts = list(parts)
    assert parts == [s[:4], s[4:8], s[8:12], s[12:16], s[16:]]


def test_txt_l(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    s = "12, 12.5, 13, 14"
    save_txt(BASE_PATH + "s.txt", s)

    parts = load_txt_l(BASE_PATH + "s.txt", sep=", ")
    assert parts == ["12", "12.5", "13", "14"]

    parts = load_txt_l(BASE_PATH + "s.txt", sep=", ", dtype=float)
    assert parts == [12, 12.5, 13, 14]

    a = np.arange(5)
    save_txt_l(BASE_PATH + "s.txt", a, sep=", ")
    s = load_txt(BASE_PATH + "s.txt")
    assert s == "0, 1, 2, 3, 4"


def test_csv(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    A = np.eye(10) / 2
    df = pd.DataFrame(A, columns=[str(i) for i in range(10)])
    save_csv(BASE_PATH + "df.csv", df)
    df2 = load_csv(BASE_PATH + "df.csv")
    assert df.equals(df2)
    save_csv(BASE_PATH + "df.csv.zst", df, zstd_format=True)
    df3 = load_csv(BASE_PATH + "df.csv.zst", zstd_format=True)
    assert df.equals(df3)


def test_torch(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)
    A = torch.randn(20, 20)
    save_torch(BASE_PATH + "A.pt", A)
    B = load_torch(BASE_PATH + "A.pt")
    assert isinstance(B, torch.Tensor)
    assert (A == B).all()  # type: ignore
