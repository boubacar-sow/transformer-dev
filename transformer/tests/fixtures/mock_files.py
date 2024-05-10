r"""
This file mocks the files necessary to train the embeddings,
compute the statistics and train the model.

/!\ Important: the loop over FileSystem objects in prepare_test_buckets
    solves a slightly tricky problem.
    - Situation: In non-test environment, the file folder
        remains the same, and so the FileSystem variable is defined in the header of
        each file where it is used, just like the config dictionary.
    - Problem: with pytest, a fresh new file folder is used for each test,
        but the FileSystem objects are not recreated, so they point toward
        the wrong folder.
    - Solution: before each test, we manually change the local folder of each FileSystem
        object defined in script headers.
"""

import os
import shutil
from pathlib import Path

import torch
from omegaconf import DictConfig

from transformer.utils.config import get_config, hash_config
from transformer.utils.loaders import load_json, save_json, save_torch
from transformer.utils.misc import get_data_hash

cfg, paths = get_config()

FIXTURES_DATA_PATH = os.getcwd() / Path("./transformer/tests/fixtures/data")


def prepare_test_bucket(tmp_path: Path) -> None:
    bucket_path = tmp_path / cfg.data.bucket_name
    new_prefix = str(bucket_path)
    paths_dict = paths.dictconfig
    for key in paths_dict.keys():
        path = paths_dict[key]
        paths_dict[key] = f"{new_prefix}/{path.split('/', 2)[2]}"
    if os.path.isdir(bucket_path):
        shutil.rmtree(bucket_path)  # equivalent to rm -rf [bucket_path]
    Path(bucket_path).mkdir()


def mock_all_files(tmp_path: Path) -> None:
    prepare_test_bucket(tmp_path)

    cfg_emb = cfg.embeddings

    save_json(paths.PRs_doublons, [])

    source_target = [
        ["points_remarquables.csv", paths.points_remarquables],
        ["PRs_graph_nodes.json", paths.PRs_graph_nodes],
        ["PRs_graph_edges.json", paths.PRs_graph_edges],
        ["PRs_graph_networkx.pickle", paths.PRs_graph_networkx],
        ["missing_PRs_neighbors.json", paths.missing_PRs_neighbors_f.format(get_data_hash())],
        ["snapshots/a.json.zstd", paths.snapshot_f.format(cfg.data.days.min_train, "2024-03-08T09:40:00+00:00")],
        ["snapshots/b.json.zstd", paths.snapshot_f.format(cfg.data.days.min_train, "2024-03-08T10:40:00+00:00")],
        ["snapshots/c.json.zstd", paths.snapshot_f.format(cfg.data.days.min_test, "2024-03-08T11:40:00+00:00")],
        ["snapshots/d.json.zstd", paths.snapshot_f.format(cfg.data.days.min_test, "2024-03-08T12:40:00+00:00")],
    ]

    embeddings_to_mock: dict[str, tuple[DictConfig, str]]
    embeddings_to_mock = {
        "geographic": (cfg_emb.geographic_embedding, cfg_emb.geographic_embedding.dim),
        "random": (cfg_emb.random_embedding, cfg_emb.random_embedding.dim),
        "laplacian": (cfg_emb.laplacian_embedding, cfg_emb.laplacian_embedding.dim),
        "node2vec": (cfg_emb.node2vec_embedding, cfg_emb.node2vec_embedding.dim),
        "lignepk": (cfg_emb.lignepk_embedding, cfg_emb.lignepk_embedding.ligne_dim),
        "nextdesserte_pr": (cfg_emb.nextdesserte_embedding, cfg_emb.nextdesserte_embedding.pr_dim),
        "nextdesserte_train": (cfg_emb.nextdesserte_embedding, cfg_emb.nextdesserte_embedding.train_dim),
    }
    for key, (key_cfg, _) in embeddings_to_mock.items():
        cfg_hash = f"{key}/{hash_config(key_cfg)}"
        source_target.append([f"embeddings/{key}_key_to_emb.json", paths.embedding_key_to_emb_f.format(cfg_hash)])

    test_bucket = tmp_path / cfg.data.bucket_name
    for source, target in source_target:
        source_path = FIXTURES_DATA_PATH / source
        target_path = test_bucket / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source_path, target_path)

    for key, (key_cfg, key_dim) in embeddings_to_mock.items():
        cfg_hash = f"{key}/{hash_config(key_cfg)}"
        key_to_emb = load_json(paths.embedding_key_to_emb_f.format(cfg_hash))
        num_embeddings = len(key_to_emb)
        emb: torch.nn.Embedding = torch.nn.Embedding.from_pretrained(torch.randn(num_embeddings, key_dim))  # type: ignore
        save_torch(paths.embedding_weights_f.format(cfg_hash), emb.state_dict())
