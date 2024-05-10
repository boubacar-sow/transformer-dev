from pathlib import Path

from transformer.inputs.embeddings import geographic_embedding, random_embedding
from transformer.inputs.embeddings.embeddings import Embedding, compute_embeddings, load_embeddings
from transformer.tests.fixtures.mock_files import mock_all_files
from transformer.utils.config import get_config

cfg, _ = get_config()


def test_load_embeddings(tmp_path: Path) -> None:
    mock_all_files(tmp_path)

    compute_embeddings()
    embeddings = load_embeddings()  # the embeddings are stored as fixtures and will only be loaded
    emb = embeddings["laplacian_pr"]
    assert len(emb.key_to_emb) == emb.emb.weight.shape[0]


def test_compute_embeddings(tmp_path: Path) -> None:
    mock_all_files(tmp_path)

    cfg_hash = random_embedding.compute(cfg.embeddings.random_embedding, use_precomputed=False)
    emb = Embedding(cfg_hash)

    assert len(emb.key_to_emb) == emb.emb.weight.shape[0]
    assert cfg.embeddings.random_embedding.dim == emb.emb.weight.shape[1]

    cfg_hash = geographic_embedding.compute(cfg.embeddings.geographic_embedding, use_precomputed=False)
    emb = Embedding(cfg_hash)

    assert len(emb.key_to_emb) == emb.emb.weight.shape[0]
    assert cfg.embeddings.geographic_embedding.dim == emb.emb.weight.shape[1]
