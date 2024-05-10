from typing import Optional

import torch
import torch.nn as nn

from transformer.inputs.embeddings import (
    geographic_embedding,
    laplacian_embedding,
    lignepk_embedding,
    nextdesserte_embedding,
    node2vec_embedding,
    random_embedding,
)
from transformer.utils.config import get_config
from transformer.utils.loaders import load_json, load_torch
from transformer.utils.logger import logger
from transformer.utils.misc import get_data_hash, torch_device
from transformer.utils.s3 import copy_folder_to_s3

cfg, paths = get_config()


class Embedding:
    """
    Embeddings of PRs or train numbers or code lignes
    Example usage:
        emb = Embedding("some_very_long_hash", is_pr_embedding=True, normalize_embedding=True)
        emb.shape -> (num_embeddings, embeddings_dim)
        emb["foo"] -> tensor([1., 2., 3., 4.])                                          (floats)
        emb[["foo", "bar"]] -> tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])             (floats)
        emb.get_indices(["foo", "bar"]) -> tensor([1, 34])                              (integers)
        emb.get_values(tensor([1, 34])) -> tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]) (floats)
    """

    emb: nn.Embedding
    key_to_emb: dict[str, int]

    shape: torch.Size
    update_during_training: bool

    # Only defined for PR embeddings
    is_pr_embedding: bool
    emb_missing_prs: nn.Embedding
    key_to_emb_missing_prs: dict[str, int]
    prs_doublons: list[str]
    missing_prs_neighbors: dict[str, list[tuple[float, tuple[str, float], tuple[str, float]]]]

    def __init__(
        self,
        embedding_hash: str,
        is_pr_embedding: bool = False,
        normalize_embedding: Optional[bool] = None,
        update_during_training: bool = False,
    ) -> None:
        """
        Args:
        - embedding_hash: obtained from the embeddings computation functions
        - is_pr_embedding: True if the embeddings represent Points Remarquables, False otherwise (for ligne/train_num)
        - normalize_embedding: if not None, overrides cfg.data.preprocessing.normalize_embeddings
                               if True, center + normalize the embeddings at initialization
        """
        self.embedding_hash = embedding_hash
        self.is_pr_embedding = is_pr_embedding
        self.update_during_training = update_during_training
        if normalize_embedding is not None:
            self.normalize_embedding = normalize_embedding
        else:
            self.normalize_embedding = cfg.data.preprocessing.normalize_embeddings
        if self.is_pr_embedding:
            self.prs_doublons = load_json(paths.PRs_doublons)
            self.missing_prs_neighbors = load_json(paths.missing_PRs_neighbors_f.format(get_data_hash()))
            self.key_to_emb_missing_prs = {}
        self.load()
        if self.is_pr_embedding:
            self.update_missing_prs()

    def load(self) -> None:
        """Loads the embeddings weights from disk, along with the related key_to_emb dictionnary."""
        self.key_to_emb = load_json(paths.embedding_key_to_emb_f.format(self.embedding_hash))
        emb_data = load_torch(paths.embedding_weights_f.format(self.embedding_hash), map_location="cpu")
        self.shape = emb_data["weight"].shape
        assert len(self.key_to_emb) == self.shape[0], "The embedding weight and the index map have different lengths!"
        self.emb = nn.Embedding(*self.shape)
        self.emb.load_state_dict(emb_data)
        self.emb.requires_grad_(self.update_during_training)
        self.emb = self.emb.to(torch_device)
        if self.normalize_embedding:
            self.normalize()

    def get_indices(self, str_indices: list[str]) -> torch.Tensor:
        """
        Args:
            str_indices: list of keys to be retrieved later from the embedding. For example, it can be a list of PR IDs.

        Returns:
            A torch list of integers, giving the index of each key in str_indices in the embeddings matrix.
            Missing keys are mapped to -1.

        IMPORTANT: in the case of PR embeddings, the indices above self.emb.num_embedding refer to embeddings
            of missing PRs. The embedding index of missing PRs is obtained with (idx - self.emb.num_embedding).
            These indices refer to lines of self.emb_missing_prs (instead of self.emb for regular PRs).
        """
        if self.is_pr_embedding:
            int_indices = []
            for pr_id in str_indices:
                if pr_id in ["preDeparture", "postArrival"]:
                    int_indices.append(-1)
                elif (pr_id not in self.key_to_emb) or (pr_id in self.prs_doublons):
                    missing_index = self.key_to_emb_missing_prs.get(pr_id, 0)
                    int_indices.append(missing_index + self.emb.num_embeddings)
                else:
                    int_indices.append(self.key_to_emb[pr_id])
            return torch.Tensor(int_indices).to(dtype=int)  # type: ignore[no-any-return]
        else:
            return torch.Tensor([self.key_to_emb.get(str(t), -1) for t in str_indices]).to(  # type: ignore[no-any-return]
                dtype=int
            )

    def get_values(self, int_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            int_indices: a torch.Tensor with arbitrary shape (a, ..., z) referring embedding indices.

        Returns:
            a torch.Tensor with shape (a, ..., z, embedding_dimension) containing the embeddings. Missing
            keys (-1) are mapped to a null embedding.

        IMPORTANT: if self.is_pr_embedding is True, the indices above self.emb.num_embedding will be treated
            as indices of self.emb_missing_prs.
        """
        int_indices = int_indices.to(torch_device)

        # Compute which indices have no corresponding embedding
        zeros_mask = int_indices == -1
        int_indices[zeros_mask] = 0
        if self.is_pr_embedding:  # Compute which indices are missing PRs to be interpolated
            interpolate_mask = int_indices >= self.emb.num_embeddings
            missing_prs_indices = int_indices[interpolate_mask] - self.emb.num_embeddings
            int_indices[interpolate_mask] = 0

        # Retrieve the embeddings
        emb_value = self.emb(int_indices)

        emb_value[zeros_mask] = 0

        # Retrieve the embeddings of missing PRs
        if self.is_pr_embedding:
            emb_value[interpolate_mask] = self.emb_missing_prs(missing_prs_indices)

        return emb_value  # type: ignore[no-any-return]

    def __getitem__(self, idx: str | list[str]) -> torch.Tensor:
        is_str = isinstance(idx, str)
        if isinstance(idx, str):
            idx = [idx]
        emb = self.get_values(self.get_indices(idx).to(torch_device))
        if is_str:
            emb = emb[0]
        return emb

    def update_missing_prs(self) -> None:
        assert self.is_pr_embedding

        self.key_to_emb_missing_prs = {pr_id: i for i, pr_id in enumerate(self.missing_prs_neighbors.keys())}
        self.emb_missing_prs = nn.Embedding(
            num_embeddings=len(self.key_to_emb_missing_prs),
            embedding_dim=self.emb.embedding_dim,
        )
        self.emb_missing_prs.requires_grad_(False)
        self.emb_missing_prs = self.emb_missing_prs.to(torch_device)

        for missing_pr_id, pr_neighbors in self.missing_prs_neighbors.items():
            idx = self.key_to_emb_missing_prs[missing_pr_id]
            if len(pr_neighbors) == 0:
                self.emb_missing_prs.weight[idx] = 0
            else:
                emb_values = []
                for curr_time, prev_neighbor, next_neighbor in pr_neighbors:
                    id_1, time_1 = prev_neighbor
                    id_2, time_2 = next_neighbor
                    emb_1, emb_2 = self.get_values(self.get_indices([id_1, id_2]))
                    if abs(time_1 - time_2) < 0.1:
                        emb_value = emb_1
                    else:
                        emb_value = emb_1 + ((curr_time - time_1) / (time_2 - time_1)) * (emb_2 - emb_1)
                    emb_values.append(emb_value)

                self.emb_missing_prs.weight[idx] = torch.vstack(emb_values).mean(dim=0)

    def normalize(self) -> None:
        self.emb.weight.data -= self.emb.weight.data.mean(axis=0)[None, :]
        self.emb.weight.data /= self.emb.weight.data.std(axis=0)[None, :]


def compute_embeddings() -> None:
    """
    If needed, computes the required embeddings
    """
    logger.info(f"Computing embeddings (using precomputed if exists: {cfg.training.use_precomputed.embeddings}).")

    # Computed each embedding if it is used by the model
    use_precomputed = cfg.training.use_precomputed.embeddings
    cfg_hashes = []  # type: list[str]
    if cfg.model.use_inputs.geographic_embedding:
        cfg_hash = geographic_embedding.compute(cfg.embeddings.geographic_embedding, use_precomputed)
        cfg_hashes.append(cfg_hash)
    if cfg.model.use_inputs.laplacian_embedding:
        cfg_hash = laplacian_embedding.compute(cfg.embeddings.laplacian_embedding, use_precomputed)
        cfg_hashes.append(cfg_hash)
    if cfg.model.use_inputs.node2vec_embedding:
        cfg_hash = node2vec_embedding.compute(cfg.embeddings.node2vec_embedding, use_precomputed)
        cfg_hashes.append(cfg_hash)
    if cfg.model.use_inputs.random_embedding:
        cfg_hash = random_embedding.compute(cfg.embeddings.random_embedding, use_precomputed)
        cfg_hashes.append(cfg_hash)
    if cfg.model.use_inputs.lignepk_embedding:
        cfg_hash = lignepk_embedding.compute(cfg.embeddings.lignepk_embedding, use_precomputed)
        cfg_hashes.append(cfg_hash)
    if cfg.model.use_inputs.train_nextdesserte_embedding or cfg.model.use_inputs.pr_nextdesserte_embedding:
        cfg_hash_ = nextdesserte_embedding.compute(cfg.embeddings.nextdesserte_embedding, use_precomputed)
        cfg_hashes += list(cfg_hash_)  # nextdesserte_embedding produces two embeddings, hence cfg_hash is a tuple

    # Save the result to S3
    if cfg.data.save_results_to_s3:
        logger.info("Saving computed embeddings to S3.")
        for cfg_hash in cfg_hashes:
            copy_folder_to_s3(paths.embedding_folder_f.format(cfg_hash))


def load_embeddings() -> dict[str, Embedding]:
    logger.info("Loading embeddings.")
    embeddings = {}
    if cfg.model.use_inputs.geographic_embedding:
        cfg_hash = geographic_embedding.get_hash(cfg.embeddings.geographic_embedding)
        learn = cfg.embeddings.geographic_embedding.update_during_training
        embeddings["geographic_pr"] = Embedding(cfg_hash, is_pr_embedding=True, update_during_training=learn)
    if cfg.model.use_inputs.laplacian_embedding:
        cfg_hash = laplacian_embedding.get_hash(cfg.embeddings.laplacian_embedding)
        learn = cfg.embeddings.laplacian_embedding.update_during_training
        embeddings["laplacian_pr"] = Embedding(cfg_hash, is_pr_embedding=True, update_during_training=learn)
    if cfg.model.use_inputs.node2vec_embedding:
        cfg_hash = node2vec_embedding.get_hash(cfg.embeddings.node2vec_embedding)
        learn = cfg.embeddings.node2vec_embedding.update_during_training
        embeddings["node2vec_pr"] = Embedding(cfg_hash, is_pr_embedding=True, update_during_training=learn)
    if cfg.model.use_inputs.random_embedding:
        cfg_hash = random_embedding.get_hash(cfg.embeddings.random_embedding)
        learn = cfg.embeddings.random_embedding.update_during_training
        embeddings["random_pr"] = Embedding(cfg_hash, is_pr_embedding=True, update_during_training=learn)
    if cfg.model.use_inputs.lignepk_embedding:
        cfg_hash = lignepk_embedding.get_hash(cfg.embeddings.lignepk_embedding)
        learn = cfg.embeddings.lignepk_embedding.update_during_training
        embeddings["lignepk"] = Embedding(cfg_hash, is_pr_embedding=False, update_during_training=learn)
    if cfg.model.use_inputs.train_nextdesserte_embedding or cfg.model.use_inputs.pr_nextdesserte_embedding:
        cfg_hash_train, cfg_hash_pr = nextdesserte_embedding.get_hash(cfg.embeddings.nextdesserte_embedding)
        learn = cfg.embeddings.nextdesserte_embedding.update_during_training
        if cfg.model.use_inputs.train_nextdesserte_embedding:
            embeddings["nextdesserte_train"] = Embedding(
                cfg_hash_train, is_pr_embedding=False, update_during_training=learn
            )
        if cfg.model.use_inputs.pr_nextdesserte_embedding:
            embeddings["nextdesserte_pr"] = Embedding(cfg_hash_pr, is_pr_embedding=True, update_during_training=learn)

    return embeddings
