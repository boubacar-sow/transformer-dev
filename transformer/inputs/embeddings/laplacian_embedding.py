import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import scipy as scp  # type: ignore
import torch
import torch.nn as nn
from omegaconf import DictConfig

from transformer.utils.config import config_to_dict, get_config, hash_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_pickle, save_json, save_matplotlib_fig, save_torch
from transformer.utils.logger import logger

# Load OmegaConf config
_, paths = get_config()
fs = FileSystem()


def get_hash(cfg_laplacian: DictConfig) -> str:
    return "laplacian/" + hash_config(cfg_laplacian)


def compute(cfg_laplacian: DictConfig, use_precomputed: bool = True) -> str:
    cfg_hash = get_hash(cfg_laplacian)

    if fs.exists(paths.embedding_weights_f.format(cfg_hash)) and use_precomputed:
        logger.info(f"Using pre-computed laplacian embedding (dimension: {cfg_laplacian.dim}).")
    else:
        logger.info(f"Computing laplacian embedding (dimension: {cfg_laplacian.dim}).")

        # load graph object from file
        G = load_pickle(paths.PRs_graph_networkx)
        PR_IDs = np.array([v["gaia_id"] for v in nx.get_node_attributes(G, "properties").values()])
        nodes_to_id = {t: i for i, t in enumerate(PR_IDs)}

        use_weights = None
        # TODO maybe remove it since it does not run anyway
        if cfg_laplacian.use_weights:
            use_weights = "weight"
        L = nx.laplacian_matrix(G, weight=use_weights).astype(float)
        if cfg_laplacian.version == "CLASSIC":
            if use_weights == "weight":
                L_temp = nx.laplacian_matrix(G).astype(float)
                eigenvalues_temp, eigenvectors_temp = scp.sparse.linalg.eigsh(
                    L_temp, which="SM", k=cfg_laplacian.dim + 1
                )
                eigenvalues, eigenvectors = scp.sparse.linalg.lobpcg(
                    L,
                    eigenvectors_temp,
                    largest=False,
                )
            else:
                eigenvalues, eigenvectors = scp.sparse.linalg.eigsh(L, which="SM", k=cfg_laplacian.dim + 1)
            embedding_matrix = eigenvectors[:, 1 : cfg_laplacian.dim + 1]
        elif cfg_laplacian.version == "CLASSIC_RESCALED":
            eigenvalues, eigenvectors = scp.sparse.linalg.eigsh(L, which="SM", k=cfg_laplacian.dim + 1)
            embedding_matrix = eigenvectors[:, 1 : cfg_laplacian.dim + 1] / np.sqrt(
                eigenvalues[1 : cfg_laplacian.dim + 1]
            )
        elif cfg_laplacian.version == "GENERAL_EIGENVALUE":
            D = L + nx.adjacency_matrix(G, weight=use_weights).astype(float)
            eigenvalues, eigenvectors = scp.sparse.linalg.eigsh(L, which="SM", M=D, k=cfg_laplacian.dim + 1)
            embedding_matrix = eigenvectors[:, 1 : cfg_laplacian.dim + 1]
        elif cfg_laplacian.version == "TRANSITION":
            D = L + nx.adjacency_matrix(G, weight=use_weights).astype(float)
            eigenvalues, eigenvectors = scp.sparse.linalg.eigsh(L, which="SM", M=D, k=cfg_laplacian.dim + 1)
            embedding_matrix = np.dot(eigenvectors, np.identity(cfg_laplacian.dim + 1) - eigenvalues)[
                :, 1 : cfg_laplacian.dim + 1
            ]

        # keep only the non-empty keys, and filter embeddings to remove lines with empty keys
        if "" in nodes_to_id:
            del nodes_to_id[""]
        embedding_matrix_pr = np.array([embedding_matrix[value] for value in nodes_to_id.values()])
        nodes_to_id = {key: i for i, key in enumerate(nodes_to_id.keys())}

        # saving
        emb = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix_pr))  # type: ignore
        save_torch(paths.embedding_weights_f.format(cfg_hash), emb.state_dict())
        params = {
            "config": config_to_dict(cfg_laplacian),
            "key_type": "ID Gaia du PR",
        }
        save_json(paths.embedding_config_f.format(cfg_hash), params, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash), nodes_to_id)

        # +------------------- +
        # | Embedding plotting |
        # +------------------- +

        fig = plt.figure(dpi=300)
        plt.scatter(embedding_matrix[:, 0], embedding_matrix[:, 1], s=1)
        plt.title("Embedding 2D des PRs")
        save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash, "2D_embedding"), fig)
        plt.close(fig)

        for i in range(cfg_laplacian.dim):
            fig = plt.figure(dpi=300)
            nx.draw_networkx(
                G,
                pos=nx.get_node_attributes(G, "geometry"),
                with_labels=False,
                node_size=0.005,
                edge_color="black",
                node_color=embedding_matrix[:, i],
                width=0.05,
            )
            save_matplotlib_fig(
                paths.embedding_figures_f.format(cfg_hash, "dimension_" + str(i + 1)),
                fig,
            )
            plt.close(fig)

    return cfg_hash


if __name__ == "__main__":
    cfg, _ = get_config()
    compute(cfg.embeddings.laplacian_embedding)
