import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
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


def get_hash(cfg_geographic: DictConfig) -> str:
    return "geographic/" + hash_config(cfg_geographic)


def compute(cfg_geographic: DictConfig, use_precomputed: bool = True) -> str:
    cfg_hash = get_hash(cfg_geographic)

    if fs.exists(paths.embedding_weights_f.format(cfg_hash)) and use_precomputed:
        logger.info("Using pre-computed geographic embedding.")
    else:
        logger.info("Computing geographic embedding.")

        # load graph object from file
        G = load_pickle(paths.PRs_graph_networkx)
        coordinates = np.array(list(nx.get_node_attributes(G, "geometry").values()))
        PR_IDs = np.array([v["gaia_id"] for v in nx.get_node_attributes(G, "properties").values()])
        nodes_to_id = {t: i for i, t in enumerate(PR_IDs)}

        # keep only the non-empty keys, and filter coordinates to remove lines with empty keys
        if "" in nodes_to_id:
            del nodes_to_id[""]

        coordinates = np.array([coordinates[value] for value in nodes_to_id.values()])
        nodes_to_id = {key: i for i, key in enumerate(nodes_to_id.keys())}

        emb = nn.Embedding.from_pretrained(torch.FloatTensor(coordinates))  # type: ignore
        save_torch(paths.embedding_weights_f.format(cfg_hash), emb.state_dict())
        params = {
            "config": config_to_dict(cfg_geographic),
            "key_type": "ID Gaia du PR",
        }
        save_json(paths.embedding_config_f.format(cfg_hash), params, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash), nodes_to_id)

        fig = plt.figure(dpi=300)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], s=1)
        plt.title("Coordonnées géographiques des PRs")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash, "fig"), fig)
        plt.close(fig)

    return cfg_hash


if __name__ == "__main__":
    cfg, _ = get_config()
    cfg_geographic = cfg.embeddings.geographic_embedding
    compute(cfg_geographic)
