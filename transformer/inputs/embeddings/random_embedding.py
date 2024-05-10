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


def get_hash(cfg_random: DictConfig) -> str:
    return "random/" + hash_config(cfg_random)


def compute(cfg_random: DictConfig, use_precomputed: bool = True, make_figures: bool = True) -> str:
    cfg_hash = get_hash(cfg_random)

    if fs.exists(paths.embedding_weights_f.format(cfg_hash)) and use_precomputed:
        logger.info(f"Using pre-computed random embedding (dimension: {cfg_random.dim}).")
    else:
        logger.info(f"Computing random embedding (dimension: {cfg_random.dim}).")

        # +---------------------- +
        # | Embedding computation |
        # +---------------------- +

        # load graph object from file
        G = load_pickle(paths.PRs_graph_networkx)
        PR_IDs = np.array([v["gaia_id"] for v in nx.get_node_attributes(G, "properties").values()])
        nodes_to_id = {t: i for i, t in enumerate(PR_IDs)}
        np.random.seed(cfg_random.seed)
        random_matrix = np.random.randn(nx.number_of_nodes(G), cfg_random.dim) * cfg_random.standard_deviation

        # keep only the non-empty keys, and filter embeddings to remove lines with empty keys
        if "" in nodes_to_id:
            del nodes_to_id[""]
        nodes_list = list(G.nodes)
        pr_nodes = [nodes_list[i] for i in nodes_to_id.values()]  # nodes in the graph that correspond to actual PRs
        random_matrix = np.array([random_matrix[value] for value in nodes_to_id.values()])
        nodes_to_id = {key: i for i, key in enumerate(nodes_to_id.keys())}

        emb = nn.Embedding.from_pretrained(torch.FloatTensor(random_matrix))  # type: ignore
        save_torch(paths.embedding_weights_f.format(cfg_hash), emb.state_dict())
        params = {
            "config": config_to_dict(cfg_random),
            "key_type": "ID Gaia du PR",
        }
        save_json(paths.embedding_config_f.format(cfg_hash), params, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash), nodes_to_id)

        # +------------------- +
        # | Embedding plotting |
        # +------------------- +

        for i in range(cfg_random.dim):
            fig = plt.figure(dpi=300)
            G_pr = nx.induced_subgraph(G, pr_nodes)
            pos = nx.get_node_attributes(G_pr, "geometry")
            nx.draw_networkx(
                G_pr,
                pos=pos,
                with_labels=False,
                node_size=0.005,
                edge_color="black",
                node_color=random_matrix[:, i],
                width=0.05,
            )
            save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash, "dimension_" + str(i + 1)), fig)
            plt.close(fig)

    return cfg_hash


if __name__ == "__main__":
    cfg, _ = get_config()
    compute(cfg.embeddings.random_embedding)
