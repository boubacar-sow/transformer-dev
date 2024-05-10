import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from node2vec import Node2Vec  # type: ignore # pip install node2vec
from omegaconf import DictConfig

from transformer.utils.config import config_to_dict, get_config, hash_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_pickle, save_json, save_matplotlib_fig, save_torch
from transformer.utils.logger import logger

# Load OmegaConf config
_, paths = get_config()
fs = FileSystem()


def get_hash(cfg_node2vec: DictConfig) -> str:
    return "node2vec/" + hash_config(cfg_node2vec)


def compute(cfg_node2vec: DictConfig, use_precomputed: bool = True) -> str:
    cfg_hash = get_hash(cfg_node2vec)

    if fs.exists(paths.embedding_weights_f.format(cfg_hash)) and use_precomputed:
        logger.info(f"Using pre-computed node2vec embedding (dimension: {cfg_node2vec.dim}).")
    else:
        logger.info(f"Computing node2vec embedding (dimension: {cfg_node2vec.dim}).")

        # load graph object from file
        G = load_pickle(paths.PRs_graph_networkx)

        # Initialise
        node2vec = Node2Vec(
            G,
            dimensions=cfg_node2vec.dim,
            walk_length=cfg_node2vec.walk_length,
            num_walks=cfg_node2vec.num_walks,
            q=cfg_node2vec.q,
            p=cfg_node2vec.p,
            seed=cfg_node2vec.seed,
        )
        # Embed
        model = node2vec.fit(window=cfg_node2vec.window, min_count=1, batch_words=4)
        # Obtain embedding
        embedding_matrix = model.wv.vectors
        embedding_id_to_node_id = model.wv.index_to_key
        node_id_to_embedding_id = {int(v): k for k, v in enumerate(embedding_id_to_node_id)}
        embedding_matrix = [embedding_matrix[node_id_to_embedding_id[i]] for i in G.nodes]

        PR_IDs = {v["gaia_id"]: k for k, v in nx.get_node_attributes(G, "properties").items()}
        nodes_inverse_index = {v: k for k, v in enumerate(G.nodes)}
        nodes_to_id = {k: nodes_inverse_index[v] for k, v in PR_IDs.items()}

        # keep only the non-empty keys, and filter embeddings to remove lines with empty keys
        if "" in nodes_to_id:
            del nodes_to_id[""]
        embedding_matrix_pr = np.array([embedding_matrix[value] for value in nodes_to_id.values()])
        nodes_to_id = {key: i for i, key in enumerate(nodes_to_id.keys())}

        # Save embedding
        emb = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix_pr))  # type: ignore
        save_torch(paths.embedding_weights_f.format(cfg_hash), emb.state_dict())
        params = {
            "config": config_to_dict(cfg_node2vec),
            "key_type": "ID Gaia du PR",
        }
        save_json(paths.embedding_config_f.format(cfg_hash), params, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash), nodes_to_id)

        # +------------------- +
        # | Embedding plotting |
        # +------------------- +

        embedding_matrix = np.array(embedding_matrix)

        for i in range(cfg_node2vec.dim):
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
            save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash, "dimension_" + str(i + 1)), fig)
            plt.close(fig)

    return cfg_hash


if __name__ == "__main__":
    cfg, _ = get_config()
    compute(cfg.embeddings.node2vec_embedding)
