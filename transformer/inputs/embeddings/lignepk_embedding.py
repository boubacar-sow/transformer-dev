from multiprocessing import Pool
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from networkx.algorithms.shortest_paths.generic import shortest_path  # type: ignore
from omegaconf import DictConfig

from transformer.inputs.resources.pr_infos import compute_pr_infos
from transformer.inputs.resources.train_type import get_train_type_str
from transformer.inputs.snapshots.statistics import load_statistics
from transformer.utils.config import config_to_dict, get_config, hash_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import save_json, save_matplotlib_fig, save_torch
from transformer.utils.logger import logger
from transformer.utils.times import days_between

# Embedding of the ligne number of a pr, using a small neural network taking as
# entry 2 line embeddings and 2 point kilometriques
# (therefore representing 2 PR) and trying to predict the path duration (in time)
# and path length (in number of PR) of the minimal
# journey between the 2 PR.


# Load OmegaConf config
_, paths = get_config()
fs = FileSystem()


def makeBatch(args: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Auxiliary function used in compute_lignepk_embedding. Given a seed, generates a random batch.
    """
    seed, cfg_lignepk, prs, G, code_ligne_to_id, pr_to_lignepk, inter_pr_mean, inter_pr_std, pk_mean, pk_std = args

    seed = cfg_lignepk.seed + seed
    gen = torch.random.manual_seed(seed)
    randIds = torch.randint(
        len(prs),
        (
            2,
            cfg_lignepk.batch_size,
        ),
        generator=gen,
    )

    randFrom = [prs[i] for i in randIds[0]]
    randTo = [prs[i] for i in randIds[1]]

    graph_paths = [shortest_path(G, prFrom, prTo, weight="weight") for prFrom, prTo in zip(randFrom, randTo)]
    pathDurations = [
        nx.path_weight(G, p, "weight") for p in graph_paths
    ]  # distance between the two PR on their shortest path in minutes
    pathLengths = [len(p) for p in graph_paths]  # number of PRs on said shortest path, odd choice tbh

    x_ligne = torch.tensor(
        [
            (
                code_ligne_to_id[pr_to_lignepk.get(prFrom, pr_to_lignepk["other"])[0]],
                code_ligne_to_id[pr_to_lignepk.get(prTo, pr_to_lignepk["other"])[0]],
            )
            for prFrom, prTo in zip(randFrom, randTo)
        ]
    )
    x_pk = torch.tensor(
        [
            (
                pr_to_lignepk.get(prFrom, pr_to_lignepk["other"])[1],
                pr_to_lignepk.get(prTo, pr_to_lignepk["other"])[1],
            )
            for prFrom, prTo in zip(randFrom, randTo)
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(list(zip(pathLengths, pathDurations)), dtype=torch.float)
    y -= inter_pr_mean
    y /= inter_pr_std
    # but the scale of the two feature to learn is not the same, this cannot be good. and path length is odd!!!!
    x_pk -= pk_mean
    x_pk /= pk_std
    return x_ligne, x_pk, y


def get_hash(cfg_lignepk: DictConfig) -> str:
    return "lignepk/" + hash_config(cfg_lignepk)


def compute(cfg_lignepk: DictConfig, use_precomputed: bool = True) -> str:  # noqa: C901
    cfg_hash = get_hash(cfg_lignepk)

    if fs.exists(paths.embedding_weights_f.format(cfg_hash)) and use_precomputed:
        logger.info(f"Using pre-computed lignepk embedding (dimension: {cfg_lignepk.ligne_dim} (+ 1 for pk).")
    else:
        logger.info(f"Computing lignepk embedding (dimension: {cfg_lignepk.ligne_dim} (+ 1 for pk).")

        lignepk_days_train = days_between(
            cfg_lignepk.min_date,
            cfg_lignepk.max_date,
        )
        lignepk_days_test = days_between(
            cfg_lignepk.min_test_date,
            cfg_lignepk.max_test_date,
        )
        lignepk_days_train = sorted((d for d in lignepk_days_train if (d not in lignepk_days_test)))
        # lignepk_days = sorted(lignepk_days_train + lignepk_days_test)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Working on device: {DEVICE}")

        # +------------- +
        # | Get the data |
        # +------------- +

        pr_info = compute_pr_infos()

        logger.info("Computing number of occurences of each train number...")
        # Get frequent train_nums, beware! this is where data from each day file is used
        all_stats = load_statistics()
        train_num_nbrs = all_stats.train_num
        if isinstance(cfg_lignepk.min_train_num_nbr, int):
            train_nums = sorted(t for t, n in train_num_nbrs.items() if n >= cfg_lignepk.min_train_num_nbr)
        else:
            train_nums = sorted(
                t for t, n in train_num_nbrs.items() if n >= cfg_lignepk.min_train_num_nbr * len(lignepk_days_train)
            )

        # Filter the trains using some criteria, here, remove the trains that are RER
        def train_num_filter(train_num: str) -> bool:
            return "_rer" not in get_train_type_str(train_num)

        train_nums = [t for t in train_nums if train_num_filter(t)]

        # Returns a dict, with keys: two PR and values: an histogram of theoretic time in the sillons
        # between the two consecutive PRs
        logger.info("Computing clearing times distributions...")
        inter_pr_times = all_stats.inter_pr.clearing_times
        inter_pr_counts = all_stats.inter_pr.counts
        # get_inter_pr_clearing_times(
        #     lignepk_days_train,
        #     sillon_tagger=sillon_tagger,
        #     sillon_filter=sillon_filter,
        #     scope="local",
        #     obs_key="theo_sec",
        # )

        # For symmetry, solves the directional problem by fusing prA to prB and prB to prA
        inter_pr_symmetric: dict[tuple[str, ...], float] = {}
        for k, v in inter_pr_times.items():
            sorted_key = tuple(sorted(k))
            if sorted_key in inter_pr_symmetric:
                inter_pr_symmetric[sorted_key] = (inter_pr_symmetric[sorted_key] + v.mean) / 2
            else:
                inter_pr_symmetric[sorted_key] = v.mean
        # Replaces the histogram by a median, over tuples of PRs that were frequently seen,
        # reducing the size of the file
        inter_pr_times1 = {k: v for k, v in inter_pr_symmetric.items() if inter_pr_counts[k] >= cfg_lignepk.threshold1}
        inter_pr_times2 = {k: v for k, v in inter_pr_symmetric.items() if inter_pr_counts[k] >= cfg_lignepk.threshold2}
        del inter_pr_times
        # threshold independent du nombre de jours, etrange

        # Beware! this is also where we use data! to obtain the ligne code and point kilometrique
        def None_to_0(x: Optional[float]) -> float:
            return x if x is not None else 0

        pr_to_lignepk = {}
        for pr_id, pr in pr_info.items():
            code_ligne, pk = pr.get_single_code_ligne_pk()
            pr_to_lignepk[pr_id] = (None_to_0(code_ligne), None_to_0(pk))

        # +---------------------+
        # | Build the PRs graph |
        # +---------------------+

        logger.info("Building the PRs graph from clearing times...")
        G = nx.Graph()

        for k in inter_pr_times1:  # Build edges using frequently observed tuples of adjacent PRs
            if k[0] != k[1]:
                G.add_edge(*k)
        gnodes = set(G.nodes)

        # Add edges from less frequently observed tuples of PRS if it connects two precendtly unconnected components
        for k in inter_pr_times2:
            if k[0] != k[1]:
                if (k[0] not in gnodes) or (k[1] not in gnodes):
                    G.add_edge(*k)
                else:
                    try:
                        _ = shortest_path(G, k[0], k[1])
                    except nx.NetworkXNoPath:
                        G.add_edge(*k)

        # Defines the weights on the edges as the median theoretic time of travel between them
        for prFrom, prTo in G.edges:
            s = 0.0
            n = 0
            if (prFrom, prTo) in inter_pr_times2:
                s += inter_pr_times2[(prFrom, prTo)]
                n += 1
            if (prTo, prFrom) in inter_pr_times2:
                s += inter_pr_times2[(prTo, prFrom)]
                n += 1
            assert n > 0, f"{(prFrom, prTo)} not found."
            assert n == 1, f"{(prFrom, prTo)} not found."
            G[prFrom][prTo]["weight"] = s / n

        # Only keeping the largest connected component, only valid if the other are extremely small
        if not nx.is_connected(G):
            conns = list(nx.connected_components(G))
            lens = list(map(len, conns))
            logger.info(f"Reducing from {len(conns)} connected components with lengths {str(lens)}.")
            i = np.argmax(lens)
            G = G.subgraph(conns[i])
            assert nx.is_connected(G)

        prs = list(G.nodes)
        logger.info(f"PR coverage = {len(prs)}")
        logger.info(str(["postArrival" in prs, "other" in prs, "preDeparture" in prs]))  # TODO find a solution for that

        # +------------------------+
        # | Prepare data for model |
        # +------------------------+

        pr_to_lignepk = {
            k: v for k, v in pr_to_lignepk.items() if k in G.nodes
        }  # smaller dict with only PRs that are in the graph
        pr_to_lignepk["other"] = (
            0,
            0,
        )  # relevant for PRs in the graph but for which we don't know the ligne code and pk
        pr_to_lignepk["preDeparture"] = (1, 0)  # inutile aussi?
        pr_to_lignepk["postArrival"] = (2, 0)  # douteux!
        # an embedding should be available for all prs, even those we've never seen
        pr_to_lignepk = {
            k: (v[0], (0 if str(v[1]) == "nan" else v[1])) for k, v in pr_to_lignepk.items()
        }  # si pas de pk, le remplace par 0
        code_ligne_to_id = {
            li: i for i, li in enumerate(sorted({v[0] for v in pr_to_lignepk.values()}))
        }  # donne des numeros arbitraires au lignes

        inter_pr_mean = np.mean(
            list(inter_pr_times1.values())
        )  # faire attention! temps de trajet entre deux pr consecutifs?
        inter_pr_std = np.std(list(inter_pr_times1.values()))
        pk_mean = np.mean([xi[1] for xi in pr_to_lignepk.values()])
        pk_std = np.std([xi[1] for xi in pr_to_lignepk.values()])

        # +--------------------------------------------+
        # | We define the embedding model and train it |
        # +--------------------------------------------+

        lEmb = nn.Embedding(len(code_ligne_to_id), cfg_lignepk.ligne_dim).to(device=DEVICE)

        model = nn.Sequential(
            nn.Linear(2 * cfg_lignepk.ligne_dim + 2, cfg_lignepk.model_dim),
            nn.PReLU(),
            nn.Linear(cfg_lignepk.model_dim, 2),
            nn.PReLU(),
        ).to(device=DEVICE)
        optim = torch.optim.Adam(set(model.parameters()) | set(lEmb.parameters()), lr=1e-3)

        # Training of the neural network, whose first hidden layer is used as embedding of the two lignes
        # To stop it use ctrl+C, thus launching
        # the plotting of the training procedure
        logger.info("Starting embeddings training...")
        metrics = {"loss_i": [], "loss_t": []}  # type: dict[str, list[float]]
        with Pool(cfg_lignepk.pool) as p:
            args = [
                (
                    seed,
                    cfg_lignepk,
                    prs,
                    G,
                    code_ligne_to_id,
                    pr_to_lignepk,
                    inter_pr_mean,
                    inter_pr_std,
                    pk_mean,
                    pk_std,
                )
                for seed in range(cfg_lignepk.n_iter)
            ]  # type: list[Any]
            batches = p.imap_unordered(makeBatch, args)
            for i, (x_ligne, x_pk, y) in enumerate(batches):  # type: ignore
                optim.zero_grad()
                x_ligne = x_ligne.to(device=DEVICE)
                x_pk = x_pk.to(device=DEVICE)
                out = model(
                    torch.cat(
                        (
                            lEmb(x_ligne).view(
                                cfg_lignepk.batch_size,
                                2 * cfg_lignepk.ligne_dim,
                            ),
                            x_pk,
                        ),
                        axis=1,
                    )  # type: ignore
                )
                y = y.to(device=DEVICE)
                loss = (out - y).abs().mean(dim=0)
                # loss = ((out - y)**2).mean(dim=0) change loss?
                loss_bis = (loss.detach().cpu() * inter_pr_std).numpy()
                # loss_i = loss.detach().cpu().numpy()
                loss_length, loss_duration = loss_bis[0], loss_bis[1]
                metrics["loss_i"].append(loss_length)
                metrics["loss_t"].append(loss_duration)
                logger.info(
                    "{:<7}   {:<10} {:<7}".format(i, round(float(loss_length), 3), round(float(loss_duration), 3))
                )
                loss = loss.sum()
                loss.backward()
                optim.step()

                if i % 30 == 0:
                    logger.info("Saving to disk...")
                    save_torch(paths.embedding_weights_f.format(cfg_hash), lEmb.state_dict())
                    # if cfg.embeddings.save_training_data:
                    #     save_torch(paths.lignepk_emb_model_f.format(cfg_hash), model.state_dict())
                    #     save_pickle(paths.lignepk_emb_losses_f.format(cfg_hash), metrics)

        params = {
            "config": config_to_dict(cfg_lignepk),
            "pr_to_lignepk": pr_to_lignepk,
            "key_type": "code ligne",
        }
        save_json(paths.embedding_config_f.format(cfg_hash), params, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash), code_ligne_to_id)

        # +--------------------------------+
        # | Plotting of training procedure |
        # +--------------------------------+

        step = 1
        logger.info("Making figure...")
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(metrics["loss_i"][::step], "b" + "-", label="loss itineraire")
        ax.legend()
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(metrics["loss_t"][::step], "r" + "-", label="loss temps")
        ax.legend()
        fig.suptitle("Loss evolution during training of lignepk embedding")
        save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash, "loss"), fig)
        plt.close(fig)

    return cfg_hash


if __name__ == "__main__":
    cfg, _ = get_config()
    compute(cfg.embeddings.lignepk_embedding)
