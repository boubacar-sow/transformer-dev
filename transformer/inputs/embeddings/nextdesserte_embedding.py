import random
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.inputs.data_structures import SillonChunk
from transformer.inputs.snapshots.snapshots_files import RawSnapshotDataset, get_snapshots_files
from transformer.inputs.snapshots.statistics import load_statistics
from transformer.inputs.utils.sillons_filters import SillonFilter, SillonFilterTrainNums
from transformer.inputs.utils.train_num_filters import TrainNumFilter, TrainNumFilterNotTrainType
from transformer.utils.config import config_to_dict, get_config, hash_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_pickle, save_json, save_matplotlib_fig, save_pickle, save_torch
from transformer.utils.logger import logger
from transformer.utils.times import days_between

# computes both a train number embedding and a pr embedding using a neural net with input a pair of one-hot encode
# train and pr, and trying to predict a
# one hot encoded pr, using a neural net with 2 hidden layers (the first one giving the embeddings)


# Load OmegaConf config
cfg, paths = get_config()
fs = FileSystem()


def get_hash(cfg_nextdesserte: DictConfig) -> tuple[str, str]:
    cfg_hash_train = "nextdesserte_train/" + hash_config(cfg_nextdesserte)
    cfg_hash_pr = "nextdesserte_pr/" + hash_config(cfg_nextdesserte)
    return cfg_hash_train, cfg_hash_pr


def compute(cfg_nextdesserte: DictConfig, use_precomputed: bool = True) -> tuple[str, str]:  # noqa: C901
    cfg_hash_train, cfg_hash_pr = get_hash(cfg_nextdesserte)

    if (
        fs.exists(paths.embedding_weights_f.format(cfg_hash_train))
        and fs.exists(paths.embedding_weights_f.format(cfg_hash_pr))
        and use_precomputed
    ):
        logger.info(
            f"Using pre-computed nextdesserte embedding (dimensions: {cfg_nextdesserte.pr_dim}, {cfg_nextdesserte.train_dim})."  # noqa: E501
        )
    else:
        logger.info(
            f"Computing nextdesserte embedding (dimensions: {cfg_nextdesserte.pr_dim}, {cfg_nextdesserte.train_dim})."
        )

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Working on device: {DEVICE}")

        nextdesserte_days_test = days_between(cfg_nextdesserte.min_test_date, cfg_nextdesserte.max_test_date)
        nextdesserte_days_train = days_between(cfg_nextdesserte.min_date, cfg_nextdesserte.max_date)
        nextdesserte_days_train = sorted((d for d in nextdesserte_days_train if (d not in nextdesserte_days_test)))
        # nextdesserte_days = sorted(nextdesserte_days_train + nextdesserte_days_test)

        all_stats = load_statistics()

        def glob(prefix: str) -> list[str]:
            """
            Example: prefix = "path/to/file/test_"
            Could return: ["path/to/file/test_1.txt", "path/to/file/test_2.txt"]
            """
            folder, filename = prefix.rsplit("/", 1)
            files = fs.ls(folder)
            return [f for f in files if f.startswith(prefix)]

        # +------------------------------ +
        # | Obtain IDs for trains and PRs |
        # +------------------------------ +

        def get_train_to_id(train_num_filter: TrainNumFilter) -> tuple[dict[str, int], list[str]]:
            # Get frequent train_nums
            train_num_nbrs = all_stats.train_num
            if isinstance(cfg_nextdesserte.min_train_num_nbr, int):
                train_nums = sorted(t for t, n in train_num_nbrs.items() if n >= cfg_nextdesserte.min_train_num_nbr)
            else:
                train_nums = sorted(
                    t
                    for t, n in train_num_nbrs.items()
                    if n >= cfg_nextdesserte.min_train_num_nbr * len(nextdesserte_days_train)
                )
            # Filter per train_type
            train_nums = [t for t in train_nums if train_num_filter(t)]
            # Get train_to_id, a dict that gives a unique ID to each train number
            train_to_id = {t: i + 1 for i, t in enumerate(train_nums)}
            train_to_id["other"] = 0
            return train_to_id, train_nums

        def get_pr_to_id(train_nums: list[int]) -> tuple[dict[str, int], list[tuple[str, ...]]]:
            # Get frequent prs, those that have been seen more than nextdesserte_MIN_TRAIN_NUM_NBR
            pr_nbr_passages = all_stats.pr_nbr_passage
            prs = sorted(
                {pr for pr, n in pr_nbr_passages.items() if n >= cfg_nextdesserte.min_train_num_nbr}
            )  # not the option of a percentage of days
            # Get pr_to_id, a dict that gives a unique ID to each PR
            pr_to_id = {t: i + 3 for i, t in enumerate(sorted({pr[0] for pr in prs}))}
            pr_to_id["other"] = 0
            pr_to_id["preDeparture"] = 1
            pr_to_id["postArrival"] = 2
            return pr_to_id, prs

        train_num_filter = TrainNumFilterNotTrainType(typ="_rer")
        # def train_num_filter(train_num):
        #     return "_rer" not in train_info.get_train_type_str(train_num)  # remove rer trains

        train_to_id, train_nums = get_train_to_id(train_num_filter)
        pr_to_id, prs = get_pr_to_id(train_nums)  # type: ignore

        # +-------------------------+
        # | Handling embedding data |
        # +-------------------------+

        emb_path = paths.embedding_weights_f.format(cfg_hash_train).rsplit("/", 1)[0]
        data_path_f = emb_path + "/data/{}.pickle"  # f-string for training data storage

        def flatten_list(data: list[list[Any]]) -> list[Any]:
            res = []
            for d in data:
                res += d
            return res

        def create_nextdesserte_emb_data(
            days: list[str], sillon_filter: SillonFilter, verbose: bool = True
        ) -> tuple[list[Any], ...]:
            next_pr = []
            next_pr2 = []
            next_pr3 = []
            if verbose:
                logger.info("Generating data for train embedding")

            files = get_snapshots_files().statistics_snapshots

            data_set = RawSnapshotDataset(files)
            data_loader = DataLoader(
                data_set,
                collate_fn=lambda x: x,  # use custom collate_fn to allow returning custom types
                shuffle=False,
                batch_size=1,
                num_workers=4,
            )

            for batch in data_loader:
                raw_snapshot: list[SillonChunk] = batch[0]
                # sillons = {k: pr_filter(v) for k, v in sillons.items() if sillon_filter(v)}
                sillons = [sillon for sillon in raw_snapshot if sillon_filter(sillon)]
                sills = [
                    (
                        train_to_id.get(sill.train_num, train_to_id["other"]),
                        [pr_to_id["preDeparture"]]
                        + [pr_to_id.get(pr, pr_to_id["other"]) for pr in sill.prev_prs + sill.foll_prs]
                        + [pr_to_id["postArrival"]] * 3,
                    )
                    for sill in sillons
                ]
                next_pr.append(
                    [(train_num, pr, prs[i + 1]) for train_num, prs in sills for i, pr in enumerate(prs[:-3])]
                )
                next_pr2.append(
                    [(train_num, pr, prs[i + 2]) for train_num, prs in sills for i, pr in enumerate(prs[:-3])]
                )
                next_pr3.append(
                    [(train_num, pr, prs[i + 3]) for train_num, prs in sills for i, pr in enumerate(prs[:-3])]
                )
            next_pr = list(set(flatten_list(next_pr)))
            next_pr2 = list(set(flatten_list(next_pr2)))  # shady? why set?
            next_pr3 = list(set(flatten_list(next_pr3)))
            random.shuffle(next_pr)
            random.shuffle(next_pr2)
            random.shuffle(next_pr3)
            return next_pr, next_pr2, next_pr3

        def create_and_save_nextdesserte_emb_data(days: list[str], sillon_filter: SillonFilter) -> None:
            delete_nextdesserte_emb_data()
            days_per_month = defaultdict(list)  # {"2022-02" : ["2022-02-01", "2022-02-02", "2022-02-03", ...]}
            for day in days:
                days_per_month[day[:-3]].append(day)
            logger.info("Generating training data for train embedding")
            for month in tqdm(sorted(days_per_month)):
                data = create_nextdesserte_emb_data(days_per_month[month], sillon_filter)
                save_pickle(data_path_f.format("month_" + month), data)

        def exists_nextdesserte_emb_data() -> bool:
            path = data_path_f.format("shuffled_")
            files = glob(path.rsplit(".", 1)[0])
            return len(files) > 0

        def delete_nextdesserte_emb_data() -> None:
            path = data_path_f.format("")
            files = glob(path.rsplit(".", 1)[0])
            for f in files:
                fs.rm(f)

        def shuffle_saved_nextdesserte_emb_data() -> None:
            path = data_path_f.format("month_")
            files = glob(path.rsplit(".", 1)[0])
            n = len(files)
            for i in tqdm(range(n)):
                xx = [[], [], []]  # type: list[list[Any]]
                for f in files:
                    x = load_pickle(f)
                    xx[0] += x[0][i::n]
                    xx[1] += x[1][i::n]
                    xx[2] += x[2][i::n]
                random.shuffle(xx[0])
                random.shuffle(xx[1])
                random.shuffle(xx[2])
                save_pickle(
                    data_path_f.format("shuffled_" + str(i)),
                    xx,
                )
                # shuffles all the months and returns as many files as there were months but mixed

        def load_all_train_data() -> tuple[list[Any], ...]:
            path = data_path_f.format("shuffled_")
            files = glob(path.rsplit(".", 1)[0])

            next_pr = []
            next_pr2 = []
            next_pr3 = []
            for f in files:
                d, dp, dpp = load_pickle(f)
                next_pr += d
                next_pr2 += dp
                next_pr3 += dpp
            random.shuffle(next_pr)
            random.shuffle(next_pr2)
            random.shuffle(next_pr3)
            return next_pr, next_pr2, next_pr3

        # +-----------------------+
        # | Compute training data |
        # +-----------------------+

        if not exists_nextdesserte_emb_data():
            sillon_filter = SillonFilterTrainNums(train_nums)
            # pr_filter = PrFilterPrsWithSillonTagger(prs, SillonTaggerAll())
            delete_nextdesserte_emb_data()
            create_and_save_nextdesserte_emb_data(nextdesserte_days_train, sillon_filter)
            shuffle_saved_nextdesserte_emb_data()

        # +--------------------------------------------+
        # | We define the embedding model and train it |
        # +--------------------------------------------+

        tEmb = nn.Embedding(len(train_to_id), cfg_nextdesserte.train_dim).to(device=DEVICE)
        pEmb = nn.Embedding(len(pr_to_id), cfg_nextdesserte.pr_dim).to(device=DEVICE)

        model = nn.Sequential(
            nn.Linear(
                cfg_nextdesserte.train_dim + cfg_nextdesserte.pr_dim,
                cfg_nextdesserte.model_dim,
            ),
            nn.PReLU(),
            nn.Linear(cfg_nextdesserte.model_dim, len(pr_to_id)),
        ).to(device=DEVICE)  # manque la normalisation pour que ca somme a un: ou est le softmax???
        optim = torch.optim.Adam(set(model.parameters()) | set(pEmb.parameters()) | set(tEmb.parameters()), lr=1e-3)

        loss_fn = nn.CrossEntropyLoss()

        # train model for embedding
        ii = 0
        d, dp, dpp = load_all_train_data()

        logger.info("Starting training...")
        metrics = {"loss": [], "top1": [], "top3": []}  # type: dict[str, list[float]]
        batch_size = cfg_nextdesserte.batch_size
        for _ in range(cfg_nextdesserte.n_iter):
            for _ in range(max((len(d) // batch_size + 1) // 5, 1)):
                ii += 1
                optim.zero_grad()
                rand_sel = torch.randint(
                    len(d),
                    size=(int(cfg_nextdesserte.prop_norm * batch_size),),
                )
                rand_selp = torch.randint(
                    len(dp),
                    size=(int(cfg_nextdesserte.prop_skip * batch_size),),
                )
                rand_selpp = torch.randint(
                    len(dpp),
                    size=(int(cfg_nextdesserte.prop_dskip * batch_size),),
                )

                sl = torch.tensor([d[i] for i in rand_sel], dtype=torch.long, device=DEVICE)
                slp = torch.tensor([dp[i] for i in rand_selp], dtype=torch.long, device=DEVICE)
                slpp = torch.tensor([dpp[i] for i in rand_selpp], dtype=torch.long, device=DEVICE)

                sl = torch.cat((sl, slp, slpp), dim=0)

                emb = torch.cat((tEmb(sl[:, 0]), pEmb(sl[:, 1])), dim=1)

                # Mask random components of the embeddings : TODO delete?
                mask = torch.rand(len(sl), device=DEVICE).ge(cfg_nextdesserte.mask_prop)
                emb *= mask[:, None]

                out = model(emb)

                if (sl[:, 2] != (sl[:, 2] % len(pr_to_id))).any():
                    raise RuntimeError("Imcomprehensible error from torch / cuda : wrong classes in sl ??")
                loss = loss_fn(out, sl[:, 2] % len(pr_to_id))
                loss.backward()
                optim.step()

                out = torch.topk(out, 3).indices
                top1 = float((out[:, 0] == sl[:, 2]).sum())
                top3 = float(((out.T - sl[:, 2]) == 0).sum())

                if ii % (cfg_nextdesserte.log_frequency) == 0:
                    logger.info(
                        "{:<7}   {:<10} {:<7} {:<7}".format(
                            round((ii * cfg_nextdesserte.batch_size) / 10**6, 3),
                            round(100 * float(loss), 3),
                            round(100 * top1 / len(sl), 3),
                            round(100 * top3 / len(sl), 3),
                        )
                    )
                    metrics["loss"].append(100 * float(loss))
                    metrics["top1"].append(100 * top1 / len(sl))  # TODO pourquoi converge a 2/3 et pas 1/3????
                    metrics["top3"].append(100 * top3 / len(sl))
                if ii % (cfg_nextdesserte.saving_frequency) == 0:
                    logger.info("Saving to disk...")
                    save_torch(
                        paths.embedding_weights_f.format(cfg_hash_train),
                        tEmb.state_dict(),
                    )
                    save_torch(
                        paths.embedding_weights_f.format(cfg_hash_pr),
                        pEmb.state_dict(),
                    )
                    # if cfg.embeddings.save_training_data:
                    #     save_torch(
                    #         paths.nextdesserte_emb_model_f.format(cfg_nextdesserte.name),
                    #         model.state_dict(),
                    #     )
                    #     save_pickle(paths.nextdesserte_emb_losses_f.format(cfg_nextdesserte.name), metrics)

        params_train = {
            "params": config_to_dict(cfg_nextdesserte),
            "key_type": "train number",
        }
        params_pr = {
            "params": config_to_dict(cfg_nextdesserte),
            "key_type": "ID Gaia du PR",
        }
        save_json(paths.embedding_config_f.format(cfg_hash_train), params_train, indent=4)
        save_json(paths.embedding_config_f.format(cfg_hash_pr), params_pr, indent=4)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash_train), train_to_id)
        save_json(paths.embedding_key_to_emb_f.format(cfg_hash_pr), pr_to_id)

        # +--------------------------------+
        # | Plotting of training procedure |
        # +--------------------------------+

        step = 1

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(metrics["loss"][::step], "r" + "-", label="loss")
        ax.legend()
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(metrics["top1"][::step], "g" + "-", label="top1")
        ax.plot(metrics["top3"][::step], "b" + "-", label="top3")
        ax.legend()
        fig.suptitle("Loss evolution during training of nextdesserte embedding")
        save_matplotlib_fig(paths.embedding_figures_f.format(cfg_hash_train), fig)
        plt.close(fig)

    return cfg_hash_train, cfg_hash_pr
