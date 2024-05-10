import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer.inputs.data_structures import MeanStd, NDMeanStd, SillonChunk
from transformer.inputs.resources.pr_graph import get_graph_pr_ids
from transformer.inputs.resources.pr_infos import compute_pr_infos
from transformer.inputs.resources.train_type import get_nbr_train_types, get_train_type_eigenvect
from transformer.inputs.snapshots.snapshots_files import RawSnapshotDataset, get_snapshots_files
from transformer.inputs.snapshots.time_transform import TimeTransform
from transformer.inputs.utils.sillons_filters import SILLON_FILTERS
from transformer.inputs.utils.sillons_taggers import SILLON_TAGGERS
from transformer.inputs.utils.train_num_filters import TRAIN_NUM_FILTERS
from transformer.utils.config import get_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_json, load_pickle, save_json, save_pickle
from transformer.utils.logger import logger
from transformer.utils.misc import get_data_hash, reverse_list_index, weighted_dict_mean_std
from transformer.utils.s3 import copy_folder_to_s3
from transformer.utils.stats_counter import StatsCounter

cfg, paths = get_config()
fs = FileSystem()


@dataclass
class InterPrStats:
    """
    Statistics on the travel times or delays between consecutive PRs.
    The times are normalized using TimeTransform
    """

    counts: dict[tuple[str, ...], int]
    clearing_times: dict[tuple[str, ...], MeanStd]
    diff_delay: dict[tuple[str, ...], MeanStd]
    middle_diff_delay: dict[tuple[str, ...], MeanStd]


@dataclass
class AllStats:
    """Dataclass containing the statistics summarizing the training data set"""

    delays: MeanStd
    prev_theo_times: MeanStd
    foll_theo_times: MeanStd
    pk: MeanStd
    nbr_train: MeanStd
    week_day: MeanStd
    current_time: MeanStd

    train_type: NDMeanStd
    train_num: dict[str, int]
    pr_nbr_passage: dict[tuple[str, ...], int]

    avg_inter_pr_diff_delay: tuple[MeanStd, MeanStd]
    avg_inter_pr_middle_diff_delay: tuple[MeanStd, MeanStd]

    inter_pr: InterPrStats

    def __repr__(self) -> str:
        return {
            "delays": self.delays,
            "prev_theo_times": self.prev_theo_times,
            "foll_theo_times": self.foll_theo_times,
            "pk": self.pk,
            "nbr_train": self.nbr_train,
            "week_day": self.week_day,
            "current_time": self.current_time,
            "train_type": self.train_type,
            "train_num": self.train_num,
        }.__repr__()


class StatisticsFactory:
    """
    A class used to store the distribution of quantities of interest,
    and aggregate them into the statistics used in the Transformer training.
    """

    inter_pr_clearing_time_counter: dict[tuple[str, ...], StatsCounter]
    inter_pr_diff_delay_counter: dict[tuple[str, ...], StatsCounter]

    def __init__(self) -> None:
        self.pr_infos = compute_pr_infos()
        self.middle_diff_delay_bounds = cfg.model.use_inputs.middle_diff_delay_bounds

        self.train_num_filter = TRAIN_NUM_FILTERS[cfg.data.filters.train_num_filter]
        self.sillon_filter = SILLON_FILTERS[cfg.data.filters.sillon_filter]
        self.global_sillon_tagger = SILLON_TAGGERS[cfg.data.filters.global_sillon_tagger]
        self.diff_delay_sillon_tagger = SILLON_TAGGERS[cfg.data.filters.diff_delay_sillon_tagger]

        self.delays_counter = StatsCounter()
        self.prev_theo_times_counter = StatsCounter()
        self.foll_theo_times_counter = StatsCounter()
        self.pk_counter = StatsCounter()
        self.nbr_train_counter = StatsCounter()
        self.week_day_counter = StatsCounter()
        self.current_time_counter = StatsCounter()

        self.train_num_counter = Counter()  # type: Counter[str]
        self.pr_nbr_passage_counter = Counter()  # type: Counter[tuple[str, ...]]

        self.train_type_counter = [StatsCounter() for _ in range(get_nbr_train_types())]

        bin_size = cfg.data.stats.bin_size
        self.inter_pr_clearing_time_counter = defaultdict(lambda: StatsCounter(bin_size=bin_size))
        self.inter_pr_diff_delay_counter = defaultdict(lambda: StatsCounter(bin_size=bin_size))

        self.time_transform = TimeTransform()

        graph_pr_ids = get_graph_pr_ids()
        pr_doublons = load_json(paths.PRs_doublons)
        self.non_missing_pr_ids = (set(graph_pr_ids) - set(pr_doublons)).union({"preDeparture", "postArrival"})
        # syntax: missing_prs_neighbors[pr_id] = [(current_time, (pr_id_1, time_1), (pr_id_2, time_2)), ...]
        self.missing_prs_neighbors: dict[str, list[tuple[float, tuple[str, float], tuple[str, float]]]] = defaultdict(
            list
        )

    def update(self, raw_snapshot: list[SillonChunk], snapshot_day: str, snapshot_time: int) -> None:  # noqa: C901
        """
        Update the Counters with the values in the given snapshot.
        """

        nbr_trains = np.array(
            [(t.prev_obs_types[-1] != "T") and (t.foll_obs_types[0] != "O") for t in raw_snapshot]
        ).sum()
        self.nbr_train_counter.add(nbr_trains)
        day = time.strptime(snapshot_day, "%Y-%m-%d").tm_wday
        self.week_day_counter.add(day)
        self.current_time_counter.add(snapshot_time)

        for sillon in raw_snapshot:
            if not self.train_num_filter(sillon.train_num):
                continue

            if not self.sillon_filter(sillon):
                continue

            global_tag = self.global_sillon_tagger(sillon)
            diff_delay_tag = self.diff_delay_sillon_tagger(sillon)

            pr_list = sillon.prev_prs + sillon.foll_prs

            n_prev = cfg.model.n_prev  # number of previous PRs
            n_foll = cfg.model.n_foll  # number of upcoming PRs to predict

            n_pre_departure = min(
                n_foll, reverse_list_index(["preDeparture"] + pr_list, "preDeparture")
            )  # idx of the first pr different from preDeparture
            n_post_arrival = min(
                n_foll, (pr_list + ["postArrival"]).index("postArrival")
            )  # idx of the last pr different from postArrival
            n_pre_departure_prev = min(n_prev, n_pre_departure)
            n_post_arrival_foll = max(0, n_post_arrival - n_prev)

            delays = sillon.prev_delays + sillon.foll_delays
            times = sillon.prev_theo_times + sillon.foll_theo_times

            delays_nn = self.time_transform.delay_to_nn(
                np.array(delays[n_pre_departure:n_post_arrival]), normalize=False
            )
            delays_nn = delays_nn[~np.isnan(delays_nn)]
            prev_theo_times_nn = self.time_transform.times_to_nn(
                np.array(sillon.prev_theo_times[n_pre_departure_prev:]), normalize=False
            )
            foll_theo_times_nn = self.time_transform.times_to_nn(
                np.array(sillon.foll_theo_times[:n_post_arrival_foll]), normalize=False
            )
            self.delays_counter.add(delays_nn)
            self.prev_theo_times_counter.add(prev_theo_times_nn)
            self.foll_theo_times_counter.add(foll_theo_times_nn)

            prev_pks = [self.pr_infos[pr].get_single_code_ligne_pk()[1] for pr in sillon.prev_prs]
            foll_pks = [self.pr_infos[pr].get_single_code_ligne_pk()[1] for pr in sillon.prev_prs]
            tab = np.array(prev_pks + foll_pks)[n_pre_departure:n_post_arrival]
            tab = tab[tab != 0]
            self.pk_counter.add(tab)

            self.train_num_counter.update([sillon.train_num])

            train_type = get_train_type_eigenvect(sillon.train_num)
            for i in range(len(train_type)):
                self.train_type_counter[i].add(float(train_type[i]))

            last_non_missing_prs: list[tuple[str, float] | None] = [None, None]
            missing_prs: list[dict[str, Any]] = []
            for i, (pr_id, pr_time) in enumerate(zip(pr_list, times)):
                if pr_id in ["preDeparture", "postArrival"]:
                    continue

                # Count pr passages
                global_key = (pr_id,) + global_tag
                self.pr_nbr_passage_counter.update([global_key])

                # Register missing PRs position in their sillons
                if pr_id not in self.non_missing_pr_ids:
                    if len(self.missing_prs_neighbors[pr_id]) < cfg.data.stats.max_sillons_for_missing_prs:
                        pass
                    missing_prs.append(
                        {
                            "pr_id": pr_id,
                            "pr_time": pr_time,
                            "prev_1": last_non_missing_prs[0],
                            "prev_2": last_non_missing_prs[1],
                            "foll_1": None,
                            "foll_2": None,
                        }
                    )
                else:
                    if last_non_missing_prs[0] is None and last_non_missing_prs[1] is not None:
                        for i in range(len(missing_prs)):
                            missing_pr = missing_prs[i]
                            if missing_pr["foll_1"] is None:
                                missing_pr["foll_1"] = (pr_id, pr_time)
                            elif missing_pr["foll_2"] is None:
                                missing_pr["foll_2"] = (pr_id, pr_time)

                    last_non_missing_prs = [last_non_missing_prs[1], (pr_id, pr_time)]

            for missing_pr in missing_prs:
                self._update_missing_prs_neighbors(**missing_pr)

            for i in range(n_pre_departure, n_post_arrival + 1):
                j = i + 1
                if pr_list[i] == "preDeparture" and pr_list[j] == "preDeparture":
                    continue
                global_key = (pr_list[i], pr_list[j]) + global_tag
                diff_delay_key = (pr_list[i], pr_list[j]) + diff_delay_tag
                times_diff = times[j] - times[i]
                delay_diff = delays[j] - delays[i]
                self.inter_pr_clearing_time_counter[global_key].add(times_diff)
                if not np.isnan(delay_diff):
                    self.inter_pr_diff_delay_counter[diff_delay_key].add(delay_diff)

    def _update_missing_prs_neighbors(
        self,
        pr_id: str,
        pr_time: float,
        prev_1: tuple[str, float] | None,
        prev_2: tuple[str, float] | None,
        foll_1: tuple[str, float] | None,
        foll_2: tuple[str, float] | None,
    ) -> None:
        """
        Update missing_prs_neighbors in place to account for a new observation.

        Args:
        missing_prs_neighbors: dict such that missing_prs_neighbors[pr_id] corresponds gives the relative position
                                (in time) of the PR with respect to two other non-missing PRs in a sillon.

        """

        if ((prev_1 is not None) or (prev_2 is not None)) and ((foll_1 is not None) or (foll_2 is not None)):
            if prev_2 is not None:
                pr_id_1, time_1 = prev_2
            else:
                pr_id_1, time_1 = prev_1  # type: ignore[misc]
            if foll_1 is not None:
                pr_id_2, time_2 = foll_1
            else:
                pr_id_2, time_2 = foll_2  # type: ignore[misc]
        elif (prev_1 is not None) and (prev_2 is not None):
            pr_id_1, time_1 = prev_1
            pr_id_2, time_2 = prev_2
        elif (foll_1 is not None) and (foll_2 is not None):
            pr_id_1, time_1 = foll_1
            pr_id_2, time_2 = foll_2
        else:
            return

        self.missing_prs_neighbors[pr_id].append((pr_time, (pr_id_1, time_1), (pr_id_2, time_2)))

    def save_summary(self, cfg_hash: str) -> None:
        """
        Computes the statistics over the StatsCounters and save the result.
        The statistics will be copied to S3 if cfg.data.save_results_to_s3 is True
        """

        # Average simple global statistics
        delay = self.delays_counter.mean_std()
        prev_theo_times = self.prev_theo_times_counter.mean_std()
        foll_theo_times = self.foll_theo_times_counter.mean_std()
        pk = self.pk_counter.middle_mean_std(lower_bound=0.25, upper_bound=0.75)
        nbr_train = self.nbr_train_counter.mean_std()
        week_day = self.week_day_counter.mean_std()
        current_time = self.current_time_counter.mean_std()

        # Average per-category global statistics
        train_type_mean_std = [self.train_type_counter[i].mean_std() for i in range(len(self.train_type_counter))]
        train_type = NDMeanStd(
            mean=np.array([train_type_mean_std[i].mean for i in range(len(train_type_mean_std))]),
            std=np.array([train_type_mean_std[i].std for i in range(len(train_type_mean_std))]),
        )
        train_num = dict(self.train_num_counter)
        pr_nbr_passage = dict(self.pr_nbr_passage_counter)

        # Compute MeanStd for inter_pr
        inter_pr_clearing_times = {k: v.mean_std() for k, v in self.inter_pr_clearing_time_counter.items()}
        inter_pr_counts = {k: v.nbr() for k, v in self.inter_pr_diff_delay_counter.items()}
        inter_pr_diff_delay = {
            k: v.mean_std(transform=self.time_transform._delay_to_nn_func)
            for k, v in self.inter_pr_diff_delay_counter.items()
        }
        inter_pr_middle_diff_delay = {
            k: v.middle_mean_std(
                lower_bound=self.middle_diff_delay_bounds[0],
                upper_bound=self.middle_diff_delay_bounds[1],
                transform=self.time_transform._delay_to_nn_func,
            )
            for k, v in self.inter_pr_diff_delay_counter.items()
        }

        # Compute average over the MeanStd of inter_pr
        avg_inter_pr_diff_delay = weighted_dict_mean_std(
            inter_pr_diff_delay,
            inter_pr_counts,  # type: ignore
            keys=inter_pr_counts.keys(),
        )
        avg_inter_pr_middle_diff_delay = weighted_dict_mean_std(
            inter_pr_middle_diff_delay,
            inter_pr_counts,  # type: ignore
            keys=inter_pr_counts.keys(),
        )

        global_statistics = {
            "delays": delay,
            "prev_theo_times": prev_theo_times,
            "foll_theo_times": foll_theo_times,
            "pk": pk,
            "nbr_train": nbr_train,
            "week_day": week_day,
            "current_time": current_time,
            "train_type": train_type,
            "train_num": train_num,
            "pr_nbr_passage": pr_nbr_passage,
            "avg_inter_pr_diff_delay": avg_inter_pr_diff_delay,
            "avg_inter_pr_middle_diff_delay": avg_inter_pr_middle_diff_delay,
        }

        save_pickle(paths.global_statistics_f.format(cfg_hash), global_statistics, zstd_format=True)

        inter_pr_statistics = {
            "counts": inter_pr_counts,
            "clearing_times": inter_pr_clearing_times,
            "diff_delay": inter_pr_diff_delay,
            "middle_diff_delay": inter_pr_middle_diff_delay,
        }

        save_pickle(paths.inter_pr_statistics_f.format(cfg_hash), inter_pr_statistics, zstd_format=True)

        save_json(paths.config_statistics_f.format(cfg_hash), OmegaConf.to_object(cfg.data), indent=4)

        save_json(paths.missing_PRs_neighbors_f.format(cfg_hash), self.missing_prs_neighbors, indent=4)


def compute_statistics() -> None:
    cfg_hash = get_data_hash()
    path_global = paths.global_statistics_f.format(cfg_hash)
    path_inter_pr = paths.inter_pr_statistics_f.format(cfg_hash)
    path_missing_prs = paths.missing_PRs_neighbors_f.format(cfg_hash)

    logger.info(f"Data config: {cfg_hash}")

    if (
        cfg.training.use_precomputed.statistics
        and fs.exists(path_global)
        and fs.exists(path_inter_pr)
        and fs.exists(path_missing_prs)
    ):
        logger.info("Using pre-computed statistics statistics.")
        return

    logger.info("Computing statistics...")

    files = get_snapshots_files().statistics_snapshots

    data_set = RawSnapshotDataset(files)
    data_loader = DataLoader(
        data_set,
        collate_fn=lambda x: x,  # use custom collate_fn to allow returning custom types
        shuffle=False,
        batch_size=1,
        num_workers=cfg.training.data_loader_num_workers,
    )

    streaming_statistics = StatisticsFactory()
    for batch in tqdm(data_loader):
        raw_snapshot = batch[0]  # type: list[SillonChunk]

        snapshot_datetime = datetime.fromisoformat(raw_snapshot[0].datetime)
        snapshot_day = snapshot_datetime.strftime("%Y-%m-%d")
        snapshot_time = snapshot_datetime.hour * 60 + snapshot_datetime.minute

        streaming_statistics.update(raw_snapshot, snapshot_day, snapshot_time)

    streaming_statistics.save_summary(cfg_hash)

    if cfg.data.save_results_to_s3:
        logger.info("Saving statistics to S3.")
        copy_folder_to_s3(paths.snapshots_stats_folder_f.format(cfg_hash))


def load_statistics(cfg_hash: Optional[str] = None) -> AllStats:
    logger.info("Loading statistics.")
    cfg_hash = get_data_hash()
    global_stats = load_pickle(paths.global_statistics_f.format(cfg_hash), zstd_format=True)
    inter_pr_stats = load_pickle(paths.inter_pr_statistics_f.format(cfg_hash), zstd_format=True)

    inter_pr = InterPrStats(**inter_pr_stats)
    global_stats["inter_pr"] = inter_pr
    return AllStats(**global_stats)
