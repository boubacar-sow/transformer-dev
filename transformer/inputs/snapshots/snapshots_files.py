"""
Contains the SnapshotFiles class, which provides a list of file names which contain the snapshots
for the train and test sets.
"""

import os
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Self

from transformer.inputs.data_structures import SillonChunk
from transformer.inputs.resources.pr_infos import compute_pr_infos
from transformer.utils.config import get_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_json
from transformer.utils.logger import logger
from transformer.utils.misc import set_random_seeds
from transformer.utils.times import TIMEZONE_FRANCE, get_train_validation_test_days

cfg, paths = get_config()
fs = FileSystem()


def load_raw_snapshot(filename: str) -> list[SillonChunk]:
    pr_info = compute_pr_infos()
    raw_data = load_json(filename, zstd_format=True)
    raw_snapshot = [SillonChunk(**completed_sillon) for completed_sillon in raw_data]

    for t in raw_snapshot:
        if t.train_num == "":
            t.train_num = "0"
        t.foll_obs_times = [x if x is not None else np.nan for x in t.foll_obs_times]  # replace None with NaN
        t.foll_delays = [x if x is not None else np.nan for x in t.foll_delays]  # replace None with NaN

        if cfg.model.use_inputs.lignepk_embedding:
            for pr in t.prev_prs:
                code_ligne, pk = pr_info[pr].get_single_code_ligne_pk()
                t.prev_codes_ligne.append(str(code_ligne))
                t.prev_pks.append(pk)

            for pr in t.foll_prs:
                code_ligne, pk = pr_info[pr].get_single_code_ligne_pk()
                t.foll_codes_ligne.append(str(code_ligne))
                t.foll_pks.append(pk)
    return raw_snapshot


class RawSnapshotDataset(Dataset[list[SillonChunk]]):
    """
    Dataset producing objects of the form (snapshot_file_name, list[SillonChunk])
    """

    def __init__(self, snapshots_files: list[str]) -> None:
        self.snapshots_files = snapshots_files

    def __len__(self) -> int:
        """
        Denotes the total number of samples
        """
        return len(self.snapshots_files)

    def __getitem__(self, index: int) -> list[SillonChunk]:
        """
        Loads one sample of data, i.e. one snapshot
        """
        filename = self.snapshots_files[index]
        return load_raw_snapshot(filename)


@dataclass
class SnapshotFiles:
    statistics_snapshots: list[str]
    train_snapshots: list[str]
    validation_snapshots: list[str]
    test_snapshots: list[str]


class SnapshotFilesLoader:
    """
    Class for returning list of existing files containing snapshots
    Example of use:
        SnapshotFilesLoader(DAYS_TEST).get() # get all test files, ie files from days DAYS_TEST
        SnapshotFilesLoader(DAYS_TRAIN).shuffle().get() # get all train files, shuffled
        SnapshotFilesLoader("2020-04-17").get() # get data files of the day "2020-04-17"
        SnapshotFilesLoader(DAYS_TEST).get_per_day() # get all test files, as a dict where keys are days
        SnapshotFilesLoader(DAYS_TEST[::2]).get() # get data files of half test days
        SnapshotFilesLoader(DAYS_TEST).sample(interval=3).get() # get one third of data files
                                                            # ie a snapshot every 12 minutes (modulo randomization)
    """

    def __init__(
        self,
        days: list[str],
        verbose: bool = False,
        random_seed: int | None = None,
    ):
        """
        days : (str) or (list(str)) : (list of) days from where we extract data files
            "days" could also be a list of data files, or a dict of list of data files (ie data files
                sorted per day)
        verbose : (bool) = True : whether it prints the number of samples detected
        """

        self.random_seed = random_seed if random_seed is not None else cfg.training.random_seed

        if isinstance(days, list) and (len(days) > 0) and ("/" in days[0]):
            self.files = days.copy()
        else:
            if isinstance(days, str):
                days = [days]
            self.files = []
            for day in sorted(days):
                dirname = os.path.dirname(paths.snapshot_f).format(day)
                if fs.exists(dirname):
                    self.files += sorted(fs.ls(dirname))
            if verbose:
                logger.info("SnapshotFiles: detected {} snapshots for {} days.".format(len(self.files), len(days)))
            self.files.sort()

        assert len(self.files) > 0, "No snapshot were found!"

    def sample(self, interval_minutes: int) -> Self:
        """
        Sampling strategy:
        1. Divide the day into bins of `interval` minutes.
        2. For each bin, if it contains at least one snapshot file,
            pick one at random.
        """
        random.seed(self.random_seed)

        interval_seconds = interval_minutes * 60

        timestamps = np.array(
            [datetime.fromisoformat(x.rsplit("/", 1)[-1].split(".", 1)[0]).timestamp() for x in self.files], dtype=int
        )
        ts_bins = timestamps // interval_seconds
        min_ts, max_ts = min(timestamps), max(timestamps)
        ts_sample = []

        # Iterate over bins of the form [k*interval, (k+1)*interval[
        for i in range(min_ts // interval_seconds, max_ts // interval_seconds + 1):
            # Select the timestamps in the bin
            ts_bin = timestamps[ts_bins == i]
            if len(ts_bin) > 0:
                ts_sample.append(np.random.choice(ts_bin))  # Pick one at random

        self.files = [f for f, t in zip(self.files, timestamps) if (t in ts_sample)]
        return self

    def select_times(self, min_time: int, max_time: int) -> Self:
        """
        Removes all files that are before and after given times in each day (in Europe/Paris timezone).
        Args:
            min_hour: mininum time in the day, in minutes
            max_hour: maximum time in the day, in minutes
        """
        new_files = []
        for f in self.files:
            dt_string = f.rsplit("/", 1)[-1].split(".", 1)[0]  # e.g., 2024-03-06T00:34:00+00:00
            dt = datetime.fromisoformat(dt_string).astimezone(TIMEZONE_FRANCE)
            time_minutes = dt.hour * 60 + dt.minute
            if time_minutes >= min_time and time_minutes <= max_time:
                new_files.append(f)
        self.files = new_files
        return self

    def get(self) -> list[str]:
        logger.debug("SnapshotFiles: using {} snapshots.".format(len(self.files)))
        return self.files


def get_snapshots_files(seed: int | None = None, verbose: bool = False) -> SnapshotFiles:
    if seed is not None:
        set_random_seeds(seed)

    days_train_validation, days_test = get_train_validation_test_days()

    # Get train + validation snapshots
    sample_minutes = cfg.data.snapshots.train_nbr_minutes_interval
    train_validation_snapshots = (
        SnapshotFilesLoader(days_train_validation, verbose=False)
        .sample(sample_minutes)
        .select_times(cfg.data.snapshots.train_start_minute, cfg.data.snapshots.train_end_minute)
        .get()
    )

    # Split train set and validation set
    validation_set_size = int(cfg.data.snapshots.validation_set_proportion * len(train_validation_snapshots))
    validation_indices = np.random.choice(len(train_validation_snapshots), size=validation_set_size)
    train_indices = [i for i in range(len(train_validation_snapshots)) if i not in validation_indices]

    train_snapshots = [train_validation_snapshots[i] for i in train_indices]
    validation_snapshots = [train_validation_snapshots[i] for i in validation_indices]

    validation_numpy = np.array([train_validation_snapshots[i] for i in validation_indices])
    np.random.shuffle(validation_numpy)
    validation_snapshots = list(validation_numpy)

    # Get the snapshots used to compute the statistics
    sample_minutes = cfg.data.stats.nbr_minutes_interval
    days_statistics = days_train_validation[:: cfg.data.stats.nbr_days_interval]
    statistics_snapshots = (
        SnapshotFilesLoader(days_statistics, verbose=False)
        .sample(sample_minutes)
        .select_times(cfg.data.snapshots.train_start_minute, cfg.data.snapshots.train_end_minute)
        .get()
    )

    # Get tests snapshots
    test_snapshots = (
        SnapshotFilesLoader(days_test, verbose=False)
        .sample(cfg.data.snapshots.test_nbr_minutes_interval)
        .select_times(cfg.data.snapshots.test_start_minute, cfg.data.snapshots.test_end_minute)
        .get()
    )

    if verbose:
        logger.info(f"SnapshotFiles: {len(statistics_snapshots)} statistics snapshots.")
        logger.info(f"SnapshotFiles: {len(train_snapshots)} training snapshots.")
        logger.info(f"SnapshotFiles: {len(validation_snapshots)} validation snapshots.")
        logger.info(f"SnapshotFiles: {len(test_snapshots)} test snapshots.")

    return SnapshotFiles(
        statistics_snapshots=statistics_snapshots,
        train_snapshots=train_snapshots,
        validation_snapshots=validation_snapshots,
        test_snapshots=test_snapshots,
    )
