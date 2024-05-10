import numpy as np

from transformer.inputs.data_structures import MeanStd, NDMeanStd
from transformer.inputs.resources.train_type import get_nbr_train_types
from transformer.inputs.snapshots.statistics import AllStats, InterPrStats
from transformer.utils.config import get_config

cfg, paths = get_config()


class MockInterPrStats(InterPrStats):
    counts: dict[tuple[str, ...], int]
    clearing_times: dict[tuple[str, ...], MeanStd]
    diff_delay: dict[tuple[str, ...], MeanStd]
    middle_diff_delay: dict[tuple[str, ...], MeanStd]

    def __init__(self) -> None:
        count_names = [
            "local_counts",
        ]
        mean_std_names = [
            "clearing_times",
            "diff_delay",
            "middle_diff_delay",
        ]

        count = {("PR-A", "PR-B", "voyageurs"): 1, ("PR-B", "PR-C", "voyageurs"): 2}
        for name in count_names:
            self.__setattr__(name, count)

        mean_std = {("PR-A", "PR-B", "voyageurs"): MeanStd(0.5, 1.3), ("PR-V", "PR-C", "voyageurs"): MeanStd(2.4, 5)}
        for name in mean_std_names:
            self.__setattr__(name, mean_std)


class MockAllStats(AllStats):
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

    inter_pr: MockInterPrStats

    def __init__(self) -> None:
        mean_std = MeanStd(1.2, 3.4)
        self.delays = mean_std
        self.prev_theo_times = mean_std
        self.foll_theo_times = mean_std
        self.pk = mean_std
        self.nbr_train = mean_std
        self.week_day = mean_std
        self.current_time = mean_std

        dim = get_nbr_train_types()
        self.train_type = NDMeanStd(np.ones(dim), np.ones(dim))
        self.pr_nbr_passage = {("PR-A",): 1, ("PR-B",): 2, ("PR-C",): 3}

        self.avg_inter_pr_diff_delay = (mean_std, mean_std)
        self.avg_inter_pr_middle_diff_delay = (mean_std, mean_std)

        self.inter_pr = MockInterPrStats()
