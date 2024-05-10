"""
Contains the SnapshotDataset and SnapshotDataModule classes, which are the
main classes to load the train/validation/test data.
"""

import lightning.pytorch as pl
import torch

from transformer.inputs.data_loading import load_snapshot
from transformer.inputs.data_structures import Snapshot
from transformer.inputs.embeddings.embeddings import Embedding
from transformer.inputs.snapshots.snapshots_files import get_snapshots_files
from transformer.inputs.snapshots.statistics import AllStats
from transformer.inputs.snapshots.time_transform import TimeTransform
from transformer.inputs.utils.train_num_filters import TRAIN_NUM_FILTERS, TrainNumFilter
from transformer.training.modules import TransformerModel
from transformer.utils.config import get_config

cfg, paths = get_config()


class SnapshotDataset(torch.utils.data.Dataset[Snapshot]):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(
        self,
        snapshots_files: list[str],
        embeddings: dict[str, Embedding],
        all_stats: AllStats,
        time_transform: TimeTransform,
        train_num_filter: TrainNumFilter,
        return_translation: bool = False,
        return_foll_theo_times: bool = False,
        return_foll_delays: bool = False,
        return_train_type: bool = False,
        return_train_num: bool = False,
        return_pr_cich: bool = False,
        return_day_time: bool = False,
    ):
        """Initialization"""
        self.snapshots_files = snapshots_files
        self.embeddings = embeddings
        self.all_stats = all_stats
        self.time_transform = time_transform
        self.train_num_filter = train_num_filter
        self.return_translation = return_translation
        self.return_foll_theo_times = return_foll_theo_times
        self.return_foll_delays = return_foll_delays
        self.return_train_type = return_train_type
        self.return_train_num = return_train_num
        self.return_pr_cich = return_pr_cich
        self.return_day_time = return_day_time

    def __len__(self) -> int:
        """
        Denotes the total number of samples
        """
        return len(self.snapshots_files)

    def __getitem__(self, index: int) -> Snapshot:
        """
        Generates one sample of data, i.e. one snapshot
        """
        # Select sample
        filename = self.snapshots_files[index]
        return load_snapshot(
            filename,
            self.train_num_filter,
            self.embeddings,
            self.all_stats,
            self.time_transform,
            self.return_translation,
            self.return_foll_theo_times,
            self.return_foll_delays,
            self.return_train_type,
            self.return_train_num,
            self.return_pr_cich,
        )


class SnapshotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model: TransformerModel,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataloader_params = {"num_workers": num_workers, "collate_fn": (lambda x: x)}
        self.all_stats = model.all_stats
        self.time_transform = TimeTransform(model.all_stats)
        self.embeddings = model.embeddings
        self.train_num_filter = TRAIN_NUM_FILTERS[cfg.data.filters.train_num_filter]

        # Define train and test periods
        self.snapshots_files = get_snapshots_files(verbose=True)

    def train_dataloader(self) -> torch.utils.data.DataLoader[Snapshot]:
        cfg_training = cfg.training

        train_set = SnapshotDataset(
            self.snapshots_files.train_snapshots,
            self.embeddings,
            self.all_stats,
            self.time_transform,
            self.train_num_filter,
            return_translation=True,
            return_foll_theo_times=cfg_training.postprocessing_forget_negative_preds,
            return_foll_delays=False,
            return_train_type=False,  # Was used to filter voyageur trains in loss
            return_train_num=False,
        )

        return torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            batch_size=cfg_training.batch_size,
            **self.dataloader_params,  # type: ignore
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[Snapshot]:
        cfg_training = cfg.training

        val_set = SnapshotDataset(
            self.snapshots_files.validation_snapshots,
            self.embeddings,
            self.all_stats,
            self.time_transform,
            self.train_num_filter,
            return_translation=True,  # forget_negatives,
            return_foll_theo_times=True,
            return_foll_delays=True,
            return_train_type=False,  # Was used to filter voyageur trains in loss
            return_train_num=False,
        )

        return torch.utils.data.DataLoader(
            val_set,
            shuffle=False,
            batch_size=cfg_training.batch_size,
            **self.dataloader_params,  # type: ignore
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[Snapshot]:
        cfg_training = cfg.training

        test_set = SnapshotDataset(
            self.snapshots_files.test_snapshots,
            self.embeddings,
            self.all_stats,
            self.time_transform,
            self.train_num_filter,
            return_translation=True,  # forget_negatives,
            return_foll_theo_times=True,
            return_foll_delays=True,
            return_train_type=False,  # Was used to filter voyageur trains in loss
            return_train_num=True,
            return_pr_cich=True,
        )

        return torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            batch_size=cfg_training.batch_size,
            **self.dataloader_params,  # type: ignore
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader[Snapshot]:
        cfg_training = cfg.training

        pred_set = SnapshotDataset(
            self.snapshots_files.test_snapshots,
            self.embeddings,
            self.all_stats,
            self.time_transform,
            self.train_num_filter,
            return_translation=True,  # forget_negatives,
            return_foll_theo_times=True,
            return_foll_delays=True,
            return_train_type=False,  # Was used to filter voyageur trains in loss
            return_train_num=True,
            return_pr_cich=True,
            return_day_time=True,
        )

        return torch.utils.data.DataLoader(
            pred_set,
            shuffle=False,
            batch_size=cfg_training.batch_size,
            **self.dataloader_params,  # type: ignore
        )
