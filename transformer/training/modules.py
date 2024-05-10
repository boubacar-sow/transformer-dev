"""
Defines modules and subparts of the transformer model
"""

import time
from collections import defaultdict
from typing import Any, Mapping, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger

from transformer.inputs.data_loading import aggregate_data_batch
from transformer.inputs.data_structures import Snapshot, SnapshotBatch
from transformer.inputs.embeddings.embeddings import Embedding
from transformer.inputs.resources.train_type import get_nbr_train_types
from transformer.inputs.snapshots.statistics import AllStats
from transformer.inputs.snapshots.time_transform import TimeTransform
from transformer.training.metrics.compute_metrics import compute_test_metrics_on_batch, log_test_metrics
from transformer.training.metrics.losses import LOSSES
from transformer.utils.config import get_config
from transformer.utils.misc import log_progress

cfg, paths = get_config()

Tensor = torch.Tensor


def compute_input_dim() -> int:
    """
    Computes the dimension of the feature vectors given as inputs to the
    neural network
    """

    n_prev = cfg.model.n_prev
    n_foll = cfg.model.n_foll
    use_inputs = cfg.model.use_inputs
    cfg_emb = cfg.embeddings

    input_dim: int = (
        0
        + n_prev * 2  # delays and times
        + n_foll
        + use_inputs.train_type * get_nbr_train_types()
        + use_inputs.nbr_train_now
        + use_inputs.year_day * 2 * use_inputs.year_day_dim
        + use_inputs.week_day
        + use_inputs.current_time
        + use_inputs.diff_delay * 2 * (n_prev + n_foll - 1)
        + use_inputs.diff_delay * 2 * (n_prev + n_foll - 1)
        + use_inputs.train_nextdesserte_embedding * cfg_emb.nextdesserte_embedding.train_dim
        + use_inputs.pr_nextdesserte_embedding * cfg_emb.nextdesserte_embedding.pr_dim * (n_prev + n_foll)
        + use_inputs.lignepk_embedding * (cfg_emb.lignepk_embedding.ligne_dim + 1) * (n_prev + n_foll)  # +1 for pk
        + use_inputs.laplacian_embedding * cfg_emb.laplacian_embedding.dim * (n_prev + n_foll)
        + use_inputs.node2vec_embedding * cfg_emb.node2vec_embedding.dim * (n_prev + n_foll)
        + use_inputs.geographic_embedding * cfg_emb.geographic_embedding.dim * (n_prev + n_foll)
        + use_inputs.random_embedding * cfg_emb.random_embedding.dim * (n_prev + n_foll)
    )

    return input_dim


class TransformerModel(nn.Module):
    # TODO was more heads experimented ? 2 seems quite
    # low given the high dimensional input
    def __init__(self, all_stats: AllStats, embeddings: dict[str, Embedding]):
        super().__init__()

        self.all_stats = all_stats
        self.time_transform = TimeTransform(all_stats)
        self.embeddings = embeddings

        cfg_model = cfg.model

        input_dim = compute_input_dim()
        self.pre_fc = nn.Linear(input_dim, cfg_model.model_dim)
        self.fc_norm = nn.LayerNorm(cfg_model.model_dim)

        encoder_norm = nn.LayerNorm(cfg_model.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg_model.model_dim,
            nhead=cfg_model.nhead,
            dim_feedforward=cfg_model.model_dim,
            dropout=cfg.training.dropout,
        )

        self.transformers = nn.TransformerEncoder(encoder_layer, cfg_model.depth, norm=encoder_norm)  # type: ignore

        for name, embedding in embeddings.items():
            if embedding.update_during_training:
                self.register_parameter(name, embedding.emb.weight)

        self.out_fc = nn.Linear(cfg_model.model_dim, cfg_model.n_foll)

        with torch.no_grad():
            self.out_fc.weight *= 0.0  # type: ignore
            self.out_fc.bias *= 0.0  # type: ignore

        self.relu = nn.LeakyReLU()

    def forward(self, x: Tensor, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        Note: key_mask[i,j]=True means that the value is ignored by the TransformerEncoder layers
        For NLP, it would be "ignore word j in sentence i", here it is "ignore train j in snapshot i".
        """
        y: Tensor = self.pre_fc(x)
        y = self.fc_norm(y)
        y = self.transformers(y, src_key_padding_mask=key_mask)
        y = self.out_fc(y)
        return y

    def process_output(self, output: Tensor, snapshot_batch: SnapshotBatch) -> tuple[Tensor, Tensor]:
        """
        Process the transformer neural network output to ensure that no prediction is made in the past

        Returns:
            output: output of the neural network, thresholded to stay in the future
            translation_shift: can be either zero or the translation, depending on whether the model allows
                               predictions in the past
        """
        foll_theo_times = snapshot_batch.foll_theo_times
        translation = snapshot_batch.translation_delays

        if cfg.data.preprocessing.remove_translation_in_outputs:
            translation_shift = torch.zeros_like(snapshot_batch.y)
        else:
            translation_shift = translation  # type: ignore[assignment]

        if cfg.training.postprocessing_forget_negative_preds:
            # force the predicted arrival times to be in the future (i.e., >= 0 since t=0 is now)
            output_time = self.time_transform.nn_to_delay(output, mod=torch) + translation_shift + foll_theo_times
            output_time = self.relu(output_time)
            output = self.time_transform.delay_to_nn(output_time - translation_shift - foll_theo_times, mod=torch)

        return output, translation_shift


class TransformerLightningModule(pl.LightningModule):
    metrics: dict[str, dict[tuple[str, ...], dict[str, float]]]
    logger: MLFlowLogger

    def __init__(self, model: TransformerModel):
        super().__init__()

        self.time_transform = model.time_transform
        self.model = model
        self.loss_fn = LOSSES[cfg.training.loss]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.training.learning_rate)

        if cfg.training.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.8, patience=10, threshold=1 / 10, cooldown=10, verbose=True, min_lr=1e-6
            )
            # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            #     optim, lambda x: .99 if x >= 400 else 1.)
            # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            #     optim, 1e-3, 8e-3, step_size_up=100)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
        else:
            return {"optimizer": optimizer}

    # +----------------------------------+
    # | Train, Validation and Test steps |
    # +----------------------------------+

    def training_step(self, data_batch: list[Snapshot], batch_idx: int) -> Tensor:
        snapshot_batch = aggregate_data_batch(data_batch, self.model.embeddings)

        out = self.model(snapshot_batch.x, snapshot_batch.key_mask)
        out, _ = self.model.process_output(out, snapshot_batch)

        # Don't count observations after train arrivals and NaN observations
        loss = self.loss_fn(out[snapshot_batch.obs_mask], snapshot_batch.y[snapshot_batch.obs_mask])
        loss = loss.sum() / snapshot_batch.obs_mask.sum()

        self.log("train_loss", loss, batch_size=snapshot_batch.x.shape[1], on_step=True, on_epoch=True)
        return loss

    def validation_step(self, data_batch: list[Snapshot], batch_idx: int) -> Tensor:
        snapshot_batch = aggregate_data_batch(data_batch, self.model.embeddings)

        out = self.model(snapshot_batch.x, snapshot_batch.key_mask)
        out, translation_shift = self.model.process_output(out, snapshot_batch)

        mask_sum = snapshot_batch.obs_mask.sum()

        loss = self.loss_fn(out, snapshot_batch.y)
        # Don't count observations after train arrivals and NaN observations
        loss = loss[snapshot_batch.obs_mask].sum() / mask_sum

        out_minutes = (self.time_transform.nn_to_delay(out, mod=torch) + translation_shift) * snapshot_batch.obs_mask
        y_minutes = (
            self.time_transform.nn_to_delay(snapshot_batch.y, mod=torch) + translation_shift
        ) * snapshot_batch.obs_mask  # corresponds to foll_delays
        l1error = (y_minutes - out_minutes)[snapshot_batch.obs_mask].abs().sum() / mask_sum

        self.log("val_loss", loss, batch_size=snapshot_batch.x.shape[1], on_step=False, on_epoch=True)
        self.log("val_l1error", l1error, batch_size=snapshot_batch.x.shape[1], on_step=False, on_epoch=True)
        return loss

    def test_step(self, data_batch: list[Snapshot], batch_idx: int) -> Tensor:
        snapshot_batch = aggregate_data_batch(data_batch, self.model.embeddings)

        out = self.model(snapshot_batch.x, snapshot_batch.key_mask)
        out, translation_shift = self.model.process_output(out, snapshot_batch)

        mask_sum = snapshot_batch.obs_mask.sum()

        loss = self.loss_fn(out, snapshot_batch.y)
        # Don't count observations after train arrivals and NaN observations
        loss = loss[snapshot_batch.obs_mask].sum() / mask_sum

        out_minutes = (self.time_transform.nn_to_delay(out, mod=torch) + translation_shift) * snapshot_batch.obs_mask
        y_minutes = (
            self.time_transform.nn_to_delay(snapshot_batch.y, mod=torch) + translation_shift
        ) * snapshot_batch.obs_mask  # corresponds to foll_delays

        self.log("test_loss", loss, batch_size=snapshot_batch.x.shape[1], on_step=False, on_epoch=True)

        batch_metrics = compute_test_metrics_on_batch(
            snapshot_batch=snapshot_batch,
            ground_truth=y_minutes,
            predictions=[out_minutes, snapshot_batch.translation_delays],  # type: ignore[list-item]
            prediction_names=["transformer", "translation"],
            device=self.device,
        )

        self.update_metrics(batch_metrics)

        return loss

    def on_test_epoch_start(self) -> None:
        self.metrics = defaultdict(
            lambda: defaultdict(lambda: {"sum": 0, "count": 0})
        )  # example: {"per_pr": {"key_1": {"sum": 0, "count": 0}, "key_2": {"sum": 0, "count": 0}}}
        self.time_start = time.time()

    def update_metrics(self, batch_metrics: dict[str, dict[tuple[str, ...], dict[str, float]]]) -> None:
        """
        Add the metric updates (new sums and counts for each keys) to the current metrics values.
        """

        for key in batch_metrics.keys():
            for fine_grain_key in batch_metrics[key].keys():
                self.metrics[key][fine_grain_key]["sum"] += batch_metrics[key][fine_grain_key]["sum"]
                self.metrics[key][fine_grain_key]["count"] += batch_metrics[key][fine_grain_key]["count"]

    def on_test_epoch_end(self) -> None:
        # Compute metrics means from sum and count
        for key in self.metrics.keys():
            for k in self.metrics[key].keys():
                self.metrics[key][k] = {
                    "mean": self.metrics[key][k]["sum"] / self.metrics[key][k]["count"],
                    "count": self.metrics[key][k]["count"],
                    "sum": self.metrics[key][k]["sum"],
                }

        # Log metrics to MLFlow
        if cfg.training.mlflow.use_mlflow:
            log_test_metrics(self, self.metrics)

    # +------------------+
    # | Progress logging |
    # +------------------+

    def on_train_epoch_start(self) -> None:
        self.time_start = time.time()

    def on_validation_epoch_start(self) -> None:
        self.time_start = time.time()

    def on_train_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int) -> None:
        if batch_idx % 10 == 9:
            for emb in self.model.embeddings.values():
                if emb.is_pr_embedding and emb.update_during_training:
                    emb.update_missing_prs()

        log_progress(
            step=f"Training epoch {self.current_epoch}",
            batch_idx=batch_idx + 1,
            batch_per_epoch=int(self.trainer.num_training_batches),
            epoch_time_seconds=int(time.time() - self.time_start),
        )

    def on_validation_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        log_progress(
            step=f"Validation epoch {self.current_epoch}",
            batch_idx=batch_idx + 1,
            batch_per_epoch=int(self.trainer.num_val_batches[0]),
            epoch_time_seconds=int(time.time() - self.time_start),
        )

    def on_test_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        log_progress(
            step="Test step",
            batch_idx=batch_idx + 1,
            batch_per_epoch=int(self.trainer.num_test_batches[0]),
            epoch_time_seconds=int(time.time() - self.time_start),
        )
