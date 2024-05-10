"""
Script that trains the transformer model
"""

from typing import Optional

import lightning.pytorch as pl
import optuna
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import SimpleProfiler
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from transformer.inputs.data_module import SnapshotDataModule
from transformer.inputs.embeddings.embeddings import load_embeddings
from transformer.inputs.snapshots.statistics import load_statistics
from transformer.training.modules import (
    TransformerLightningModule,
    TransformerModel,
    compute_input_dim,
)
from transformer.utils.config import get_config
from transformer.utils.logger import logger
from transformer.utils.mlflow import load_checkpoint_from_mlflow, log_config_to_mlflow

cfg, paths = get_config()


def get_lightning_objects(
    run_id: str = "",
    trial: Optional[optuna.Trial] = None,
    checkpoint: Optional[str] = None,
) -> tuple[TransformerLightningModule, SnapshotDataModule, pl.Trainer]:
    """
    Builds and returns the three main objects of the pytorch lightning training/testing pipeline:
    - the LightningModule (which contains the implementation of the train/val/test steps)
    - the DataModule (which generates train, validation and test data loaders)
    - the Trainer (providing a scikit-learn-like .fit function which performs the training)

    Args:
        run_id: MLFlow run id (only used when cfg.training.mlflow.use_mlflow is True)
        trial: optional Optuna trial, used only for hyperparameter tuning.
    """

    logger.info("Defining Pytorch Lightning objects: LightningModule, DataModule, Trainer")

    # +------------------------+
    # | MODEL AND DATA LOADING |
    # +------------------------+

    all_stats = load_statistics()
    embeddings = load_embeddings()
    model = TransformerModel(all_stats, embeddings)

    if checkpoint is None:
        lightning_module = TransformerLightningModule(model=model)
    else:
        lightning_module = TransformerLightningModule.load_from_checkpoint(checkpoint, model=model)

    data_module = SnapshotDataModule(model=model, num_workers=cfg.training.data_loader_num_workers)

    # +----------+
    # | TRAINING |
    # +----------+

    trainer_callbacks = []
    if trial is not None:  # optuna trial pruning
        early_stopping_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        trainer_callbacks.append(early_stopping_callback)

    if cfg.training.mlflow.use_mlflow:
        mlflow_logger = MLFlowLogger(
            run_id=run_id, tracking_uri=cfg.training.mlflow.tracking_uri, log_model=cfg.training.mlflow.log_model
        )

    profiler = SimpleProfiler(dirpath=paths.logs_profiling, filename="profile")

    trainer = pl.Trainer(
        limit_train_batches=cfg.training.batches_per_epoch,
        log_every_n_steps=1,
        max_epochs=cfg.training.nbr_epoch,
        logger=mlflow_logger if cfg.training.mlflow.use_mlflow else None,
        enable_progress_bar=False,  # TQDM progress bars don't render well in kubernetes logs
        callbacks=trainer_callbacks,
        profiler=profiler,
    )

    return lightning_module, data_module, trainer


def train_model(
    run_id: str = "", warm_start_run_id: Optional[str] = None, trial: Optional[optuna.Trial] = None
) -> tuple[TransformerLightningModule, SnapshotDataModule, pl.Trainer]:
    """
    Args:
        run_id: MLFlow run id (only used when cfg.training.mlflow.use_mlflow is True)
        trial: optional Optuna trial, used only for hyperparameter tuning.
    """

    logger.info("===Model training===")

    checkpoint = load_checkpoint_from_mlflow(warm_start_run_id) if warm_start_run_id is not None else None

    lightning_module, data_module, trainer = get_lightning_objects(run_id=run_id, trial=trial)

    if warm_start_run_id is None or run_id != warm_start_run_id:
        log_config_to_mlflow()

    logger.info(f"Input dimension: {compute_input_dim()}, model dimension: {cfg.model.model_dim}.")
    logger.info(f"Total parameters: {sum(p.numel() for p in lightning_module.parameters())}")
    logger.info(f"Starting training (checkpoint: {checkpoint}).")

    trainer.fit(
        model=lightning_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=checkpoint,
    )

    # if cfg.training.mlflow.use_mlflow and cfg.training.mlflow.log_model:
    #     mlflow.pytorch.log_model(pytorch_model=lightning_module.model, artifact_path="model")

    return lightning_module, data_module, trainer
