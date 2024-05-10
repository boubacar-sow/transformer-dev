import lightning.pytorch as pl

from transformer.inputs.data_module import SnapshotDataModule
from transformer.inputs.embeddings.embeddings import Embedding
from transformer.inputs.embeddings.geographic_embedding import compute
from transformer.training.metrics.plot_metrics import plot_error_map, plot_error_timeline
from transformer.training.modules import TransformerLightningModule
from transformer.utils.config import get_config
from transformer.utils.logger import logger

cfg, paths = get_config()


def test_model(model: TransformerLightningModule, data_module: SnapshotDataModule, trainer: pl.Trainer) -> None:
    """
    Iterate over the test set to compute various metrics. Log the result and summary plots to MLFlow.
    """

    logger.info("===Test step===")
    trainer.test(model=model, dataloaders=data_module.test_dataloader())
    # predictions = trainer.predict(model=model, dataloaders=data_module.predict_dataloader())
    # predictions_all = pd.concat(predictions)
    # save_csv("generated_data_prd/predictions.csv", predictions_all)

    if cfg.training.mlflow.use_mlflow:
        geographic_embedding = Embedding(
            compute(cfg.embeddings.geographic_embedding),
            is_pr_embedding=True,
            normalize_embedding=False,
        )  # latitude/longitude PR coordinates

        plot_error_map(model.metrics["per_pr"], model, geographic_embedding)
        plot_error_timeline(model.metrics["per_day"], model)
