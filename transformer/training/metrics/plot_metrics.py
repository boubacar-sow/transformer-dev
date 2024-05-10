from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from transformer.inputs.embeddings.embeddings import Embedding
from transformer.training.modules import TransformerLightningModule
from transformer.utils.config import get_config

cfg, paths = get_config()


def plot_error_map(
    metrics_per_pr: dict[tuple[str, ...], dict[str, float]],
    model: TransformerLightningModule,
    geographic_embedding: Embedding,
) -> None:
    """
    Create and log to MLFlow maps showing the model's error at each PR.
    Maps are created for each prediction (transformer vs translation) / metric / train_num_filter.
    """
    mlflow_run_id = model.logger.run_id

    for metric in cfg.training.test_step.metrics:
        for prediction in ["translation", "transformer"]:
            for train_num_filter in cfg.training.test_step.train_num_filters:
                coords = []
                metrics = []
                counts = []
                for pr_id in geographic_embedding.key_to_emb:
                    key = (metric, prediction, train_num_filter, pr_id)
                    if key in metrics_per_pr:
                        coords.append(geographic_embedding[pr_id].cpu().numpy())
                        metrics.append(metrics_per_pr[key]["mean"])
                        counts.append(metrics_per_pr[key]["count"])

                ax: Axes
                fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
                fig.suptitle(f"Error map for method={prediction}, metric={metric}, train_num_filter={train_num_filter}")
                coords_array = np.array(coords)
                counts_array = np.array(counts)

                vmax = np.quantile(metrics, q=0.9)
                heatmap = ax.scatter(
                    coords_array[:, 0],
                    coords_array[:, 1],
                    s=10 * np.sqrt(counts_array) / np.sqrt(counts_array.max()),
                    c=metrics,
                    vmin=0,
                    vmax=vmax,  # type: ignore[arg-type]
                    cmap="jet",
                )

                fig.colorbar(heatmap)
                fig.tight_layout()

                model.logger.experiment.log_figure(
                    run_id=mlflow_run_id,
                    figure=fig,
                    artifact_file=f"figures/error_map_{metric}_{prediction}_{train_num_filter}.png",
                )


def plot_error_timeline(
    metrics_per_day: dict[tuple[str, ...], dict[str, float]],
    model: TransformerLightningModule,
) -> None:
    """
    Create and log to MLFlow plots showing the model's average error at each day in the test set.
    Maps are created for each prediction (transformer vs translation) / metric / train_num_filter.
    """
    mlflow_run_id = model.logger.run_id

    days = []
    for key in metrics_per_day.keys():
        days.append(key[-1])
    days = sorted(set(days))  # sort days and remove doublons
    days_datetime = [datetime.strptime(day, "%Y-%m-%d") for day in days]

    for metric in cfg.training.test_step.metrics:
        for train_num_filter in cfg.training.test_step.train_num_filters:
            fig, ax = plt.subplots(figsize=(7, 2.5), dpi=200)
            for prediction in ["translation", "transformer"]:
                metrics = []
                for day in days:
                    metrics.append(metrics_per_day[(metric, train_num_filter, prediction, day)]["mean"])

                ax.plot(days_datetime, metrics, marker="o", label=prediction)

            fig.legend()
            fig.tight_layout()

            model.logger.experiment.log_figure(
                run_id=mlflow_run_id,
                figure=fig,
                artifact_file=f"figures/error_timeline_{metric}_{train_num_filter}.png",
            )
