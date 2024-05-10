import json
from collections import defaultdict

import lightning.pytorch as pl
import torch
from mlflow import MlflowClient  # type: ignore
from torch import Tensor

from transformer.inputs.data_structures import SnapshotBatch
from transformer.inputs.utils.train_num_filters import TRAIN_NUM_FILTERS
from transformer.training.metrics.metrics import BASE_METRICS
from transformer.utils.config import get_config

cfg, _ = get_config()


def compute_test_metrics_on_batch(  # noqa: C901
    snapshot_batch: SnapshotBatch,
    predictions: list[Tensor],
    ground_truth: Tensor,
    prediction_names: list[str],
    device: torch.device,
) -> dict[str, dict[tuple[str, ...], dict[str, float]]]:
    """
    Function called within trainer.test(). Compute and logs the test metrics.

    Three criteria:
    - train num filters
    - delay filters (i.e., filter the PRs in the future depending on their observed delay)
    - times filters (i.e., filter the PRs in the future depending on their scheduled time)
    """

    train_num_filters = {key: TRAIN_NUM_FILTERS[key] for key in cfg.training.test_step.train_num_filters}
    delay_bins = cfg.training.test_step.delay_bins
    delay_bins_labels = _bins_labels(delay_bins)
    times_bins = cfg.training.test_step.horizon_bins
    times_bins_labels = _bins_labels(times_bins)

    delay_indices = torch.bucketize(
        input=snapshot_batch.foll_delays,  # type: ignore
        boundaries=torch.Tensor(delay_bins).to(device),
    )
    times_indices = torch.bucketize(
        input=snapshot_batch.foll_theo_times,  # type: ignore
        boundaries=torch.Tensor(times_bins).to(device),
    )

    arrival_mask = snapshot_batch.obs_mask  # entries with the "post-arrival" status

    # First, compute the value of each test metric over the whole batch
    metric_values = {}
    for metric in cfg.training.test_step.metrics:
        for pred, pred_name in zip(predictions, prediction_names):
            metric_func = BASE_METRICS[metric]
            metric_values[(metric, pred_name)] = metric_func(pred, ground_truth).cpu().numpy()

    # Then, aggregate metric values per PR, lateness, time horizon and day
    metrics_list: dict[str, dict[tuple[str, ...], list[float]]] = {
        "per_pr": defaultdict(list),
        "per_delay": defaultdict(list),
        "per_theo_times": defaultdict(list),
        "per_day": defaultdict(list),
    }  # TODO add per_train_type

    for metric in cfg.training.test_step.metrics:  # for each metric
        for pred_name in prediction_names:  # for each prediction to evaluate (transformer, translation, ...)
            for train_num_filter in train_num_filters:  # for each filter over the train numbers
                for i, snapshot_foll_prs in enumerate(snapshot_batch.foll_prs_cich):  # type: ignore # for each shapshot in the batch
                    for j, train_foll_prs in enumerate(snapshot_foll_prs):  # for each train in the snapshot
                        if not train_num_filters[train_num_filter](snapshot_batch.train_num[i][j]):  # type: ignore
                            continue

                        for k, foll_pr in enumerate(train_foll_prs):  # for each PR in the train's itinerary
                            if not arrival_mask[j, i, k]:
                                break  # if the train has arrived, break the loop

                            value = metric_values[(metric, pred_name)][j, i, k]

                            delay_bin = delay_indices[j, i, k]
                            times_bin = times_indices[j, i, k]

                            metrics_list["per_pr"][(metric, pred_name, train_num_filter, foll_pr)].append(value)
                            metrics_list["per_delay"][
                                (metric, train_num_filter, pred_name, delay_bins_labels[delay_bin])
                            ].append(value)
                            metrics_list["per_theo_times"][
                                (metric, train_num_filter, pred_name, times_bins_labels[times_bin])
                            ].append(value)
                            metrics_list["per_day"][
                                (metric, train_num_filter, pred_name, snapshot_batch.day[i])
                            ].append(value)

    batch_metrics = {}
    for key in metrics_list:
        batch_metrics[key] = {k: {"sum": sum(v), "count": len(v)} for k, v in metrics_list[key].items()}

    return batch_metrics


def log_test_metrics(model: pl.LightningModule, metrics: dict[str, dict[tuple[str, ...], dict[str, float]]]) -> None:
    """
    For each metric (mse, mae), each prediction (transformer, translation),
    each train num filter, log the test metrics :
    - averaged overall
    - for each delay bin
    - for each time horizon bin
    """

    mlflow_client: MlflowClient = model.logger.experiment  # type: ignore
    mlflow_run_id = model.logger.run_id  # type: ignore

    # First, log detailed metrics as JSON artifacts

    for key in metrics.keys():
        metrics_for_json = {_get_metric_name(k): v for k, v in metrics[key].items()}
        metrics_json = json.dumps(metrics_for_json, indent=4)
        mlflow_client.log_text(run_id=mlflow_run_id, text=metrics_json, artifact_file=f"metrics/{key}.json")

    # Then, log global average metrics as MLFlow metrics

    # global metrics over all latenesses / time horizons / PR / day
    # we compute them by aggregating the means by lateness, weighting them with their count
    global_sum = defaultdict(float)  # type: dict[tuple[str, ...], float]
    global_count = defaultdict(float)  # type: dict[tuple[str, ...], float]

    for delay_key in metrics["per_delay"].keys():
        mean = metrics["per_delay"][delay_key]["mean"]
        count = metrics["per_delay"][delay_key]["count"]

        mlflow_key = _get_metric_name(delay_key[:-1] + ("delay", delay_key[-1]))
        model.log(name=mlflow_key, value=mean)

        global_sum[delay_key[:-1]] += mean * count
        global_count[delay_key[:-1]] += count

    for times_key in metrics["per_theo_times"].keys():
        mlflow_key = _get_metric_name(times_key[:-1] + ("horizon", times_key[-1]))
        model.log(name=mlflow_key, value=metrics["per_theo_times"][times_key]["mean"])

    for k in global_sum.keys():
        global_mean = global_sum[k] / global_count[k]
        mlflow_key = _get_metric_name(k)
        model.log(name=mlflow_key, value=global_mean)


def _get_metric_name(x: tuple[str, ...]) -> str:
    return "_".join(x)


def _bins_labels(bins: list[float]) -> list[str]:
    """
    Creates label names for the

    Example:
    Input: [0, 2.5, 6]
    Output: ['-0', '0-2.5', '2.5-6', '6--]
    """
    result = [f"-{bins[0]}"]
    for i in range(len(bins) - 1):
        result.append(f"{bins[i]}-{bins[i+1]}")
    result.append(f"{bins[-1]}-")
    return result
