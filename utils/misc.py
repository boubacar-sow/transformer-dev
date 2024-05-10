import os
import random
from typing import Any, Iterable

import numpy as np
import psutil
import torch

from transformer.inputs.data_structures import MeanStd
from transformer.utils.config import get_config, hash_config
from transformer.utils.logger import logger

cfg, paths = get_config()


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_list(data: list[list[Any]]) -> list[Any]:
    """
    Contatenates a list of lists
    """
    res = []
    for d in data:
        res += d
    return res


def weighted_mean_std(values: list[float], weights: list[float]) -> MeanStd:
    array = np.array(values)
    mean = np.average(array, weights=weights)
    std = np.average((array - mean) ** 2, weights=weights) ** 0.5
    return MeanStd(mean=mean, std=std)


def weighted_dict_mean_std(
    values: dict[Any, MeanStd], weights: dict[Any, float], keys: Iterable[Any]
) -> tuple[MeanStd, MeanStd]:
    """
    Returns:
    - mean: The (mean of means) and the (std of means)
    - std: The (mean of stds) and the (std of means)
    """
    values_mean = [values[k].mean for k in keys]
    values_std = [values[k].std for k in keys]
    weights_ = [weights[k] for k in keys]
    mean = weighted_mean_std(values_mean, weights_)
    std = weighted_mean_std(values_std, weights_)
    return mean, std


def reverse_list_index(tab: list[Any], x: Any) -> int:
    """Returns the index of the last occurence of x in tab"""
    return len(tab) - 1 - tab[::-1].index(x)


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def print_ram(step: str | None = None) -> None:
    process_id = os.getpid()
    process = psutil.Process(process_id)
    if step is None:
        step = ""
    else:
        step = f"step: {step}, "

    logger.info(
        f"Ressources ({step}process {process_id}): {int(process.memory_info().rss / (1024 * 1024))} Mo RAM"
        + f"   |   {process.cpu_percent()}% CPU"
    )


def log_progress(step: str, batch_idx: int, batch_per_epoch: int, epoch_time_seconds: int) -> None:
    """
    Print progress during a train/validation/test epoch (replaces a TQDM bar, which renders horribly bad in logs)
    """
    hour = epoch_time_seconds // 3600
    minute = (epoch_time_seconds % 3600) // 60
    second = epoch_time_seconds % 60
    elapsed_time = f"{minute:02d}:{second:02d}" if hour == 0 else f"{hour:02d}:{minute:02d}:{second:02d}"

    estimated_total_epoch_time = int(epoch_time_seconds * batch_per_epoch / batch_idx)
    total_hour = estimated_total_epoch_time // 3600
    total_minute = (estimated_total_epoch_time % 3600) // 60
    total_second = estimated_total_epoch_time % 60
    total_time = (
        f"{total_minute:02d}:{total_second:02d}"
        if total_hour == 0
        else f"{total_hour:02d}:{total_minute:02d}:{total_second:02d}"
    )

    logger.info(f"{step} ({elapsed_time} / {total_time}): {batch_idx}/{batch_per_epoch}")


def get_data_hash() -> str:
    # logger.warning("Using hard-coded data config.")
    # return "892e2fef24f8aebbf02cca2d05b725d3daffa6dafefc935b7fd5e896e7ae8142ef5b8c971e99f0170db0cbcf8fcc24d76139bb93c7a43ba40f44455c0e4dd406"  # noqa: E501
    return hash_config(cfg.data)
