from typing import Callable

from torch import Tensor


def mae_metric(pred: Tensor, y: Tensor) -> Tensor:
    return (pred - y).abs()  # type: ignore


def mse_metric(pred: Tensor, y: Tensor) -> Tensor:
    return (pred - y) ** 2  # type: ignore


BASE_METRICS: dict[str, Callable[[Tensor, Tensor], Tensor]]
BASE_METRICS = {"mae": mae_metric, "mse": mse_metric}
