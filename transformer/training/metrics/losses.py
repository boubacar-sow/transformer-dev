import math
from typing import Callable

import torch
from torch import nn

Tensor = torch.Tensor


def smooth_sqrt(x: Tensor) -> Tensor:
    threshold = 5 / 20
    x_abs = x.abs()
    x_sgn = x.sgn()
    mask = x_abs <= threshold
    mask_not = ~mask

    result = torch.empty_like(x)
    result[mask] = x[mask] / math.sqrt(threshold)
    result[mask_not] = x_abs[mask_not].sqrt() * x_sgn[mask_not]
    return result


def smooth_sqrt_l1_loss(x: Tensor, y: Tensor) -> Tensor:
    return torch.abs(smooth_sqrt(x) - smooth_sqrt(y))


LOSSES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "l2": nn.MSELoss(reduction="none"),
    "l1": nn.L1Loss(reduction="none"),
    "smooth_l1": nn.SmoothL1Loss(reduction="none", beta=5 / 20),
    "smooth_sqrt_l1": smooth_sqrt_l1_loss,
}
