from types import ModuleType
from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np

from transformer.inputs.data_structures import MeanStd
from transformer.utils.config import get_config

if TYPE_CHECKING:
    from transformer.inputs.snapshots.statistics import AllStats

cfg, paths = get_config()

T = TypeVar("T")


def _identity(x: T, mod: ModuleType = np) -> T:
    return x


def _normalize(x: T, mean_std: MeanStd) -> T:
    return (x - mean_std.mean) / mean_std.std  # type: ignore


def _unnormalize(x: T, mean_std: MeanStd) -> T:
    return x * mean_std.std + mean_std.mean  # type: ignore


class TimeTransform:
    """
    Usage:

    time_transform = TimeTransform(all_stats=all_stats)

    x = time_transform.time_to_nn(time) # formatted for the transformer input
    time = time_transform.nn_to_time(x) # time in minutes

    z = time_transform.delay_to_nn(delay) # formatted for the transformer output
    delay = time_transform.nn_to_delay(z) # delay in minutes

    For torch Tensors, use mod=torch.
    """

    def __init__(self, all_stats: Optional["AllStats"] = None):
        if all_stats is not None:
            self.delays = all_stats.delays
            self.prev_foll_theo_times = {"prev": all_stats.prev_theo_times, "foll": all_stats.foll_theo_times}

        self.preprocess = cfg.data.preprocessing
        if self.preprocess.transform_delay_sqrt == "sqrt" or self.preprocess.transform_delay_sqrt is True:
            self._delay_to_nn_func = TimeTransform._sqrt_func
            self._nn_to_delay_func = TimeTransform._sqrt_func_inverse
        elif self.preprocess.transform_delay_sqrt == "log":
            self._delay_to_nn_func = TimeTransform._log_func
            self._nn_to_delay_func = TimeTransform._log_func_inverse
        else:
            self._delay_to_nn_func = _identity
            self._nn_to_delay_func = _identity

        if self.preprocess.transform_times_sqrt == "sqrt" or self.preprocess.transform_times_sqrt is True:
            self._times_to_nn_func = TimeTransform._sqrt_func
            self._nn_to_times_func = TimeTransform._sqrt_func_inverse
        elif self.preprocess.transform_times_sqrt == "log":
            self._times_to_nn_func = TimeTransform._log_func
            self._nn_to_times_func = TimeTransform._log_func_inverse
        else:
            self._times_to_nn_func = _identity
            self._nn_to_times_func = _identity

    def times_to_nn(self, x: T, prev_foll: str = "prev", normalize: bool = True, mod: ModuleType = np) -> T:
        x = self._times_to_nn_func(x, mod)
        if normalize and self.preprocess.normalize_times:
            x = _normalize(x, self.prev_foll_theo_times[prev_foll])
        return x

    def nn_to_times(self, x: T, prev_foll: str = "prev", normalize: bool = True, mod: ModuleType = np) -> T:
        if normalize and self.preprocess.normalize_times:
            x = _unnormalize(x, self.prev_foll_theo_times[prev_foll])
        return self._nn_to_times_func(x, mod)

    def delay_to_nn(self, x: T, normalize: bool = True, mod: ModuleType = np) -> T:
        x = self._delay_to_nn_func(x, mod)
        if normalize and self.preprocess.normalize_delay:
            x = _normalize(x, self.delays)
        return x

    def nn_to_delay(self, x: T, normalize: bool = True, mod: ModuleType = np) -> T:
        if normalize and self.preprocess.normalize_delay:
            x = _unnormalize(x, self.delays)
        return self._nn_to_delay_func(x, mod)

    @staticmethod
    def _sqrt_func(x: T, mod: ModuleType = np) -> T:
        return mod.sqrt(mod.abs(x)) * mod.sign(x)  # type: ignore[no-any-return]

    @staticmethod
    def _sqrt_func_inverse(x: T, mod: ModuleType = np) -> T:
        return mod.square(x) * mod.sign(x)  # type: ignore[no-any-return]

    @staticmethod
    def _log_func(x: T, mod: ModuleType = np) -> T:
        return mod.log(1 + mod.abs(x)) * mod.sign(x)  # type: ignore[no-any-return]

    @staticmethod
    def _log_func_inverse(x: T, mod: ModuleType = np) -> T:
        return (mod.exp(mod.abs(x)) - 1) * mod.sign(x)  # type: ignore[no-any-return]
