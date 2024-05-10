from collections import Counter
from typing import Callable, Iterable, Optional

import numpy as np

from transformer.inputs.data_structures import MeanStd


def identity(x: float) -> float:
    return x


def to_bin(x: float, bin_size: float) -> float:
    return int(x / bin_size) * bin_size + bin_size / 2


class StatsCounter(Counter[float]):
    """
    If the observations to count are [10, 10, 10, 11, 14], stores information in the form {10:3, 11:1, 14:1}
    Then you can access methods .mean(), .std(), etc to compute mean, std of [10, 10, 10, 11, 14]

    StatsCounter thus represents the distribution of the observations encountered so far.
    """

    def __init__(
        self,
        iterable: Optional[Iterable[float] | dict[float, int]] = None,
        bin_size: Optional[float] = None,
    ) -> None:
        """
        iterable:
            - if given an iterable, will add the floats in it to the Counter
            - if given a dict[float, int], will set it as the internal counter state
        bin_size: if not None, the incoming data will be binned. This option can
            be used to limit the size of the dictionary.
        """
        self.bin_size = bin_size

        if iterable is not None:
            if isinstance(iterable, dict):
                super().__init__(iterable)
            else:
                self.add(iterable)

    def add(self, x: Iterable[float] | Iterable[int] | float | int) -> None:
        if self.bin_size is not None:
            if isinstance(x, float) or isinstance(x, int):
                x = to_bin(x, self.bin_size)
            else:
                x = (to_bin(y, self.bin_size) for y in x)
        if not hasattr(x, "__iter__"):
            self[x] += 1
        else:
            self += Counter(x)

    def min(self) -> float:
        # return min(0, min(self.keys()))
        return min(self)

    def max(self) -> float:
        # return max(0, max(self.keys()))
        return max(self)

    def minimum(self, threshold: float, in_place: bool = False) -> "StatsCounter":
        """
        remove values under threshold, and increase output[threshold] so that output.nbr() == input.nbr()
        """
        if in_place:
            raise RuntimeError("Unimplemented")
        res = self.__class__({k: v for k, v in self.items() if (k >= threshold)})
        nbr = self.nbr() - res.nbr()
        if nbr > 0:
            res[threshold] += nbr
        return res

    def maximum(self, threshold: float, in_place: bool = False) -> "StatsCounter":
        """
        remove values over threshold, and increase output[threshold] so that output.nbr() == input.nbr()
        """
        if in_place:
            raise RuntimeError("Unimplemented")
        res = self.__class__({k: v for k, v in self.items() if (k <= threshold)})
        nbr = self.nbr() - res.nbr()
        if nbr > 0:
            res[threshold] += nbr
        return res

    def filter(
        self,
        min_key: Optional[float] = None,
        max_key: Optional[float] = None,
    ) -> "StatsCounter":
        if min_key is None:
            if max_key is None:
                return StatsCounter(self)
            return self.__class__({k: v for k, v in self.items() if (k < max_key)})
        if max_key is None:
            return self.__class__({k: v for k, v in self.items() if (min_key <= k)})
        return self.__class__({k: v for k, v in self.items() if (min_key <= k < max_key)})

    def mean(self, transform: Optional[Callable[[float], float]] = None) -> float:
        if transform is None:
            transform = identity
        s = 0.0
        n = 0
        for k, v in self.items():
            k = transform(k)
            s += k * v
            n += v
        return float(s) / max(1, n)

    def std(self, transform: Optional[Callable[[float], float]] = None) -> float:
        if transform is None:
            transform = identity
        m = self.mean(transform=transform)
        s = 0.0
        n = 0
        for k, v in self.items():
            k = transform(k)
            s += v * (k - m) ** 2
            n += v
        return (float(s) / max(1, n)) ** 0.5  # type: ignore

    def median(self, transform: Optional[Callable[[float], float]] = None) -> float:
        if transform is None:
            transform = identity
        n = self.nbr()
        m = (n + 1) / 2
        sn = 0
        return_next = None
        keys = sorted(self.keys(), key=transform)
        for k in keys:
            if return_next is not None:
                return (transform(return_next) + transform(k)) / 2
            sn += self[k]
            if sn >= m:
                return transform(k)
            if sn >= m - 0.75:
                return_next = transform(k)
        raise Exception("This should not be reached.")

    def has_float(self) -> bool:
        for k in self.keys():
            if int(k) != k:
                return True
        return False

    def to_pos_list(self) -> list[int]:
        sc = self.get_group_stats_counter(group=1) if self.has_float() else self
        length: int = max(sc.keys()) + 1  # type: ignore
        res = [0] * length
        for k, v in sc.items():
            if k >= 0:
                res[k] = v  # type: ignore
        return res

    def to_cumsum_pos_list(self) -> list[int]:
        return list(np.cumsum(self.to_pos_list()))

    def to_neg_list(self) -> list[int]:
        sc = self.get_group_stats_counter(group=1) if self.has_float() else self
        length = -min(sc.keys()) + 1
        res = [0] * length  # type: ignore
        for k, v in sc.items():
            if k <= 0:
                res[k - 1] = v  # type: ignore
        return res

    def to_cumsum_neg_list(self) -> list[int]:
        return list(np.cumsum(self.to_neg_list()))

    def to_list(self) -> list[int]:
        return self.to_neg_list()[:-1] + self.to_pos_list()

    # def to_list_unanchored(self, xmin: Optional[int] = None, xmax: Optional[int] = None) -> list[int]:
    #     sc = self.get_group_stats_counter(group=1) if self.has_float() else self
    #     if xmin is None:
    #         xmin = int(min(sc.keys()))
    #     if xmax is None:
    #         xmax = int(max(sc.keys())) + 1
    #     length = xmax - xmin
    #     res = [0] * length
    #     for k, v in sc.items():
    #         x = int(k) - xmin
    #         if (x >= 0) and (x < length):
    #             res[x] = v
    #     return res

    # def to_list_xy_unanchored(self, xmin: Optional[int] = None, xmax: Optional[int] = None) -> list[tuple[int, int]]:
    #     sc = self.get_group_stats_counter(group=1) if self.has_float() else self
    #     if xmin is None:
    #         xmin = int(min(sc.keys()))
    #     if xmax is None:
    #         xmax = int(max(sc.keys())) + 1
    #     length = xmax - xmin
    #     res = [(x, int(0)) for x in list(range(xmin, xmax))]
    #     for k, v in sc.items():
    #         x = int(k) - xmin
    #         if (x >= 0) and (x < length):
    #             res[x] = (int(k), v)
    #     return res

    def to_cumsum_list(self) -> list[float]:
        return list(np.cumsum(self.to_list()))

    def percent(self, threshold: float = 5) -> float:
        """
        % of observations below the threshold
        """
        s = 0
        n = 0
        for k, v in self.items():
            if k <= threshold:
                s += v
            n += v
        return 100 * float(s) / max(1, n)

    def nbr(self) -> int:
        """Total number of elements in the counter"""
        return sum(self.values())

    def middle_nbr(self, lower_bound: float = 0.0, upper_bound: float = 1.0) -> float:
        return self.nbr() * (upper_bound - lower_bound)

    def middle_mean(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        transform: Optional[Callable[[float], float]] = None,
    ) -> float:
        """
        mean of the distribution samples between quantiles lower_bound and upper_bound
        """
        if transform is None:
            transform = identity
        nbr = self.nbr()
        items = sorted(self.items(), key=lambda kv: transform(kv[0]))
        lower_bound = nbr * lower_bound
        upper_bound = nbr * upper_bound
        tot = 0
        s = 0.0
        div = 0.0
        for v, n in items:
            v = transform(v)
            crossed_lower_bound = (lower_bound >= tot) and (lower_bound < tot + n)
            crossed_upper_bound = (upper_bound > tot) and (upper_bound <= tot + n)
            if crossed_lower_bound and crossed_upper_bound:
                nn = upper_bound - lower_bound
                s += nn * v
                div += nn
            elif crossed_lower_bound:
                nn = tot + n - lower_bound
                s += nn * v
                div += nn
            elif crossed_upper_bound:
                nn = upper_bound - tot
                s += nn * v
                div += nn
            elif (lower_bound < tot) and (upper_bound > tot + n):
                s += n * v
                div += n
            # print(tot, tot+n, " ", lower_bound, upper_bound, " ", crossed_lower_bound,
            #       crossed_upper_bound, (lower_bound < tot) and (upper_bound > tot + n))
            tot += n
        return s / max(1e-7, div)

    def middle_std(
        self,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        use_middle_mean: bool = True,
        middle_mean_value: Optional[float] = None,
        transform: Optional[Callable[[float], float]] = None,
    ) -> float:
        """
        std of the distribution samples between quantiles lower_bound and upper_bound
        """
        if transform is None:
            transform = identity
        nbr = self.nbr()
        if use_middle_mean:
            if middle_mean_value is None:
                m = self.middle_mean(lower_bound=lower_bound, upper_bound=upper_bound, transform=transform)
            else:
                m = middle_mean_value
        else:
            m = self.mean()
        items = sorted(self.items(), key=lambda kv: transform(kv[0]))
        lower_bound = nbr * lower_bound
        upper_bound = nbr * upper_bound
        tot = 0
        s = 0.0
        div = 0.0
        for v, n in items:
            v = transform(v)
            crossed_lower_bound = (lower_bound >= tot) and (lower_bound < tot + n)
            crossed_upper_bound = (upper_bound > tot) and (upper_bound <= tot + n)
            if crossed_lower_bound and crossed_upper_bound:
                nn = upper_bound - lower_bound
                s += nn * (v - m) ** 2
                div += nn
            elif crossed_lower_bound:
                nn = tot + n - lower_bound
                s += nn * (v - m) ** 2
                div += nn
            elif crossed_upper_bound:
                nn = upper_bound - tot
                s += nn * (v - m) ** 2
                div += nn
            elif (lower_bound < tot) and (upper_bound > tot + n):
                s += n * (v - m) ** 2
                div += n
            # print(tot, tot+n, " ", lower_bound, upper_bound, " ", crossed_lower_bound,
            #       crossed_upper_bound, (lower_bound < tot) and (upper_bound > tot + n))
            tot += n
        return (s / max(1e-7, div)) ** 0.5  # type: ignore

    def mean_std(self, transform: Optional[Callable[[float], float]] = None) -> MeanStd:
        return MeanStd(self.mean(transform=transform), self.std(transform=transform))

    def middle_mean_std(
        self, lower_bound: float = 0.0, upper_bound: float = 1.0, transform: Optional[Callable[[float], float]] = None
    ) -> MeanStd:
        middle_mean = self.middle_mean(lower_bound=lower_bound, upper_bound=upper_bound, transform=transform)
        middle_std = self.middle_std(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            use_middle_mean=True,
            middle_mean_value=middle_mean,
            transform=transform,
        )
        return MeanStd(middle_mean, middle_std)

    def get_group_stats_counter(self, group: int = 1) -> "StatsCounter":
        res = StatsCounter()
        for k, v in self.items():
            res[int(k // group)] += v
        return res

    # def plot_hist(
    #     self, *args: Any, group: Optional[float] = None, show: bool = True, norm: bool = False, **kwargs: Any
    # ) -> None:
    #     import matplotlib.pyplot as plt

    #     if group is None:
    #         xy = self.to_list_xy_unanchored()
    #     else:
    #         xy_ = StatsCounter()
    #         for k, v in self.items():
    #             xy_[(k // group)] += v
    #         xy = xy_.to_list_xy_unanchored()
    #     x: list[float] = [i[0] for i in xy]
    #     y: list[float] = [i[1] for i in xy]
    #     if group is not None:
    #         x = [i * group for i in x]
    #     if norm is not False:
    #         if norm is True:
    #             norm = np.sum(y)
    #         y = [yi / norm for yi in y]
    #     plt.plot(x, y, *args, **kwargs)
    #     if show:
    #         plt.show()  # type: ignore

    # def plot_percent(
    #     self, *args: Any, group: Optional[float] = None, show: bool = True, norm: bool = False, **kwargs: Any
    # ) -> None:
    #     import matplotlib.pyplot as plt

    #     keys = sorted(self.keys())
    #     s = 0
    #     x: list[float] = []
    #     y: list[float] = []
    #     for k in keys:
    #         x.append(k)
    #         y.append(s)
    #         s += self[k]
    #         x.append(k)
    #         y.append(s)
    #     if norm is not False:
    #         if norm is True:
    #             norm = y[-1]  # type: ignore
    #         y = [yi / norm for yi in y]
    #     plt.plot(x, y, *args, **kwargs)
    #     if show:
    #         plt.show()  # type: ignore


# Just for fun, but could be usefull when defining thresholds !

# class _Smallest:
#     instance = None
#     def __new__(cls, *args, **kwargs):
#         if not isinstance(cls.instance, cls):
#             cls.instance = object.__new__(cls)
#         return cls.instance
#     def __lt__(self, x):
#         return True
#     def __le__(self, x):
#         return True
#     def __gt__(self, x):
#         return False
#     def __ge__(self, x):
#         return False
#     def __eq__(self, x):
#         return (x is self)
#     def __ne__(self, x):
#         return (x is not self)

# Smallest = _Smallest()
