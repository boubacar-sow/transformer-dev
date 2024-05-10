import random
from typing import Callable

from transformer.inputs.data_structures import SillonChunk
from transformer.inputs.resources.train_type import get_train_type_str
from transformer.utils.config import get_config

cfg, _ = get_config()

# Abstract sillon filter


class SillonFilter:
    """
    Name must be unique for each separator !
    (do not define 'self.name' if you can't make it unique)
    """

    def __init__(self) -> None:
        self.name = "None"

    def __call__(self, sillon: SillonChunk) -> bool:
        """
        must return bool
        """
        raise RuntimeError("Can't use abstract SillonFilter class")


# Default sillon filter


class SillonFilterTrue(SillonFilter):
    def __init__(self) -> None:
        self.name = "True"

    def __call__(self, sillon: SillonChunk) -> bool:
        return True


# Logical operations on filters


class SillonFilterAnd(SillonFilter):
    def __init__(self, *filters: SillonFilter) -> None:
        if all(hasattr(filt, "name") for filt in filters):
            self.name = "__and__".join(filt.name for filt in filters)
        self.filters = filters

    def __call__(self, sillon: SillonChunk) -> bool:
        return all(filt(sillon) for filt in self.filters)


class SillonFilterOr(SillonFilter):
    def __init__(self, *filters: SillonFilter) -> None:
        if all(hasattr(filt, "name") for filt in filters):
            self.name = "__or__".join(filt.name for filt in filters)
        self.filters = filters

    def __call__(self, sillon: SillonChunk) -> bool:
        return any(filt(sillon) for filt in self.filters)


# Train type filters


class SillonFilterTrainType(SillonFilter):
    def __init__(self, typ: str) -> None:
        self.typ = typ
        self.name = "train_type_{}".format(typ)

    def __call__(self, sillon: SillonChunk) -> bool:
        return self.typ in get_train_type_str(sillon.train_num)


class SillonFilterNotTrainType(SillonFilter):
    def __init__(self, typ: str) -> None:
        self.typ = typ
        self.name = "not_train_type_{}".format(typ)

    def __call__(self, sillon: SillonChunk) -> bool:
        return self.typ not in get_train_type_str(sillon.train_num)


# class SillonFilterTransilienType(SillonFilter):
#     def __init__(self, typ: str) -> None:
#         self.typ = typ
#         self.name = "transilien_type_{}".format(typ)

#     def __call__(self, sillon: SillonChunk) -> bool:
#         return self.typ in get_ligne_transilien_str(sillon.train_num)


# class SillonFilterNotTransilienType(SillonFilter):
#     def __init__(self, typ: str) -> None:
#         self.typ = typ
#         self.name = "not transilien_type_{}".format(typ)

#     def __call__(self, sillon: SillonChunk) -> bool:
#         return self.typ not in get_ligne_transilien_str(sillon.train_num)


# class SillonFilterTrainTypeIncludeExclude:
#     """
#     exclude overwrites include
#     """
#     def __init__(self, include=None, exclude=None) -> None:
#         if include is None:
#             include = []
#         if exclude is None:
#             exclude = []
#         self.include = include
#         self.exclude = exclude
#     def __call__(self, sillon: SillonChunk) -> bool:
#         return (self.typ in get_train_type_str(sillon[0]['train_num']))


# Train num filters


class SillonFilterTrainNum(SillonFilter):
    def __init__(self, train_num: int | str) -> None:
        self.train_num = train_num
        self.name = "train_num_{}".format(train_num)

    def __call__(self, sillon: SillonChunk) -> bool:
        return sillon[0]["train_num"] == self.train_num  # type: ignore


class SillonFilterTrainNums(SillonFilter):
    def __init__(self, train_nums: list[int] | list[str]) -> None:
        self.train_nums = train_nums
        self.name = "train_nums_{}".format("_".join(sorted(train_nums)))  # type: ignore

    def __call__(self, sillon: SillonChunk) -> bool:
        return sillon.train_num in self.train_nums


class SillonFilterTrainNumFunc(SillonFilter):
    def __init__(self, train_num_func: Callable[[int | str], bool]) -> None:
        self.train_num_func = train_num_func

    def __call__(self, sillon: SillonChunk) -> bool:
        return self.train_num_func(sillon.train_num)


# Delay filters


class SillonFilterDelayMin(SillonFilter):
    def __init__(self, mi: float) -> None:
        self.mi = mi

    def __call__(self, sillon: SillonChunk) -> bool:
        return any(delay > self.mi for delay in sillon.prev_delays + sillon.foll_delays if delay is not None)


# Geographic filters


# def get_sillon_direction(sillon: SillonChunk, pr_info: PrInfo) -> str:
#     direction: list[str] = []
#     first_coords: tuple[float, float] = (0, 0)
#     last_coords: tuple[float, float] = (0, 0)
#     for obs in sillon:
#         coordinates = pr_info.pr_cich_to_info[obs.pr_cich].coordinates
#         if (obs.pr_cich in pr_info.pr_cich_to_info) and coordinates:
#             first_coords = coordinates
#             break
#     for obs in sillon[::-1]:
#         coordinates = pr_info.pr_cich_to_info[obs["pr_cich"]].coordinates
#         if (obs.pr_cich in pr_info.pr_cich_to_info) and coordinates:
#             last_coords = coordinates
#             break
#     if first_coords[1] < last_coords[1]:
#         direction.append("North")
#     elif first_coords[1] > last_coords[1]:
#         direction.append("South")
#     if first_coords[0] < last_coords[0]:
#         direction.append("East")
#     elif first_coords[0] > last_coords[0]:
#         direction.append("West")
#     return "-".join(direction)


# class SillonFilterDirection(SillonFilter):
#     def __init__(self, direction: str) -> None:
#         self.pr_info = PrInfo()
#         self.direction = direction

#     def __call__(self, sillon: SillonChunk) -> bool:
#         return self.direction in get_sillon_direction(sillon, self.pr_info)


# Prs filters


class SillonFilterContainsPrCich(SillonFilter):
    def __init__(self, pr_cich: str) -> None:
        self.pr_cich = pr_cich

    def __call__(self, sillon: SillonChunk) -> bool:
        return any(pr in self.pr_cich for pr in sillon.prev_prs + sillon.foll_prs if pr is not None)


# Other filters


class SillonFilterRandom(SillonFilter):
    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, sillon: SillonChunk) -> bool:
        return random.randint(0, self.n - 1) == 0


#################################################################################

# Misc

# filtering


def filter_sillons(sillons: list[SillonChunk], filters: list[SillonFilter]) -> list[SillonChunk]:
    return [s for s in sillons if all(filt(s) for filt in filters)]


SILLON_FILTERS: dict[str, SillonFilter] = {
    "all": SillonFilterTrue(),
}
