import abc

from transformer.inputs.resources.train_type import get_train_type_str
from transformer.utils.config import get_config

cfg, _ = get_config()


# Abstract train_num filter


class TrainNumFilter(abc.ABC):
    """
    Name must be unique for each separator !
    (do not define 'self.name' if you can't make it unique)
    """

    name = "None"

    @abc.abstractmethod
    def __init__(self) -> None: ...

    @abc.abstractmethod
    def __call__(self, train_num: str) -> bool: ...


# Default train_num filter


class TrainNumFilterTrue(TrainNumFilter):
    def __init__(self) -> None:
        self.name = "True"

    def __call__(self, train_num: str) -> bool:
        return True


# Logical operations on filters


class TrainNumFilterAnd(TrainNumFilter):
    def __init__(self, *filters: TrainNumFilter):
        if all(hasattr(filt, "name") for filt in filters):
            self.name = "__and__".join(filt.name for filt in filters)
        self.filters = filters

    def __call__(self, train_num: str) -> bool:
        return all(filt(train_num) for filt in self.filters)


class TrainNumFilterOr(TrainNumFilter):
    def __init__(self, *filters: TrainNumFilter):
        if all(hasattr(filt, "name") for filt in filters):
            self.name = "__or__".join(filt.name for filt in filters)
        self.filters = filters

    def __call__(self, train_num: str) -> bool:
        return any(filt(train_num) for filt in self.filters)


# Train type filters


class TrainNumFilterTrainType(TrainNumFilter):
    def __init__(self, typ: str):
        self.typ = typ
        self.name = "train_type_{}".format(typ)

    def __call__(self, train_num: str) -> bool:
        return self.typ in get_train_type_str(train_num)


class TrainNumFilterNotTrainType(TrainNumFilter):
    def __init__(self, typ: str):
        self.typ = typ
        self.name = "not_train_type_{}".format(typ)

    def __call__(self, train_num: str) -> bool:
        return self.typ not in get_train_type_str(train_num)


# class TrainNumFilterTransilienType(TrainNumFilter):
#     def __init__(self, typ: str):
#         self.typ = typ
#         self.name = "transilien_type_{}".format(typ)

#     def __call__(self, train_num: str) -> bool:
#         return self.typ in get_ligne_transilien_str(train_num)


# class TrainNumFilterNotTransilienType(TrainNumFilter):
#     def __init__(self, typ: str):
#         self.typ = typ
#         self.name = "not_transilien_type_{}".format(typ)

#     def __call__(self, train_num: str) -> bool:
#         return self.typ not in get_ligne_transilien_str(train_num)


# Train num filters


class TrainNumFilterTrainNum(TrainNumFilter):
    def __init__(self, train_num: str):
        self.train_num = train_num
        self.name = "train_num_{}".format(train_num)

    def __call__(self, train_num: str) -> bool:
        return train_num == self.train_num


class TrainNumFilterTrainNums(TrainNumFilter):
    def __init__(self, train_nums: list[str]):
        self.train_nums = train_nums
        self.name = "train_nums_{}".format("_".join(sorted(train_nums)))

    def __call__(self, train_num: str) -> bool:
        return train_num in self.train_nums


TRAIN_NUM_FILTERS: dict[str, TrainNumFilter] = {
    "all": TrainNumFilterTrue(),
}
