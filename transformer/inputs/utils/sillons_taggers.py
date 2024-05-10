"""
sillon_tagger is an object that is used in the loading of sillons and statistics computation.
It takes a sillon and returns a tag (tuple of strings), based e.g. on the train type or the train number.
These tags are used to group sillons that have the same characteristics.
"""

from transformer.inputs.data_structures import SillonChunk
from transformer.inputs.resources.train_type import get_train_type_str
from transformer.utils.config import get_config

cfg, _ = get_config()


# Abstract sillon tagger


class SillonTagger:
    """
    Name must be unique for each tagger !
    (do not define 'self.name' if you can't make it unique)
    """

    def __init__(self) -> None:
        self.name = "None"

    def __call__(self, sillon: SillonChunk) -> tuple[str, ...]:
        """
        must return str or hashable tuple
        """
        raise RuntimeError("Can't use abstract SillonTagger class")


# Default sillon tagger


class SillonTaggerAll(SillonTagger):
    def __init__(self) -> None:
        self.name = "all"

    def __call__(self, sillon: SillonChunk) -> tuple[str, ...]:
        return ()


# Logical operations on taggers


class SillonTaggerAnd(SillonTagger):
    def __init__(self, *taggers: SillonTagger):
        if all(hasattr(sep, "name") for sep in taggers):
            self.name = "__and__".join(sep.name for sep in taggers)
        self.taggers = taggers

    def __call__(self, sillon: SillonChunk) -> tuple[str, ...]:
        res: tuple[str, ...] = ()
        for tagger in self.taggers:
            tag = tagger(sillon)
            res += tag if isinstance(tag, tuple) else (tag,)
        return res


# Taggers


class SillonTaggerTrainNum(SillonTagger):
    def __init__(self) -> None:
        self.name = "train_num"

    def __call__(self, sillon: SillonChunk) -> tuple[str]:
        return (str(sillon.train_num),)


class SillonTaggerTrainType(SillonTagger):
    def __init__(self) -> None:
        self.is_initialized = False
        self.name = "train_type"

    def __call__(self, sillon: SillonChunk) -> tuple[str]:
        if not self.is_initialized:
            self.is_initialized = True
        return (get_train_type_str(sillon.train_num),)


# class SillonTaggerTransilienType(SillonTagger):
#     def __init__(self) -> None:
#         self.name = "transilien_type"

#     def __call__(self, sillon: SillonChunk) -> tuple[str]:
#         return (get_ligne_transilien_str(sillon.train_num),)


SILLON_TAGGERS: dict[str, SillonTagger] = {
    "all": SillonTaggerAll(),
    "train_type": SillonTaggerTrainType(),
}
