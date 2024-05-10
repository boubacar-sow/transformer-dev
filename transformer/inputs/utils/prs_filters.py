"""
This code is currently unused ; it should be updated if we need a similar system.
Reason for deprecation: not clear if the code was meant to return
- a) either a list of bools
- b) or a filtered list of PRs.
It seems ot be a) for most functions, but it is b) for the only function that was
used in the code (PrFilterPrsWithSillonTagger).
"""

from transformer.inputs.data_structures import SillonChunk
from transformer.utils.config import get_config

cfg, _ = get_config()


class PrFilter:
    """
    Name must be unique for each separator !
    (do not define 'self.name' if you can't make it unique)
    """

    def __init__(self) -> None:
        self.name = "None"

    def __call__(self, obs_or_sillon: SillonChunk) -> bool | list[bool]:
        """
        must return bool
        """
        raise RuntimeError("Can't use abstract PrFilter class")


# Prs filters


# class PrFilterPrFunc(PrFilter):
#     def __init__(self, pr_func: Callable[[str], bool]):
#         self.pr_func = pr_func

#     def __call__(self, obs_or_sillon: Observation | list[Observation]) -> bool | list[bool]:
#         if isinstance(obs_or_sillon, Observation):
#             pr_id = obs_or_sillon.code_cich  # type: str
#             return self.pr_func(pr_id)
#         else:  # sillon
#             data = [self.pr_func(obs["code_cich"]) for obs in obs_or_sillon]
#             return data


# class PrFilterPrIdFunc(PrFilter):
#     def __init__(self, pr_id_func: Callable[[str], bool]):
#         self.pr_id_func = pr_id_func

#     def __call__(self, obs_or_sillon: Observation | list[Observation]) -> bool | list[bool]:
#         if isinstance(obs_or_sillon, Observation):
#             return self.pr_id_func(obs_or_sillon["pr_id"])
#         else:  # sillon
#             return [self.pr_id_func(obs["pr_id"]) for obs in obs_or_sillon]


# class PrFilterPrCichFunc(PrFilter):
#     def __init__(self, pr_cich_func: Callable[[str], bool]):
#         self.pr_cich_func = pr_cich_func

#     def __call__(self, obs_or_sillon: Observation | list[Observation]) -> bool | list[bool]:
#         if isinstance(obs_or_sillon, Observation):
#             return self.pr_cich_func(obs_or_sillon["pr_cich"])
#         else:  # sillon
#             return [self.pr_cich_func(obs["pr_cich"]) for obs in obs_or_sillon]


# class PrFilterPrsWithSillonTagger(PrFilter):
#     def __init__(self, prs: list[str], sillon_tagger: SillonTagger):
#         self.prs = prs
#         self.sillon_tagger = sillon_tagger

#     def __call__(self, sillon: Observation | list[Observation]) -> bool:
#         if isinstance(sillon, Observation):
#             raise ValueError(
#                 "PrFilterPrsWithSillonTagger is not implemented for observations yet, use sillons instead."
#             )
#         tag = self.sillon_tagger(sillon)
#         if isinstance(tag, str):
#             tag = (tag,)
#         return [obs for obs in sillon if (obs["code_cich"],) + tag in self.prs]  # type: ignore


#################################################################################

# Misc

# filtering


# def filter_prs_in_sillons(sillons: list[SillonChunk], filters: list[PrFilter]) -> list[SillonChunk]:
#     res = [[vi for vi in s if all(filt(vi) for filt in filters)] for s in sillons]
#     return {k: v for k, v in res.items() if len(v) > 0}
