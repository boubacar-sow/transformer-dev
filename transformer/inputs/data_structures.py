from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy.typing
import torch


@dataclass
class SillonChunk:
    """
    A sillon chunk contains, for a given train, the n_prev previous scheduled times
    and delays and the n_foll upcoming times and delays.

    Snapshots are stored in pickle files as list[SillonChunk].

    Observation types: ["P", "T", "O", "A", "D"] => (Passage, Terminus, Origine, Arrivée, Départ)
    """

    # Attributes contained in snapshot files
    train_num: str
    circulation_id: str  # numéro de marche départ + jour de départ
    circulation_basic_id: str
    datetime: str  # in UTC timezone
    code_tct: str
    code_ui: str
    flags: list[str]
    prev_prs: list[str]
    foll_prs: list[str]
    prev_theo_times: list[float]
    foll_theo_times: list[float]
    prev_obs_times: list[float]
    foll_obs_times: list[float]
    prev_delays: list[float]
    foll_delays: list[float]
    prev_obs_types: list[Literal["P", "T", "O", "A", "D"]]
    foll_obs_types: list[Literal["P", "T", "O", "A", "D"]]
    prev_missing_theo_times: list[bool]
    foll_capa_ranks: list[int]

    # Attributes added when loading the snapshots
    prev_codes_ligne: list[str] = field(default_factory=list)  # used only for lignepk embedding
    prev_pks: list[float] = field(default_factory=list)  # used only for lignepk embedding
    foll_codes_ligne: list[str] = field(default_factory=list)  # used only for lignepk embedding
    foll_pks: list[float] = field(default_factory=list)  # used only for lignepk embedding


@dataclass
class Snapshot:
    """
    Processed snapshot ready to be aggregated into batches
    """

    x: torch.Tensor  # concatenated train vector informations
    y: torch.Tensor  # variable to predict
    embeddings_indices: dict[str, torch.Tensor]  # concatenated train embeddings indices
    mask: torch.Tensor  # mask[i,j] = True iff (train i arrive after PR j) AND (observation j is not missing)
    day: str  # e.g. "2023-01-24"
    time: int  # Snapshot time in seconds in the current day

    # optional attributes
    foll_theo_times: Optional[torch.Tensor] = None  # theoretical arrival times in the next PRs
    foll_delays: Optional[torch.Tensor] = None  # observed arrival times in the next PRs
    translation_delays: Optional[torch.Tensor] = None  # arrival times predicted by the translation
    train_type: Optional[torch.Tensor] = None  # type of each train
    train_num: Optional[list[str]] = None  # train numbers
    prev_prs_cich: Optional[list[list[str]]] = None  # CICH of the previous PRs for each sillon / PR
    foll_prs_cich: Optional[list[list[str]]] = None  # idem for following PRs


@dataclass
class SnapshotBatch:
    # neural network inputs/outputs
    x: torch.Tensor  # neural network input
    y: torch.Tensor  # neural network output
    key_mask: torch.Tensor  # mask[i,j] = does snapshot i contain less than j trains
    obs_mask: (
        torch.Tensor
    )  # mask[i,j,k] = True iff (train i in snapshot j arrives after PR k) AND (the observation is not missing)

    day: list[str]
    time: list[int]

    # optional attributes
    foll_theo_times: Optional[torch.Tensor] = None  # theoretical arrival times in the next PRs
    foll_delays: Optional[torch.Tensor] = None  # observed arrival times in the next PRs
    translation_delays: Optional[torch.Tensor] = None  # arrival times predicted by the translation
    train_type: Optional[torch.Tensor] = None  # type of each train
    train_num: Optional[list[list[str]]] = None  # train numbers
    prev_prs_cich: Optional[list[list[list[str]]]] = None  # CICH of the previous PRs for each sillon / PR
    foll_prs_cich: Optional[list[list[list[str]]]] = None  # idem for following PRs


@dataclass
class MeanStd:
    mean: float
    std: float


@dataclass
class NDMeanStd:  # n-dimensional MeanStd
    mean: numpy.typing.ArrayLike
    std: numpy.typing.ArrayLike
