import torch
from circulation_numerotation import classification_tct_ui, enums_tct_ui

from transformer.utils.config import get_config

cfg, paths = get_config()


def get_nbr_train_types() -> int:
    return len(enums_tct_ui.Type)


def _safe_parse_tct(tct: str) -> enums_tct_ui.TrainType:
    try:
        return classification_tct_ui.parse_tct(tct)
    except KeyError:
        return enums_tct_ui.TrainType(type=enums_tct_ui.Type.AUTRE)


def get_train_type_str(tct: str) -> str:
    return str(_safe_parse_tct(tct))


all_types = sorted(enums_tct_ui.Type, key=lambda x: str(x))
Id = torch.eye(len(all_types))
dic = {t: Id[i] for i, t in enumerate(all_types)}


def get_train_type_eigenvect(tct: str) -> torch.Tensor:
    key = _safe_parse_tct(tct).type
    return dic[key]  # type: ignore
