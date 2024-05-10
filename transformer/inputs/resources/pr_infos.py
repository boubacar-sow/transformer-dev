import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyproj
from railway_utils.pks_and_anomalies import PyPk  # type: ignore[import-untyped]

from transformer.utils.config import get_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_csv, load_pickle
from transformer.utils.logger import logger

_, paths = get_config()
fs = FileSystem()

wgs84 = pyproj.CRS("EPSG:4326")
utm = pyproj.CRS("EPSG:3857")


@dataclass
class PR:
    """PR = Point Remarquable"""

    ID: str
    code_ci: int
    code_ch: str
    code_cich: str
    codes_ligne_pks: list[tuple[int, float]]
    coordinates: tuple[float, float]

    def get_single_code_ligne_pk(self, at_random: bool = False) -> tuple[int, float]:
        if at_random:
            idx = random.randint(0, len(self.codes_ligne_pks) - 1)
        else:
            idx = 0
        return self.codes_ligne_pks[idx]


def default_pr_factory() -> PR:
    return PR(
        ID="default",
        code_ci=0,
        code_ch="00",
        code_cich="000000",
        codes_ligne_pks=[(0, 0)],
        coordinates=(47.57679341553953, 3.6395354927589407),
    )


def pk_externe_to_pk_float(pk_str: str) -> float:
    """
    Input : pk of the form 3400567, ie int("34" + "00" + "567")
    Ouput : pk of the form 34.567
    """
    pk = PyPk(pk_str=pk_str)
    pk_float = pk.previous_rk
    sign = 1 if pk.sign == "+" else -1
    pk_float += sign * pk.dist_from_prev_rk / 1000
    return pk_float  # type: ignore[no-any-return]


def compute_pr_infos() -> dict[str, PR]:
    pr_infos = defaultdict(default_pr_factory)
    df = load_csv(paths.points_remarquables)
    data: list[dict[str, Any]] = df.to_dict("records")

    for row in data:
        lrs = json.loads(row["lrs"].replace("'", '"').replace("None", "null"))
        if "rgi_line_geo" in lrs:
            codes_ligne_pks = lrs["rgi_line_geo"]
        else:
            codes_ligne_pks_with_doublons = lrs["rgi_track_geo"]  # one PK for each track
            codes_ligne_pks = []
            lignes = []
            for d in codes_ligne_pks_with_doublons:
                if d["ligne"] not in lignes:
                    lignes.append(d["ligne"])
                    codes_ligne_pks.append(d)

        codes_ligne_pks = [
            (
                int(ligne_pk["ligne"]),
                pk_externe_to_pk_float(ligne_pk["pk"]),
            )
            for ligne_pk in codes_ligne_pks
        ]

        pr_centroid = tuple(json.loads(row["centroid_geographic_coordinates"]))

        if isinstance(row["code_ch"], float) and math.isnan(row["code_ch"]):
            row["code_ch"] = "00"

        pr = PR(
            ID=row["id"],
            code_ci=row["code_ci"],
            code_ch=row["code_ch"],
            code_cich=str(row["code_ci"]) + row["code_ch"],
            codes_ligne_pks=codes_ligne_pks,
            coordinates=pr_centroid,
        )

        pr_infos[pr.ID] = pr

    return pr_infos


@lru_cache
def get_pr_infos() -> dict[str, PR]:
    """
    Provide information on PRs given their ID Gaia
    """

    logger.info("Loading PR informations.")
    if fs.exists(paths.pr_infos_cache):
        return load_pickle(paths.pr_infos_cache)  # type: ignore[no-any-return]
    else:
        return compute_pr_infos()
