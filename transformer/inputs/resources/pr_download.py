"""
Download files from PDI services containing information on the PRs
"""

import json
import math
import os
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from transformer.inputs.resources.pr_infos import compute_pr_infos
from transformer.utils.config import get_config
from transformer.utils.file_system import FileSystem
from transformer.utils.loaders import load_json, save_csv, save_json, save_pickle
from transformer.utils.logger import logger
from transformer.utils.s3 import copy_folder_to_s3

cfg, paths = get_config()


token = os.getenv("PRS_DATA_TOKEN")


def get_data_from_api(url: str) -> list[Any]:
    objects = []
    header = {"Authorization": f"Bearer {token}"}
    with requests.Session() as session:
        session.headers.update(header)
        next_url = url
        while next_url is not None:
            resp = session.get(next_url)
            resp_json = resp.json()
            objects.extend(resp_json["features"])
            next_url = resp_json["next"]
    return objects


def get_pr_infos_from_cassini() -> pd.DataFrame:
    layer_slug = "cassini_v2_gaia_point_remarquable"
    view_slug = "full_rgi_line_geo_centroid"
    url = f"https://gateway.dev.dgexsol.fr/chartis/v2/layer/{layer_slug}/geojson/{view_slug}/"
    objects = get_data_from_api(url)
    for o in objects:
        if o["geometry"]["type"] == "Point":
            o["properties"]["centroid_geographic_coordinates"] = json.dumps(o["geometry"]["coordinates"])
        else:
            o["properties"]["centroid_geographic_coordinates"] = json.dumps([math.nan, math.nan])
    objects = [o["properties"] for o in objects]
    df = pd.read_json(json.dumps(objects), orient="records")
    return df


def get_pr_graph_from_cassini() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_slug = "rgi_line_geo"

    url = f"https://gateway.dev.dgexsol.fr/cassini-v2/graph/{layer_slug}/pr/nodes/"
    nodes = get_data_from_api(url)

    url = f"https://gateway.dev.dgexsol.fr/cassini-v2/graph/{layer_slug}/pr/edges/"
    edges = get_data_from_api(url)

    return nodes, edges


def download_and_process_pr_data(use_precomputed: bool = True) -> bool:
    """
    Download from chartis and cassini:
    - the list of PRs and infos about them
    - the list of nodes and edges on the PRs graph

    Only re-download if the data is older than cfg.data.use_cached_pr_data_for_K_days days.
    Returns True if a download has been performed.
    """

    # Check if the PR data is recent enough
    fs = FileSystem()
    if use_precomputed and fs.exists(paths.pr_metadata):
        pr_metadata = load_json(paths.pr_metadata)
        date = datetime.strptime(pr_metadata["date"], "%Y-%m-%d")
        date_diff = datetime.now() - date
        days_diff = date_diff.total_seconds() // (60 * 60 * 24)
        if days_diff < cfg.data.use_cached_pr_data_for_K_days:
            logger.info(f"Found PR data from less than {cfg.data.use_cached_pr_data_for_K_days} days.")
            logger.info(" => Skipping PR download and graph computation")
            return False

    logger.info("Downloading PR data")
    df = get_pr_infos_from_cassini()
    save_csv(paths.points_remarquables, df)

    logger.info("Downloading PRs graph data")
    nodes, edges = get_pr_graph_from_cassini()
    save_json(paths.PRs_graph_nodes, nodes, indent=4)
    save_json(paths.PRs_graph_edges, edges)

    # Save the date (used to check)
    save_json(paths.pr_metadata, {"date": datetime.now().strftime("%Y-%m-%d")})

    pr_infos = compute_pr_infos()
    save_pickle(paths.pr_infos_cache, pr_infos)

    if cfg.data.save_results_to_s3:
        copy_folder_to_s3(paths.pr_data)

    return True
