from pathlib import Path

import networkx as nx  # type: ignore

from transformer.inputs.resources.pr_graph import build_pr_graph
from transformer.tests.fixtures.mock_files import mock_all_files
from transformer.utils.config import get_config
from transformer.utils.loaders import load_json, load_pickle

cfg, paths = get_config()


def test_graph_computation(tmp_path: Path) -> None:
    mock_all_files(tmp_path)

    build_pr_graph(make_figures=False)
    G = load_pickle(paths.PRs_graph_networkx)
    PRs_doublons = load_json(paths.PRs_doublons)
    connected_components = list(nx.connected_components(G))
    assert len(connected_components) == 1
    assert len(PRs_doublons) == 0
