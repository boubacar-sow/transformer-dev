from collections import Counter, defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import pyproj

from transformer.utils.config import get_config
from transformer.utils.loaders import load_json, load_pickle, save_json, save_matplotlib_fig, save_pickle
from transformer.utils.logger import logger

# Load OmegaConf config
cfg, paths = get_config()


def build_pr_graph(make_figures: bool = False) -> None:  # noqa: C901
    """
    Given nodes and edges, build a graph of the Points Remarquables (but not only: the graph also
    contains non-pr nodes), and returns its largest connected component.

    Saves files containing:
        - a networkx.Graph object
        - a list of IDs Gaia who were found in two very distant nodes (=> error in the data)
    """

    liste_noeuds = load_json(paths.PRs_graph_nodes)
    liste_aretes = load_json(paths.PRs_graph_edges)

    geodesic = pyproj.Geod(ellps="WGS84")

    # +------------------- +
    # | Building the graph |
    # +------------------- +

    count_nodes_that_cannot_be_added = 0

    G = nx.Graph()
    for i in range(len(liste_noeuds)):
        if liste_noeuds[i]["geometry"] is not None:
            if "gaia_id" in liste_noeuds[i]["properties"]:
                gaia_id = liste_noeuds[i]["properties"]["gaia_id"]
            else:
                gaia_id = ""

            G.add_node(
                liste_noeuds[i]["properties"]["id"],
                geometry=liste_noeuds[i]["geometry"]["coordinates"][0],
                properties=liste_noeuds[i]["properties"],
                gaia_id=gaia_id,
            )
        else:
            count_nodes_that_cannot_be_added += 1

    count_edges_that_cannot_be_added = 0
    coordinates = nx.get_node_attributes(G, "geometry")
    for i in range(len(liste_aretes)):
        id_origine = liste_aretes[i]["properties"]["source"]
        id_destination = liste_aretes[i]["properties"]["target"]
        if G.has_node(id_origine) and G.has_node(id_destination) and "statuses" in liste_aretes[i]["properties"].keys():
            if all(
                status in ["Exploitée", "Transférée en voie de service"]
                for status in liste_aretes[i]["properties"]["statuses"].split(",")
            ):
                vol_oiseau = vol_oiseau = geodesic.line_length(
                    lons=[coordinates[id_origine][0], coordinates[id_destination][0]],
                    lats=[coordinates[id_origine][1], coordinates[id_destination][1]],
                )
                G.add_edge(
                    id_origine,
                    id_destination,
                    length_m=liste_aretes[i]["properties"]["length"],  # length in meters
                    vol_oiseau=vol_oiseau,
                )
        else:
            count_edges_that_cannot_be_added += 1
    count_alert = 0

    nodes_to_fuse = []
    nodes_per_gaia_id = defaultdict(list)
    for node in G.nodes(data=True):
        node_id, node_data = node
        nodes_per_gaia_id[node_data["gaia_id"]].append(node)
    for gaia_id, nodes_with_this_gaia_id in nodes_per_gaia_id.items():
        if gaia_id == "" or len(nodes_with_this_gaia_id) <= 1:
            continue
        for i in range(len(nodes_with_this_gaia_id)):
            for j in range(i + 1, len(nodes_with_this_gaia_id)):
                node_i = nodes_with_this_gaia_id[i]
                node_j = nodes_with_this_gaia_id[j]
                if node_i[0] != node_j[0] and node_i[1]["gaia_id"] == node_j[1]["gaia_id"]:
                    vol_oiseau = geodesic.line_length(
                        lons=[node_i[1]["geometry"][0], node_j[1]["geometry"][0]],
                        lats=[node_i[1]["geometry"][1], node_j[1]["geometry"][1]],
                    )
                if vol_oiseau < 10000:
                    nodes_to_fuse.append([node_i[0], node_j[0]])
                else:
                    count_alert += 1

    for i in range(len(nodes_to_fuse)):
        if G.has_node(nodes_to_fuse[i][0]) and G.has_node(nodes_to_fuse[i][1]):
            nx.contracted_nodes(G, nodes_to_fuse[i][0], nodes_to_fuse[i][1], self_loops=False, copy=False)

    # TODO fix later the issue of a few nodes having the same gaia_id
    counter = Counter(list(nx.get_node_attributes(G, "gaia_id").values()))
    logger.debug(f"PRs having same gaia_id: {counter.most_common()[:100]}")
    logger.debug(f"PR doublons : {[k for k in counter.keys() if counter[k] >= 2]}")
    PRs_doublons = [k for k in counter.keys() if counter[k] >= 2]

    # add weights on edges based on distance between nodes
    sigma = np.std(list(nx.get_edge_attributes(G, "vol_oiseau").values()))
    weights = {
        key: np.exp(-((value / (2 * sigma)) ** 2)) for key, value in nx.get_edge_attributes(G, "vol_oiseau").items()
    }
    nx.set_edge_attributes(G, weights, "weight")

    # remove self loops
    count_self_loop = nx.number_of_selfloops(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Stats on graph building, it does not sum up because some nodes in the original file
    # are actually the same node (identified by id_gaia)
    logger.debug(
        "".join(
            [
                f"Nodes: {G.number_of_nodes()}",
                f"| Original number of nodes: {len(liste_noeuds)}",
                f"| Nodes that cannot be added: {count_nodes_that_cannot_be_added}",
                f"| Pairs of nodes that were fused: {len(nodes_to_fuse)}",
                f"| Pairs of nodes sharing gaia_id but further away than 10 km: {count_alert}",
            ]
        )
    )
    logger.debug(
        "".join(
            [
                f"| Edges: {G.number_of_edges()}",
                f"| Original number of edges: {len(liste_aretes)}",
                f"| Edges that cannot be added: {count_edges_that_cannot_be_added}",
                f"| Self edges that were removed: {count_self_loop}",
            ]
        )
    )

    # Dealing with issues regarding connected components, we only keep the largest connected component
    logger.debug(
        "".join(
            [
                f"The graph is connected: {nx.is_connected(G)}",
                f", it has {nx.number_connected_components(G)}",
                f"components and {G.number_of_nodes()}",
                "nodes.",
            ]
        )
    )
    logger.debug(
        "".join(
            [
                "The components have respective number of nodes ",
                str([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]),
            ]
        )
    )
    largest_cc = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc).copy()
    logger.debug(
        "".join(
            [
                f"The new graph is connected: {nx.is_connected(G_connected)}",
                f"it has {nx.number_connected_components(G_connected)}",
                f"components and {G_connected.number_of_nodes()}",
                "nodes.",
            ]
        )
    )

    save_pickle(paths.PRs_graph_networkx, G_connected)
    save_json(paths.PRs_doublons, PRs_doublons)

    if make_figures:
        plot_graph(G, G_connected)


def plot_graph(G: nx.Graph, G_connected: nx.Graph) -> None:
    fig = plt.figure(dpi=300)
    colors = list(mcolors.TABLEAU_COLORS.values())
    i = 0
    for c in nx.connected_components(G):
        temp = G.subgraph(c).copy()
        suspiciousness = [
            0.5 if value > 100000 else 0.05 for value in nx.get_edge_attributes(temp, "vol_oiseau").values()
        ]
        count = sum([1 if value > 100000 else 0 for value in nx.get_edge_attributes(temp, "vol_oiseau").values()])
        if i == 0 and count > 0:
            logger.debug(f"There are {count} suspicious edges at least, they are too long.")
        if temp.number_of_nodes() == G_connected.number_of_nodes():
            node_size = 0.005
        else:
            node_size = 0.05
        nx.draw_networkx(
            temp,
            pos=nx.get_node_attributes(temp, "geometry"),
            with_labels=False,
            node_size=node_size,
            width=suspiciousness,
            node_color=colors[i],
            edge_color=colors[i],
        )
        i += 1
        i = i % len(colors)
        if i == 0:
            i = 1
    save_matplotlib_fig(paths.PRs_graph_figure_f.format("all_nodes"), fig)
    plt.close(fig)

    fig, _ = plt.subplots(dpi=300)
    colors = list(mcolors.TABLEAU_COLORS.values())
    i = 0
    for c in nx.connected_components(G):
        temp = G.subgraph(c).copy()
        suspiciousness = [
            0.5 if value > 100000 else 0.05 for value in nx.get_edge_attributes(temp, "vol_oiseau").values()
        ]
        count = sum([1 if value > 100000 else 0 for value in nx.get_edge_attributes(temp, "vol_oiseau").values()])
        if i == 0 and count > 0:
            logger.debug(f"There are {count} suspicious edges at least, they are too long.")
        if temp.number_of_nodes() == G_connected.number_of_nodes():
            node_size = 0.005
        else:
            node_size = 0.05
        if temp.number_of_nodes() > 1:
            nx.draw_networkx(
                temp,
                pos=nx.get_node_attributes(temp, "geometry"),
                with_labels=False,
                node_size=node_size,
                width=suspiciousness,
                node_color=colors[i],
                edge_color=colors[i],
            )

        i += 1
        i = i % len(colors)

    plt.axis("scaled")
    save_matplotlib_fig(paths.PRs_graph_figure_f.format("connected_components_larger_than_1"), fig)
    plt.close(fig)

    fig = plt.figure(dpi=300)
    nx.draw_networkx(
        G_connected,
        pos=nx.get_node_attributes(G_connected, "geometry"),
        with_labels=False,
        node_size=0.005,
        edge_color=colors[0],
        node_color=colors[0],
        width=0.05,
    )
    save_matplotlib_fig(paths.connected_PRs_graph_figure, fig)
    plt.close(fig)


def get_graph_pr_ids() -> list[str]:
    """Return the list of Gaia IDs of the PRs in the graph"""
    G = load_pickle(paths.PRs_graph_networkx)
    PR_IDs = np.array([v["gaia_id"] for v in nx.get_node_attributes(G, "properties").values()])
    return [pr_id for pr_id in list(PR_IDs) if pr_id != ""]
