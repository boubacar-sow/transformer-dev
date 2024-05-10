from transformer.utils.config import get_config
from transformer.utils.loaders import load_json, load_pickle, save_json

cfg, paths = get_config()

if __name__ == "__main__":
    G = load_pickle(paths.PRs_graph_networkx)

    a = 0
    node_list = []
    for i in range(40):
        nei = G.neighbors(a)
        node_list.extend(list(nei))
        a = node_list[-1]
    node_set = set(node_list)

    nodes = load_json(paths.PRs_graph_nodes)
    edges = load_json(paths.PRs_graph_edges)

    selected_nodes = [nodes[i] for i in node_set]
    selected_edges = [
        e for e in edges if e["properties"]["target"] in node_set and e["properties"]["source"] in node_set
    ]

    save_json("transformer/tests/fixtures/data/PRs_graph_edges.json", selected_edges, indent=4)
    save_json("transformer/tests/fixtures/data/PRs_graph_nodes.json", selected_nodes, indent=4)
