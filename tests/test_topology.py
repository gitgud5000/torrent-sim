import numpy as np
import networkx as nx
import pytest

from torrent_sim.config import GraphConfig
from torrent_sim.core.topology import (
    SEED_PEER_ID,
    create_initial_graph,
    add_peer_with_topology,
)


def test_create_initial_graph_has_seed():
    G = create_initial_graph()
    assert SEED_PEER_ID in G.nodes
    assert G.number_of_nodes() == 1


@pytest.mark.parametrize("graph_type", ["barabasi", "erdos_renyi"])
def test_add_peer_increases_nodes_and_edges(graph_type):
    cfg = GraphConfig(graph_type=graph_type, avg_degree=4)
    rng = np.random.default_rng(123)

    G = create_initial_graph()
    for peer_id in range(1, 6):
        add_peer_with_topology(G, peer_id, cfg, rng)

    assert G.number_of_nodes() == 6
    # Should have some edges
    assert G.number_of_edges() > 0
