from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import GraphConfig

# 3.2. Initial graph with seed
# We need at least 1 node: the seed. Let’s standardize that peer id:
# Convention: peer_id == 0 is the OG seed.

SEED_PEER_ID = 0


def create_initial_graph() -> nx.Graph:
    """
    Create and initial swarm graph containing only the seed peer.

    - Node 0 is the seed peer.
    - No edges yet; those are added as new peers arrive.
    """

    G = nx.Graph()
    G.add_node(SEED_PEER_ID)
    return G


# 3.3. Helper: degree-weighted choice (for BA-like)
# When we do preferential attachment, we need to pick existing nodes with probability ∝ degree.
# Add this helper:


def _degree_weighted_choice(G: nx.Graph, k: int, rng: np.random.Generator) -> list[int]:
    """
    Sample k distinct nodes from G with probability proportional to degree(node).

    If all degrees are zero (e.g. initial state, only seed node) fall back to uniform sampling.
    """

    nodes = list(G.nodes)
    n = len(nodes)
    if n == 0:
        return []

    degrees = np.array([G.degree(node) for node in nodes], dtype=float)
    total_degree = degrees.sum()

    if total_degree <= 0:
        # all degrees zero, fall back to uniform sampling without replacement
        k = min(k, n)
        return list(rng.choice(nodes, size=k, replace=False))

    probs = degrees / total_degree
    k = min(k, n)
    chosen_indices = rng.choice(len(nodes), size=k, replace=False, p=probs)
    return [nodes[i] for i in chosen_indices]


# 3.4. BA-style attachment for new peer
# When a new peer arrives, we want to connect it to m existing peers:
# avg_degree is over all nodes. For an undirected graph, each edge contributes 2 to total degree.
# Rough heuristic: m ≈ max(1, avg_degree // 2).


def _add_peer_barabasi_like(
    G: nx.Graph, peer_id: int, cfg: GraphConfig, rng: np.random.Generator
) -> None:
    """
    Add a new peer using Barabási–Albert-style preferential attachment rule.

    - New peer is added as node `peer_id`.
    - It connects to m existing peers, where:
        m ≈ max(1, avg_degree // 2)
    - Existing nodes are chosen with probability ∝ degree(node).
    """
    if peer_id in G:
        raise ValueError(f"Peer {peer_id} already exists in graph")
    G.add_node(peer_id)

    # Determine how many edges to add for this new peer.
    m = max(1, cfg.avg_degree // 2)

    if len(G) == 1:
        # Only the new peer exists (shouldn't really happen if we always start from seed),
        # but guard angainst it anyway.
        return

    existing_nodes = [node for node in G.nodes if node != peer_id]
    # Make temporary graph without the new node for degree calculations.
    H = G.subgraph(existing_nodes)

    targets = _degree_weighted_choice(H, k=m, rng=rng)
    for target in targets:
        G.add_edge(peer_id, target)
    # Pedagogical note:
    # This is effectively an online version of the BA model:
    # each new node “likes” nodes that already have many links.
    # Over time, hubs emerge, and average distance between nodes tends to be small.


# ER-style attachment for new peer
# For ER-style, we connect to each existing node with probability p.
# We want expected degree near avg_degree. For a node joining into n_existing nodes, its expected degree is:
#   E[degree] = p * n_existing
# So choose p = avg_degree / n_existing, capped to 1.


def _add_peer_erdos_renyi_like(
    G: nx.Graph, peer_id: int, cfg: GraphConfig, rng: np.random.Generator
) -> None:
    """
    Add a new peer in an Erdős-Rényi-like fashion.

    - New peer is added as node `peer_id`.
    - For each existing node, we add an edge with probability p, where:
        p is chosen so that expected degree is ~avg_degree.
        p = min(1, avg_degree / n_existing)
    """
    if peer_id in G:
        raise ValueError(f"Peer {peer_id} already exists in graph")

    existing_nodes = list(G.nodes)

    G.add_node(peer_id)

    n_existing = len(existing_nodes)
    if n_existing == 0:
        # Only this node exists, nothing to connect to.
        return

    # Choose p so taht E[degree] ≈ avg_degree for this new node.
    p = cfg.avg_degree / n_existing
    p = float(np.clip(p, 0.0, 1.0))

    for node in existing_nodes:
        if rng.random() < p:
            G.add_edge(peer_id, node)


# Here, degree distribution is more “Poisson-ish.”
# No strong hubs, but still random enough and not fully connected.


def add_peer_with_topology(
    G: nx.Graph, peer_id: int, cfg: GraphConfig, rng: np.random.Generator
) -> None:
    """
    Add a new peer to the swarm according to the specified topology model.

    Dispatches based on `cfg.graph_type`:
    - "barabasi": Barabási-Albert-like preferential attachment.
    - "erdos_renyi": Erdős-Rényi-like random attachment.
    """
    graph_type = cfg.graph_type.lower()

    if graph_type == "barabasi":
        _add_peer_barabasi_like(G, peer_id, cfg, rng)
    elif graph_type == "erdos_renyi":
        _add_peer_erdos_renyi_like(G, peer_id, cfg, rng)
    else:
        raise ValueError(f"Unknown graph_type: {cfg.graph_type}")
