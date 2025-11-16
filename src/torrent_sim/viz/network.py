from __future__ import annotations
from typing import Optional


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from torrent_sim.engine import SimulationResult
from torrent_sim.model import SwarmState
from torrent_sim.topology import SEED_PEER_ID

def _as_swarm(obj) -> SwarmState:
    """
    Accept either a SimulationResult or a SwarmState and return the SwarmState.
    """
    if isinstance(obj, SimulationResult):
        return obj.swarm
    if isinstance(obj, SwarmState):
        return obj
    raise TypeError(f"Unsupported type for swarm: {type(obj)!r}")

def plot_swarm_snapshot(
    result_or_swarm,
    layout: str = "spring",
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Clean, balanced swarm snapshot:
    - smaller nodes
    - visible but subtle edges
    - neutral colormap
    - seed highlighted cleanly
    """
    swarm = _as_swarm(result_or_swarm)
    G: nx.Graph = swarm.graph

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.set_facecolor("#fafafa")

    # --- layout ---
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout in {"kamada_kawai", "kk"}:
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    num_pieces = swarm.num_pieces
    node_ids = list(G.nodes)

    completion = []
    sizes = []

    # --- build attributes ---
    for node_id in node_ids:
        peer = swarm.peers.get(node_id)
        if peer is None:
            frac = 0.0
            up_mbps = 1.0
        else:
            frac = peer.completion_fractions(num_pieces)
            up_mbps = peer.bandwidth.up_mbps

        completion.append(frac)

        # MUCH smaller size range now:
        # base 25, up to ~80 px depending on bw
        sizes.append(25.0 + 55.0 * np.log1p(up_mbps))

    # --- edges: visible, subtle ---
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        width=2,
        alpha=0.7,
        edge_color="#bbbbbb",
    )

    # --- nodes ---
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=sizes,
        node_color=completion,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        linewidths=0.6,
        edgecolors="#444444",
    )

    # --- seed highlight: minimal & clean ---
    if SEED_PEER_ID in G:
        idx = node_ids.index(SEED_PEER_ID)
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=[SEED_PEER_ID],
            node_size=sizes[idx],
            node_color=[completion[idx]],
            cmap="cividis",
            vmin=0.0, vmax=1.0,
            ax=ax,
            linewidths=1.3,
            edgecolors="#e53935",  # subtle red outline
        )

    # --- colorbar ---
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("completion fraction")

    ax.set_title(f"Swarm snapshot (t = {swarm.current_time:.1f}s)")
    ax.set_axis_off()

    if show and ax.figure:
        plt.show()

    return ax
