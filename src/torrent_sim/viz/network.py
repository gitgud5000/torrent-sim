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
    show_base_edges: bool = False,
) -> plt.Axes:
    """
    Swarm snapshot with:
    - nodes colored by completion
    - node size ~ upload capacity
    - optional grey base edges weighted by total capacity
    - green arcs = u -> v upload capacity
    - red arcs   = v -> u upload capacity
    - arc width and color encode absolute capacity (clipped)
    """
    swarm = _as_swarm(result_or_swarm)
    G: nx.Graph = swarm.graph

    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 10))

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

    completion: list[float] = []
    sizes: list[float] = []

    # --- node attributes ---
    for node_id in node_ids:
        peer = swarm.peers.get(node_id)
        if peer is None:
            frac = 0.0
            up_mbps = 1.0
        else:
            frac = peer.completion_fraction(num_pieces)
            up_mbps = peer.bandwidth.up_mbps

        completion.append(frac)
        # small-ish nodes, scaled by upload bw
        sizes.append(25.0 + 55.0 * np.log1p(up_mbps))

    # --- edge capacities ---
    edges = list(G.edges())

    edge_total_caps: list[float] = []
    up_edges: list[tuple[int, int]] = []
    up_caps: list[float] = []
    down_edges: list[tuple[int, int]] = []
    down_caps: list[float] = []

    for u, v in edges:
        peer_u = swarm.peers.get(u)
        peer_v = swarm.peers.get(v)
        if peer_u is None or peer_v is None:
            edge_total_caps.append(0.0)
            continue

        # directional capacities in Mbps
        cap_u_to_v = min(peer_u.bandwidth.up_mbps, peer_v.bandwidth.down_mbps)
        cap_v_to_u = min(peer_v.bandwidth.up_mbps, peer_u.bandwidth.down_mbps)

        total_cap = cap_u_to_v + cap_v_to_u
        edge_total_caps.append(total_cap)

        if cap_u_to_v > 0:
            up_edges.append((u, v))
            up_caps.append(cap_u_to_v)
        if cap_v_to_u > 0:
            down_edges.append((v, u))  # reversed direction
            down_caps.append(cap_v_to_u)

    # clip to keep widths/colors sane if we ever have huge links
    cap_clip = 100.0  # Mbps

    # base grey widths (absolute-ish)
    widths = [0.4 + 0.04 * min(cap, cap_clip) for cap in edge_total_caps]

    def _arc_widths(
        caps: list[float],
        base: float = 0.3,
        scale: float = 0.02,
    ) -> list[float]:
        # narrower than base edges so arcs don't dominate
        return [base + scale * min(c, cap_clip) for c in caps]

    up_widths = _arc_widths(up_caps)
    down_widths = _arc_widths(down_caps)

    # map capacity -> color intensity in chosen colormap
    def _cap_colors(
        caps: list[float],
        cmap,
        cap_clip: float,
    ) -> list[tuple[float, float, float, float]]:
        colors: list[tuple[float, float, float, float]] = []
        for c in caps:
            norm = min(c, cap_clip) / cap_clip  # 0..1
            val = 0.3 + 0.7 * norm  # avoid super-pale colors
            colors.append(cmap(val))
        return colors

    greens = plt.cm.Greens
    reds = plt.cm.Reds

    up_colors = _cap_colors(up_caps, greens, cap_clip)
    down_colors = _cap_colors(down_caps, reds, cap_clip)

    # --- optional base edges: grey, capacity-aware on undirected G ---
    if show_base_edges:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=edges,
            width=widths,
            alpha=0.25,
            edge_color="#dddddd",
        )

    # --- directional arcs: draw on DiGraphs so rad works ---
    if up_edges:
        H_up = nx.DiGraph()
        H_up.add_nodes_from(G.nodes())
        H_up.add_edges_from(up_edges)

        nx.draw_networkx_edges(
            H_up,
            pos=pos,
            ax=ax,
            edgelist=up_edges,
            width=up_widths,
            edge_color=up_colors,  # green scale, u -> v
            alpha=0.9,
            connectionstyle="arc3,rad=0.20",
            arrowstyle="-",
            arrows=True,
        )

    if down_edges:
        H_down = nx.DiGraph()
        H_down.add_nodes_from(G.nodes())
        H_down.add_edges_from(down_edges)

        nx.draw_networkx_edges(
            H_down,
            pos=pos,
            ax=ax,
            edgelist=down_edges,
            width=down_widths,
            edge_color=down_colors,  # red scale, v -> u
            alpha=0.9,
            connectionstyle="arc3,rad=-0.20",
            arrowstyle="-",
            arrows=True,
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

    # --- seed highlight ---
    if SEED_PEER_ID in G:
        idx = node_ids.index(SEED_PEER_ID)
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=[SEED_PEER_ID],
            node_size=sizes[idx],
            node_color=[completion[idx]],
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=1.3,
            edgecolors="#e53935",
        )

    # --- colorbar for node completion ---
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("completion fraction")

    ax.set_title(f"Swarm snapshot (t = {swarm.current_time:.1f}s)")
    ax.set_axis_off()

    if show and ax.figure:
        plt.show()

    return ax
