from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colors as mcolors

from torrent_sim.engine import SimulationResult
from torrent_sim.model import SwarmState
from torrent_sim.topology import SEED_PEER_ID


class ArcSpec(NamedTuple):
    u: int
    v: int
    width: float
    color: tuple[float, float, float, float]
    rad: float


type Node = dict[int, tuple[float, float]]  # mapping node_id -> (x, y)


def _as_swarm(obj) -> SwarmState:
    """
    Accept either a SimulationResult or a SwarmState and return the SwarmState.
    """
    if isinstance(obj, SimulationResult):
        return obj.swarm
    if isinstance(obj, SwarmState):
        return obj
    raise TypeError(f"Unsupported type for swarm: {type(obj)!r}")


def _compute_layout(G: nx.Graph, layout: str) -> Node:
    """
    Compute a 2D layout for the swarm graph.

    Parameters
    ----------
    G : nx.Graph
        Swarm topology (undirected).
    layout : str
        One of: "spring", "kamada_kawai" / "kk", "circular", "random".

    Returns
    -------
    pos : dict[node, (x, y)]
        Mapping from node to 2D position.
    """
    layout = layout.lower()

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=0.15)  # k adjusts spacing
    elif layout in {"kamada_kawai", "kk"}:
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G, seed=42)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    return pos


def _compute_node_style(
    swarm: SwarmState, G: nx.Graph
) -> tuple[list[int], list[tuple], list[float]]:
    """
    Returns (node_ids, node_colors, node_sizes).

    - Node colors:
        * seed (OG) (SEED_PEER_ID): fixed blue fill.
        * Others: cividis_r(completion_fraction).
    - Node sizes:
        * Proportional to degree: size_base + size_scale * degree.
    """
    num_pieces = swarm.num_pieces
    node_ids: list[int] = list(G.nodes())

    cmap = plt.get_cmap("cividis_r")  # reversed: 0 -> yellow, 1 -> blue

    # visual parameters
    size_base = 100.0
    size_scale = 35.0

    seed_fill_rgba = mcolors.to_rgba("#1565c0")

    node_colors: list[tuple] = []
    node_sizes: list[float] = []

    for node_id in node_ids:
        peer = swarm.peers.get(node_id)
        if peer is None:
            frac = 0.0
        else:
            frac = peer.completion_fraction(num_pieces)

        if node_id == SEED_PEER_ID:
            color = seed_fill_rgba
        else:
            color = cmap(frac)

        # size based on degree
        degree = G.degree(node_id)
        size = size_base + size_scale * float(degree)

        node_colors.append(color)
        node_sizes.append(size)

    return node_ids, node_colors, node_sizes


def _draw_nodes(
    ax: plt.Axes,
    pos: dict[int, tuple[float, float]],
    node_ids: list[int],
    node_colors: list[tuple],
    node_sizes: list[float],
    seed_id: int,
) -> plt.Collection:
    """
    Draw all nodes with given colors and sizes, then overlay the seed
    with a purple outline.

    Returns the PathCollection for the main node set (for colorbar).
    """
    # draw all nodes in one pass
    nodes = nx.draw_networkx_nodes(
        G=None,  # we'll pass pos & nodelist explicitly below
        pos=pos,
        nodelist=node_ids,
        node_size=node_sizes,
        node_color=node_colors,
        ax=ax,
        linewidths=0.6,
        edgecolors="#444444",
    )

    # Labels with slight shadow effect for readability
    for node_id in node_ids:
        x, y = pos[node_id]
        ax.text(
            x,
            y,
            str(node_id),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Highlight seed (OG) with purple outline if present
    if seed_id in node_ids:
        idx = node_ids.index(seed_id)
        seed_size = node_sizes[idx]
        seed_color = node_colors[idx]  # fill color already set in _compute_node_style

        nx.draw_networkx_nodes(
            G=None,
            pos=pos,
            nodelist=[seed_id],
            node_size=[seed_size],
            node_color=[seed_color],
            ax=ax,
            linewidths=4,
            edgecolors="#b077d4",
        )
        ax.text(
            pos[seed_id][0],
            pos[seed_id][1],
            str(seed_id),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    return nodes


# ---- Links/Arcs computation ----
def _compute_link_rates(swarm: SwarmState) -> dict[tuple[int, int], float]:
    """
    Returns mapping (u, v) -> rate_bits_per_second at current snapshot,
    aggregated over active downloads.

    For each active DownloadTask on receiver peer R from sender S:
      - count uploads per sender and downloads per receiver,
      - compute per-task rate as min(sender_share, receiver_share),
      - accumulate into link_rates[(S, R)].
    """
    upload_counts: dict[int, int] = {}
    download_counts: dict[int, int] = {}

    # First pass: count how many uploads/downloads each peer is involved in
    for recv in swarm.peers.values():
        # Only consider peers that have joined
        if recv.join_time > swarm.current_time:
            continue
        for task in recv.active_downloads:
            # Skip completed tasks
            if task.remaining_bits <= 0:
                continue
            send_id = task.from_peer_id
            recv_id = recv.peer_id
            upload_counts[send_id] = upload_counts.get(send_id, 0) + 1
            download_counts[recv_id] = download_counts.get(recv_id, 0) + 1

    link_rates: dict[tuple[int, int], float] = {}

    # Second pass: compute per-task rates and aggregate
    for recv in swarm.peers.values():
        if recv.join_time > swarm.current_time:
            continue

        recv_id = recv.peer_id
        recv_bw = recv.bandwidth
        num_downloads = download_counts.get(recv_id, 0)

        for task in recv.active_downloads:
            if task.remaining_bits <= 0:
                continue
            send_id = task.from_peer_id
            if send_id not in swarm.peers:
                continue

            send = swarm.peers[send_id]
            send_bw = send.bandwidth
            num_uploads = upload_counts.get(send_id, 0)

            # per-direction bandwidth shares
            sender_share = send_bw.up_bps / max(1, num_uploads)
            recv_share = recv_bw.down_bps / max(1, num_downloads)

            rate = min(sender_share, recv_share)  # bits per second
            if rate <= 0:
                continue

            key = (send_id, recv_id)
            link_rates[key] = link_rates.get(key, 0.0) + rate

    return link_rates


def _compute_arc_specs(
    swarm: SwarmState,
    G: nx.Graph,
    link_rates: dict[tuple[int, int], float],
    rate_clip_mbps: float = 100.0,
    # Changed default rad_val to be slightly higher for better separation
    rad_val: float = 0.15,
) -> list[ArcSpec]:
    """
    For each directed flow (u, v) with rate > 0:
      - compute width from rate (bps -> Mbps, clipped),
      - choose color:
          * GREEN if u is more complete than v (u seeding v),
          * BLUE otherwise (flow from less->more complete),
      - assign rad so u->v and v->u are visually separated
        when both directions exist.
    """
    num_pieces = swarm.num_pieces

    # completion fractions per node
    completion_frac: dict[int, float] = {}
    for node_id in G.nodes():
        peer = swarm.peers.get(node_id)
        completion_frac[node_id] = peer.completion_fraction(num_pieces) if peer else 0.0

    # convert rates to Mbps and drop non-positive
    link_rates_mbps: dict[tuple[int, int], float] = {}
    for (u, v), rate_bps in link_rates.items():
        if rate_bps > 0:
            link_rates_mbps[(u, v)] = rate_bps / 1e6

    green_cmap = plt.get_cmap("Greens")
    blue_cmap = plt.get_cmap("Blues")

    width_min = 1.5  # Slightly thicker minimum for visibility
    width_scale = 20  # Adjusted scale

    arc_specs: list[ArcSpec] = []

    # We process every single flow.
    # Crucially, we do NOT check if the reverse flow exists.
    # We ALWAYS curve.
    for (u, v), r_mbps in link_rates_mbps.items():
        # Always apply positive curvature.
        # In Matplotlib, 'rad=0.15' curves the line to the RIGHT
        # relative to the direction u->v.
        # This means u->v curves one way, and v->u curves the OTHER way spatially,
        # creating a perfect separation (eye shape).
        rad = rad_val

        r_eff = min(r_mbps, rate_clip_mbps)
        norm = r_eff / rate_clip_mbps if rate_clip_mbps > 0 else 0.0
        width = width_min + width_scale * norm

        f_u = completion_frac.get(u, 0.0)
        f_v = completion_frac.get(v, 0.0)

        if f_u >= f_v:
            # Seeding flow (More complete -> Less complete)
            # Scale color from light green to dark green
            color_val = 0.4 + 0.6 * norm
            color = green_cmap(color_val)
        else:
            # Leeching/Backwards flow (Less complete -> More complete)
            color_val = 0.4 + 0.6 * norm
            color = blue_cmap(color_val)

        arc_specs.append(ArcSpec(u=u, v=v, width=width, color=color, rad=rad))

    return arc_specs


def _draw_arcs(
    ax: plt.Axes, pos: dict[int, tuple[float, float]], arc_specs: list[ArcSpec]
) -> None:
    """
    Draw directional arcs for flows using ArcSpec list.

    Each ArcSpec has:
      - u, v: endpoints
      - width: line width
      - color: RGBA tuple
      - rad: curvature (positive / negative / zero)
    """
    if not arc_specs:
        return

    # We'll draw each arc individually so we can vary rad per edge.
    for arc in arc_specs:
        u, v, width, color, rad = arc
        H = nx.DiGraph()
        H.add_edge(u, v)

        # connectionstyle "arc3,rad=..." creates the curve.
        # mutation_scale controls the size of the arrow head.
        nx.draw_networkx_edges(
            H,
            pos=pos,
            ax=ax,
            edgelist=[(u, v)],
            width=[width],
            edge_color=[color],
            alpha=0.85,  # Slight transparency
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",  # Sharp triangle arrow
            arrowsize=15,  # Explicit arrow size
            min_source_margin=15,  # Don't start line inside the node
            min_target_margin=15,  # Don't end line inside the node
        )


def plot_swarm_snapshot(
    result_or_swarm,
    layout: str = "spring",
    ax: plt.Axes | None = None,
    show: bool = True,
    show_topology: bool = False,
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot a swarm snapshot:

    - Nodes:
        * color = completion (cividis_r), seed forced to blue fill with purple outline
        * size   = base + scale * degree
    - Edges:
        * optional faint grey topology edges (if show_topology=True)
        * directional arcs:
            - (u -> v) drawn if there's active flow
            - width ∝ current rate (bps -> Mbps, clipped)
            - color:
                GREEN scale if u is more complete than v (seeding direction)
                BLUE scale otherwise (backwards flow)
            - rad chosen so u->v and v->u separate when both exist
    """
    swarm = _as_swarm(result_or_swarm)
    G: nx.Graph = swarm.graph

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)  # Adjusted fig size/dpi

    # Ensure higher DPI for crisper rendering even if an Axes was provided
    if ax.figure is not None:
        ax.figure.set_dpi(150)
    ax.set_facecolor("#fafafa")

    # 1) Layout
    pos = _compute_layout(G, layout)

    # # 2) Node style
    node_ids, node_colors, node_sizes = _compute_node_style(swarm, G)

    # 3) Link rates & arc specs
    link_rates = _compute_link_rates(swarm)
    arc_specs = _compute_arc_specs(swarm, G, link_rates)

    # Optional faint topology background (The potential connections)
    if show_topology and len(G.edges) > 0:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            width=0.8,
            alpha=0.40,  # Very faint
            edge_color="#999999",
            style="dashed",  # Dashed lines for topology
        )

    _draw_arcs(ax, pos, arc_specs)

    # 6) Nodes (with seed outlined purple)
    nodes = _draw_nodes(ax, pos, node_ids, node_colors, node_sizes, SEED_PEER_ID)

    # 7) Axes cosmetics
    ax.set_axis_off()
    ax.set_title(f"Swarm Snapshot (t = {swarm.current_time:.1f}s)", fontsize=14)

    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("cividis_r"), norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Completion Fraction", rotation=270, labelpad=15)

    if show and ax.figure:
        plt.show()

    return ax
