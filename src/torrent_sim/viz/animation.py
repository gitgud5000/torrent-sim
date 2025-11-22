from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from torrent_sim.core.engine import SimulationTrajectory
from torrent_sim.viz.network import (
    compute_layout,
    plot_swarm_snapshot,
)


def animate_swarm(
    trajectory: SimulationTrajectory,
    layout: str = "spring",
    interval_ms: int = 200,
    show_topology: bool = False,
    save_path: str | None = None,
    frame_step: int = 1,
) -> FuncAnimation:
    """
    Create a matplotlib.animation.FuncAnimation for a swarm trajectory.

    PERFORMANCE OPTIMIZATIONS:
    - Layout is computed once on the final topology and reused for all frames
    - This eliminates the most expensive operation (force-directed layout calculation)
    - For very long simulations, use frame_step > 1 to animate every Nth frame

    Parameters
    ----------
    trajectory : SimulationTrajectory
        Output of run_timestep_sim_with_frames.
    layout : str
        Layout name for graph visualization (spring, kamada_kawai, circular, random).
    interval_ms : int
        Delay between frames in milliseconds.
    show_topology : bool
        Whether to draw faint grey topology edges under the flow arcs.
    save_path : str, optional
        If provided, save animation (e.g. 'swarm.mp4' or 'swarm.gif').
    frame_step : int, optional
        Animate every Nth frame (default 1 = all frames). Use higher values
        (e.g., 5 or 10) to speed up very long simulations at the cost of temporal resolution.

    Returns
    -------
    anim : FuncAnimation
    """
    frames = trajectory.frames
    if not frames:
        raise ValueError("No frames in trajectory; did you record snapshots?")

    # Downsample frames if requested
    if frame_step > 1:
        frames = frames[::frame_step]

    # OPTIMIZATION: Pre-compute layout once on the final frame's topology
    # This is the single biggest performance win - layout calculation (especially spring)
    # is expensive and deterministic, so we compute it once and reuse for all frames.
    # For growing topologies, nodes are added incrementally and positioned within the
    # pre-computed layout space (filtered in plot_swarm_snapshot).
    final_graph = frames[-1].swarm.graph
    pos = compute_layout(final_graph, layout)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw initial frame with pre-computed positions
    plot_swarm_snapshot(
        frames[0].swarm,
        layout=layout,
        ax=ax,
        show=False,
        show_topology=show_topology,
        show_colorbar=True,
        pos=pos,
    )

    def _update(i: int):
        # Clear and redraw. While we could optimize further with artist updates,
        # the layout pre-computation already provides ~10-100x speedup for spring layout.
        # Further optimization would require significant complexity to track and update
        # individual artists (nodes, edges, labels) rather than full redraws.
        ax.clear()
        frame = frames[i]
        # frame.swarm is a SwarmState at that time
        plot_swarm_snapshot(
            frame.swarm,
            layout=layout,
            ax=ax,
            show=False,
            show_topology=show_topology,
            show_colorbar=False,
            pos=pos,  # Reuse pre-computed positions
        )
        # return artists for FuncAnimation; here we just return the Axes
        return (ax,)

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        # For mp4 you'll need ffmpeg installed; for gif, use Pillow writer
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path, writer="ffmpeg")

    return anim
