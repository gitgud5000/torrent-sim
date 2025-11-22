from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from torrent_sim.config import Config
from torrent_sim.engine import (
    SimulationTrajectory,
    run_timestep_sim_with_frames,
)
from torrent_sim.viz.network import (
    plot_swarm_snapshot,  # adjust import to your module name
)


def animate_swarm(
    trajectory: SimulationTrajectory,
    layout: str = "spring",
    interval_ms: int = 200,
    show_topology: bool = False,
    save_path: str | None = None,
) -> FuncAnimation:
    """
    Create a matplotlib.animation.FuncAnimation for a swarm trajectory.

    Parameters
    ----------
    trajectory : SimulationTrajectory
        Output of run_timestep_sim_with_frames.
    layout : str
        Layout name passed to plot_swarm_snapshot.
    interval_ms : int
        Delay between frames in milliseconds.
    show_topology : bool
        Whether to draw faint grey topology edges under the flow arcs.
    save_path : str, optional
        If provided, save animation (e.g. 'swarm.mp4' or 'swarm.gif').

    Returns
    -------
    anim : FuncAnimation
    """
    frames = trajectory.frames
    if not frames:
        raise ValueError("No frames in trajectory; did you record snapshots?")

    fig, ax = plt.subplots(figsize=(10, 8))

    plot_swarm_snapshot(
        trajectory.frames[0].swarm,
        layout=layout,
        ax=ax,
        show=False,
        show_topology=show_topology,
        show_colorbar=True,
    )
    # We’ll reuse the same Axes and just redraw per frame for simplicity.
    # For typical swarm sizes this is fine.
    def _update(i: int):
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
        )
        # return artists for FuncAnimation; here we just return the Axes
        return ax,

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
