from __future__ import annotations

import functools
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from torrent_sim.core.engine import SimulationFrame, SimulationTrajectory
from torrent_sim.viz.network import (
    compute_layout,
    plot_swarm_snapshot,
)


def _render_frame_args(
    i: int,
    frame: SimulationFrame,
    *,
    total_frames: int,
    output_dir: pathlib.Path,
    layout: str,
    pos: dict,
    show_topology: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Render a single frame of the simulation to a PNG file."""
    fig, ax = plt.subplots(figsize=(10, 8))
    is_first_frame = i == 0
    is_first_frame = True

    plot_swarm_snapshot(
        frame.swarm,
        layout=layout,
        ax=ax,
        show=False,
        show_topology=show_topology,
        show_colorbar=is_first_frame,
        pos=pos,
    )

    # # Add a frame counter
    # ax.text(
    #     0.98,
    #     0.98,
    #     f"Frame: {i+1}/{total_frames}",
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    # )

    # Set fixed axis limits to prevent jitter
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Use tight_layout with padding instead of bbox_inches='tight'
    fig.tight_layout(pad=3.0)

    output_path = output_dir / f"frame_{i:06d}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _create_video_from_frames(
    frame_dir: pathlib.Path,
    save_path: str,
    fps: int,
    gpu_encoder: str | None = None,
) -> None:
    """Create a video from a directory of frames using ffmpeg."""
    glob_pattern = frame_dir / "frame_%06d.png"
    output_path = str(save_path)

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to save animations as videos."
        )

    video_codec = gpu_encoder if gpu_encoder else "libx264"

    # Base command
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(glob_pattern),
        "-c:v", video_codec,
        "-pix_fmt", "yuv420p",
    ]

    # Add encoder-specific options
    if video_codec == "libx264":
        command.extend(["-preset", "slow", "-crf", "22"])
    # Add other GPU encoder options here if needed
    # For example, for NVIDIA:
    if video_codec == "h264_nvenc":
        command.extend(["-preset", "p7", "-cq", "22"])

    # Add filters and final arguments
    command.extend([
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-vsync", "cfr",
        "-r", str(fps),
        "-y",
        output_path,
    ]
)

    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        print(f"Animation saved successfully to {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video with ffmpeg.")
        print("Command:", " ".join(command))
        print("ffmpeg stdout:\n", e.stdout)
        print("ffmpeg stderr:\n", e.stderr)
        if gpu_encoder:
            print(
                f"\nHint: The GPU encoder '{gpu_encoder}' might not be supported by your ffmpeg build."
            )
            print("Check available encoders with: `ffmpeg -encoders | grep 264`")
        raise


def animate_swarm(
    trajectory: SimulationTrajectory,
    layout: str = "spring",
    interval_ms: int = 200,
    show_topology: bool = False,
    save_path: str | None = None,
    frame_step: int = 1,
    gpu_encoder: str | None = None,
    **kwargs: Any,
) -> FuncAnimation | None:
    """
    Create an animation for a swarm trajectory.

    If `save_path` is provided, this function uses parallel processing to render
    frames and stitches them together with ffmpeg. This is significantly faster
    for saving long animations. In this mode, the function returns `None`.

    If `save_path` is not provided, it returns a `matplotlib.animation.FuncAnimation`
    object for interactive viewing (e.g., in a Jupyter notebook).

    Parameters
    ----------
    trajectory : SimulationTrajectory
        Output of run_timestep_sim_with_frames.
    layout : str
        Layout name for graph visualization (spring, kamada_kawai, circular, random).
    interval_ms : int
        Delay between frames in milliseconds.
    show_topology : bool
        Whether to draw faint grey topology edges.
    save_path : str, optional
        If provided, save animation (e.g., 'swarm.mp4'). GIFs are not supported
        in parallel mode.
    frame_step : int, optional
        Animate every Nth frame to speed up long simulations.
    gpu_encoder : str, optional
        Name of a GPU-accelerated ffmpeg encoder to use (e.g., 'h264_nvenc' for NVIDIA).
        If None, defaults to the CPU-based 'libx264'.
    **kwargs : dict, optional
        Additional keyword arguments, including `processes` for parallel execution.

    Returns
    -------
    anim : FuncAnimation or None
        Returns a FuncAnimation for interactive display or None if saving to file.
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

    # --- Parallel Rendering to File ---
    if save_path:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = pathlib.Path(temp_dir_str)
            num_frames = len(frames)
            fps = 1000 / interval_ms

            # FIX: Determine fixed axis limits to prevent jitter
            if pos:
                pos_arr = np.array(list(pos.values()))
                min_coords = pos_arr.min(axis=0)
                max_coords = pos_arr.max(axis=0)
                xlim = (min_coords[0] - 0.1, max_coords[0] + 0.1)
                ylim = (min_coords[1] - 0.1, max_coords[1] + 0.1)
            else:
                xlim = ylim = (-1.1, 1.1)  # Default if no nodes

            render_func = functools.partial(
                _render_frame_args,
                total_frames=num_frames,
                output_dir=temp_dir,
                layout=layout,
                pos=pos,
                show_topology=show_topology,
                xlim=xlim,
                ylim=ylim,
            )

            processes = kwargs.get("processes", os.cpu_count())
            print(
                f"Rendering {num_frames} frames in parallel using {processes} processes..."
            )

            with multiprocessing.Pool(processes=processes) as pool:
                # Use enumerate to pass the index `i` to the render function
                pool.starmap(render_func, enumerate(frames))

            print("All frames rendered. Now creating video...")
            _create_video_from_frames(
                temp_dir, save_path, fps=int(fps), gpu_encoder=gpu_encoder
            )
        return None

    # --- Interactive Animation ---
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

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=1000 / interval_ms)

    return anim
