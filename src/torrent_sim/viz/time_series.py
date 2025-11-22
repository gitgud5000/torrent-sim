from __future__ import annotations

import matplotlib.pyplot as plt

from torrent_sim.core.engine import SimulationResult, SimulationMetrics


def _as_metrics(obj) -> SimulationMetrics:
    """
    Accept either a `SimulationResult` or `SimulationMetrics` and return metrics.
    """
    if isinstance(obj, SimulationResult):
        return obj.metrics
    if isinstance(obj, SimulationMetrics):
        return obj
    raise TypeError(f"Unsupported type for metrics: {type(obj)!r}")


def plot_avg_completion(
    result_or_metrics,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot average completion fraction vs time.

    Parameters
    ----------
    result_or_metrics : SimulationResult | SimulationMetrics
        Either a full SimulationResult or just its metrics.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure+axes is created.
    show : bool, default True
        If True, call plt.show() at the end (ignored if ax is passed and you
        want to compose multiple plots).
    """
    m = _as_metrics(result_or_metrics)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(m.times, m.avg_completion_fraction, label="avg completion fraction")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("average completion (0–1)")
    ax.set_title("Average completion vs time")
    ax.grid(True)
    ax.legend()

    if show and ax.figure:
        plt.show()

    return ax

def plot_peer_counts(
    result_or_metrics,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot total number of peers and completed peers vs time.
    """
    m = _as_metrics(result_or_metrics)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(m.times, m.num_peers, label="total peers")
    ax.plot(m.times, m.num_completed_peers, label="completed peers")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("peers")
    ax.set_title("Peers over time")
    ax.grid(True)
    ax.legend()

    if show and ax.figure:
        plt.show()

    return ax

def plot_basic_overview(
    result_or_metrics,
    show: bool = True,
) -> None:
    """
    Make a 2-row figure:
      - top: avg completion vs time
      - bottom: total/completed peers vs time
    """
    m = _as_metrics(result_or_metrics)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Top: average completion
    ax1.plot(m.times, m.avg_completion_fraction, label="avg completion fraction")
    ax1.set_ylabel("avg completion (0–1)")
    ax1.set_title("Torrent swarm evolution")
    ax1.grid(True)
    ax1.legend()

    # Bottom: peer counts
    ax2.plot(m.times, m.num_peers, label="total peers")
    ax2.plot(m.times, m.num_completed_peers, label="completed peers")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("peers")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()

    if show:
        plt.show()
