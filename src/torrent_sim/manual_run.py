from __future__ import annotations

from torrent_sim.config import default_config
from torrent_sim.core.engine import run_timestep_sim


def main() -> None:
    # Start from defaults
    cfg = default_config()

    # Make it a bit lighter/faster for quick tests
    cfg.time.max_time = 60000.0        # simulate 10 minutes
    cfg.time.dt = 1.0                # 1 second timestep
    cfg.arrival.max_peers = 30       # cap peers
    cfg.arrival.arrival_rate = 2.3   # ~1 peer every 10 seconds

    print("Running simulation...")
    result = run_timestep_sim(cfg)
    metrics = result.metrics

    # Basic summary
    print(f"\nSimulated time range: {metrics.times[0]:.1f}s -> {metrics.times[-1]:.1f}s")
    print(f"Total peers (including seed): {len(result.swarm.peers)}")

    # Final values
    final_completed = metrics.num_completed_peers[-1]
    final_avg_completion = metrics.avg_completion_fraction[-1]

    print(f"Completed peers at end: {final_completed}")
    print(f"Average completion fraction at end: {final_avg_completion:.3f}")

    # Show a few checkpoints
    print("\nSample checkpoints:")
    for t, n, c, avg in zip(
        metrics.times[::max(1, len(metrics.times)//5)],
        metrics.num_peers[::max(1, len(metrics.num_peers)//5)],
        metrics.num_completed_peers[::max(1, len(metrics.num_completed_peers)//5)],
        metrics.avg_completion_fraction[::max(1, len(metrics.avg_completion_fraction)//5)],
    ):
        print(
            f"t={t:6.1f}s | peers={n:3d} | completed={c:3d} | "
            f"avg_completion={avg:5.2f}"
        )


if __name__ == "__main__":
    main()
