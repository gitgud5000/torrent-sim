from torrent_sim.config import default_config
from torrent_sim.core.engine import run_timestep_sim


def test_run_timestep_sim_basic():
    cfg = default_config()
    cfg.time.max_time = 100.0
    cfg.arrival.max_peers = 20

    result = run_timestep_sim(cfg)

    assert result.metrics.times, "Metrics should not be empty"
    assert len(result.metrics.times) == len(result.metrics.avg_completion_fraction)
    assert len(result.swarm.peers) >= 1  # at least seed
