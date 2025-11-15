from torrent_sim.config import ArrivalConfig, TimeConfig
from torrent_sim.arrival import generate_poisson_arrivals

def test_generate_poisson_arrivals_basic():
    arrival_cfg = ArrivalConfig(arrival_rate=0.1, max_peers=10, seed_join_time=0.0)
    time_cfg = TimeConfig(dt=1.0, max_time=100.0)

    schedule = generate_poisson_arrivals(arrival_cfg, time_cfg)
    assert len(schedule.join_times) <= arrival_cfg.max_peers - 1
    assert all(t > arrival_cfg.seed_join_time for t in schedule.join_times)
    assert all(
        schedule.join_times[i] <= schedule.join_times[i + 1]
        for i in range(len(schedule.join_times) - 1)
    )
