from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import ArrivalConfig, TimeConfig


@dataclass
class ArrivalSchedule:
    """
    Holds the join times for all non-seed peers.

    join_times[i] is the join time (in seconds) of peer (i + 1), assuming:
    - peer 0 is the seed (handled separately via seed_join_time)
    """

    join_times: list[float]

    def __len__(self) -> int:
        return len(self.join_times)

# 3️⃣ Arrival schedule generator (Poisson process)
def generate_poisson_arrivals(
    arrival_cfg: ArrivalConfig,
    time_cfg: TimeConfig,
    rng: np.random.Generator | None = None,
) -> ArrivalSchedule:
    """
    Generate a Poisson arrival schedule for non-seed peers.

    Returns and `ArrivalSchedule` where:
    - peer 0 is assumed to be the seed (using `arrival_cfg.seed_join_time`).
    - peers 1..N have join times in ascending order.
        An `ArrivalSchedule` with join times for all non-seed peers.
    """
    if rng is None:
        rng = np.random.default_rng()

    join_times: list[float] = []

    t = arrival_cfg.seed_join_time
    # we trear the seed as peer 0, so we start generation others after that
    num_peers = 1 # counting the seed already

    while t < time_cfg.max_time and num_peers < arrival_cfg.max_peers:
        # Exponential inter-arrival time with rate λ
        dt = rng.exponential(1.0 / arrival_cfg.arrival_rate)
        t += dt
        if t > time_cfg.max_time:
            break

        join_times.append(t)
        num_peers += 1

    return ArrivalSchedule(join_times=join_times)

#4️⃣ Helper for timestep loop
def arrivals_up_to_time(
        schedule: ArrivalSchedule,
        current_time: float,
        last_index: int,
) -> tuple[Sequence[int], int]:
    """
    Given an `ArrivalSchedule` and the current simulation time 

    Returns:
    - a sequence of join_times that are <= current_time and not yet consumed
    - the updated index into schedule.join_times

    `last_index` is how many arrivals have already been processed previously.
    """
    join_times = schedule.join_times
    n = len(join_times)

    idx = last_index

    while idx < n and join_times[idx] <= current_time:
        idx += 1

    # new arrivals are those between last_index and idx (exclusive of idx)
    new_arrivals =  join_times[last_index:idx]
    return new_arrivals, idx
