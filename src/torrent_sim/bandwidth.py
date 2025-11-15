from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .config import BandwidthConfig


class PeerKind(Enum):
    SYMMETRIC = auto()
    ASYMMETRIC = auto()


# 4️⃣ Bandwidth profile dataclass
# This is what the generator returns for each new peer:
@dataclass
class BandwidthProfile:
    """
    Fixed per-peer bandwidth profile for the 0->MVP simulator.

    Units:
    -up_mbps / down_mbps: Megabits per second (Mbps)
    """

    kind: PeerKind
    up_mbps: float
    down_mbps: float

    @property
    def up_bps(self) -> float:
        """Upload capacity in bits per second."""
        return self.up_mbps * 1e6

    @property
    def down_bps(self) -> float:
        """Download capacity in bits per second."""
        return self.down_mbps * 1e6


# 5️⃣ Helpers to sample positive speeds
# We’ll start with a normal distribution truncated at >0. Simple and good enough for 0→MVP.
def _sample_positive_normal(
    rng: np.random.Generator,
    mean: float,
    std_factor: float,
    min_value: float = 0.1,
) -> float:
    """
    Sample a positive value from a normal distribution:
    N(mean, (mean * std_factor)^2), truncated at min_value.

    This is a simple approximation;  good enough for 0->MVP.
    """

    std = abs(mean * std_factor)
    if std == 0:
        return max(mean, min_value)

    value = rng.normal(loc=mean, scale=std)
    if value < min_value:
        value = min_value
    return value


def sample_peer_kind(
    cfg: BandwidthConfig,
    rng: np.random.Generator,
) -> PeerKind:
    """
    Decide whether a peer is symmetric or asymmetric based on `cfg.prob_symmetric`.
    """

    u = rng.random()
    if u < cfg.prob_symmetric:
        return PeerKind.SYMMETRIC
    return PeerKind.ASYMMETRIC


# 7️⃣ Main API: sample a bandwidth profile for a peer
# This is what you’ll call when a peer joins:
def sample_bandwidth_profile(
    cfg: BandwidthConfig,
    rng: np.random.Generator | None = None,
) -> BandwidthProfile:
    """
    Sample a `BandwidthProfile` for a new peer according to `BandwidthConfig`.

    For 0->MVP, this is a fix profile (no time variation).
    """

    if rng is None:
        rng = np.random.default_rng()

    kind = sample_peer_kind(cfg, rng)

    if kind == PeerKind.SYMMETRIC:
        up = _sample_positive_normal(
            rng,
            mean=cfg.sym_up_mean_mbps,
            std_factor=cfg.up_std_factor,
        )
        down = _sample_positive_normal(
            rng,
            mean=cfg.sym_down_mean_mbps,
            std_factor=cfg.down_std_factor,
        )
    else:  # ASYMMETRIC
        up = _sample_positive_normal(
            rng,
            mean=cfg.asym_up_mean_mbps,
            std_factor=cfg.up_std_factor,
        )
        down = _sample_positive_normal(
            rng,
            mean=cfg.asym_down_mean_mbps,
            std_factor=cfg.down_std_factor,
        )
    return BandwidthProfile(
        kind=kind,
        up_mbps=up,
        down_mbps=down,
    )

def sample_many_bandwidth_profiles(
    n: int,
    cfg: BandwidthConfig,
    rng: np.random.Generator | None = None,
) -> tuple[BandwidthProfile, ...]:
    """
    Sample n bandwidth profiles and return them as an immutable tuple.
    """
    if rng is None:
        rng = np.random.default_rng()

    return tuple(sample_bandwidth_profile(cfg, rng) for _ in range(n))
