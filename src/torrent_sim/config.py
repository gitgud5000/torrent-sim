from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FileConfig:
    file_size_mb: float = 1_000  # 1 GB
    piece_size_mb: float = 1 # 1 MB

    @property
    def num_pieces(self) -> int:
        return int(self.file_size_mb / self.piece_size_mb)


@dataclass
class TimeConfig:
    dt: float = 0.5  # timestep size in seconds
    max_time: float = 3_600  # total simulated time (1 hour default)


@dataclass
class ArrivalConfig:
    arrival_rate: float = 0.1  # peers per second (Poisson rate λ)
    max_peers: int = 500
    seed_join_time: float = 0.0  # seconds, usually 0


@dataclass
class BandwidthConfig:
    # Probability a new peer is symmetric (up ~ down)
    prob_symmetric: float = 0.3

    # Symmetric peer (e.g. fiber)
    sym_up_mean_mbps: float = 100.0
    sym_down_mean_mbps: float = 100.0

    # Asymmetric peer (e.g. ADSL)
    asym_up_mean_mbps: float = 10.0
    asym_down_mean_mbps: float = 50.0

    # Relative std dev for randomness (std = mean * factor)
    up_std_factor: float = 0.3
    down_std_factor: float = 0.3


@dataclass
class GraphConfig:
    graph_type: str = "barabasi"  # or "erdos_renyi", etc.
    avg_degree: int = 4  # target average degree / connections per node


@dataclass
class LoggingConfig:
    log_interval: float = 5.0       # seconds between metric samples
    store_peer_snapshots: bool = False
    snapshot_times: tuple[float, ...] = field(default_factory=tuple)
    # e.g., snapshot_times = (300.0, 600.0, 1200.0) for t=5m,10m,20m

# ========================================
@dataclass
class FeaturesConfig:
    # Keep all off for 0->MVP, but they’re here to flip later
    use_rarest_first: bool = False
    use_choking: bool = False
    use_bandwidth_ramp: bool = False

# ========================================

@dataclass
class Config:
    file: FileConfig = field(default_factory=FileConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


def default_config() -> Config:
    """
    Return a Config instance with all default values.
    Use this as the entrypoint for creating configs.
    """
    return Config()

__all__ = [
    "FileConfig",
    "TimeConfig",
    "ArrivalConfig",
    "BandwidthConfig",
    "GraphConfig",
    "LoggingConfig",
    "FeaturesConfig",
    "Config",
    "default_config",
]
