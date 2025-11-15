import numpy as np

from torrent_sim.config import BandwidthConfig
from torrent_sim.bandwidth import (
    BandwidthProfile,
    PeerKind,
    sample_bandwidth_profile,
    sample_many_bandwidth_profiles,
)
import logging
log = logging.getLogger(__name__)



def test_sample_bandwidth_profile_basic():
    cfg = BandwidthConfig()
    rng = np.random.default_rng(123)

    profile = sample_bandwidth_profile(cfg, rng)

    print(f"\nSampled bandwidth profile: {profile}")

    assert isinstance(profile, BandwidthProfile)
    assert profile.up_mbps > 0
    assert profile.down_mbps > 0
    assert profile.kind in (PeerKind.SYMMETRIC, PeerKind.ASYMMETRIC)
    # bps conversions should be consistent
    assert profile.up_bps == profile.up_mbps * 1e6
    assert profile.down_bps == profile.down_mbps * 1e6
