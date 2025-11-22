import numpy as np
import networkx as nx

from torrent_sim.config import Config
from torrent_sim.core.model import (
    SwarmState,
    PeerState,
    DownloadTask,
)
from torrent_sim.core.bandwidth import BandwidthProfile
from torrent_sim.core.topology import SEED_PEER_ID

def test_swarm_state_initialization():
    cfg = Config()
    G = nx.Graph()
    swarm = SwarmState(config=cfg, graph=G)

    # FileConfig: 1000 MB / 1 MB pieces = 1000 pieces
    assert swarm.num_pieces == cfg.file.num_pieces
    assert swarm.piece_size_bits == cfg.file.piece_size_mb * 1024 * 1024 * 8

    # Check RNG exists
    assert isinstance(swarm.rng, np.random.Generator)

def test_initialize_seed():
    cfg = Config()
    G = nx.Graph()
    G.add_node(SEED_PEER_ID)  # needed by topology
    swarm = SwarmState(config=cfg, graph=G)

    # Give seed arbitrary bandwidth
    seed_bw = BandwidthProfile(kind=None, up_mbps=10, down_mbps=10)

    swarm.initialize_seed(seed_bw)

    seed = swarm.get_peer(SEED_PEER_ID)

    assert seed.peer_id == SEED_PEER_ID
    assert seed.join_time == cfg.arrival.seed_join_time

    # Seed must start with ALL pieces
    assert seed.num_pieces_owned == swarm.num_pieces
    assert seed.active_downloads == []
    assert seed.completed_time is None

def test_peer_state_basic_methods():
    bw = BandwidthProfile(kind=None, up_mbps=5, down_mbps=20)
    peer = PeerState(peer_id=1, join_time=5.0, bandwidth=bw)

    # Initially no pieces
    assert peer.num_pieces_owned == 0
    assert peer.completion_fraction(num_pieces=100) == 0.0
    assert not peer.is_complete(100)

    # Add some pieces
    peer.owned_pieces.update({0, 1, 2})
    assert peer.num_pieces_owned == 3
    assert peer.completion_fraction(10) == 0.3
    assert not peer.is_complete(10)

    # Complete case
    peer.owned_pieces.update(range(10))
    assert peer.is_complete(10)

def test_download_task_structure():
    task = DownloadTask(from_peer_id=0, piece_index=5, remaining_bits=1024)

    assert task.from_peer_id == 0
    assert task.piece_index == 5
    assert task.remaining_bits == 1024

def test_swarm_neighbors():
    cfg = Config()
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)
    # 2 is isolated

    swarm = SwarmState(config=cfg, graph=G)

    # Add dummy peer states so neighbors() works (graph-only isn't enough)
    swarm.peers[0] = PeerState(peer_id=0, join_time=0, bandwidth=BandwidthProfile(kind=None, up_mbps=1, down_mbps=1))
    swarm.peers[1] = PeerState(peer_id=1, join_time=0, bandwidth=BandwidthProfile(kind=None, up_mbps=1, down_mbps=1))
    swarm.peers[2] = PeerState(peer_id=2, join_time=0, bandwidth=BandwidthProfile(kind=None, up_mbps=1, down_mbps=1))

    neighbors_0 = list(swarm.neighbors(0))
    neighbors_1 = list(swarm.neighbors(1))
    neighbors_2 = list(swarm.neighbors(2))

    assert neighbors_0 == [1]
    assert neighbors_1 == [0]
    assert neighbors_2 == []
