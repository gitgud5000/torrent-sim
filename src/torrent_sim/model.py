from __future__ import annotations

from dataclasses import dataclass, field

from collections.abc import Iterator
import networkx as nx
import numpy as np

from .bandwidth import BandwidthProfile
from .config import Config
from .topology import SEED_PEER_ID


# 4. Represent an in-progress download (per peer)
# When a peer is downloading pieces, it will have partially received bits;
# we’ll track that in a small dataclass.
@dataclass
class DownloadTask:
    """
    Represents an in-progress download of a single piece from one neighbor peer.

    For the 0->MVP:
    - One task corresponds to one piece from one source peer.
    - `remaining_bits` is how much data is left to transfer.
    """

    from_peer_id: int
    piece_index: int
    remaining_bits: int
    # Engine later will decrement remaining_bits each timestep.
    # When it reaches 0, the piece is completed.


# 5. Peer state
# Now the core per-peer representation.
@dataclass
class PeerState:
    """
    State of a single peer in the swarm.

    Notes:
    - `owned_pieces` tracks which pieces this peer has fully downloaded.
    - `active_downloads` tracks in-progress piece downloads.
    - `completed_time` is when the peer first got the full file (for metrics).
    """

    peer_id: int
    join_time: int
    bandwidth: BandwidthProfile

    # Set of fully owned piece indices (0..num_pieces-1)
    owned_pieces: set[int] = field(default_factory=set)

    # In-progress downloads (from neighbors)
    active_downloads: list[DownloadTask] = field(default_factory=list)

    # Time when this peer first completed the full file; None if not yet completed
    completed_time: int | None = None

    def has_piece(self, piece_index: int) -> bool:
        """Check if the peer owns the given piece."""
        return piece_index in self.owned_pieces

    @property
    def num_pieces_owned(self) -> int:
        """Number of pieces this peer currently owns."""
        return len(self.owned_pieces)

    def completion_fraction(self, num_pieces: int) -> float:
        """Fraction of pieces owned (0.0 to 1.0)."""
        if num_pieces == 0:
            return 0.0
        return len(self.owned_pieces) / num_pieces

    def is_complete(self, num_pieces: int) -> bool:
        """Check if the peer has completed downloading the full file."""
        return self.num_pieces_owned >= num_pieces

    # Later the engine will:
    # Add the seed with all pieces in owned_pieces.
    # For other peers, start with an empty set and fill it.


# 6. Swarm / simulation state
# The “world” object that holds everything the engine will operate on.
@dataclass
class SwarmState:
    """
    Global state of the torrent swarm simulation.

    Contains:
    - Graph topology (who can connect to whom).
    - Per-peer states.
    - Simulation time and config.
    - RNG for reproducibility.
    """

    config: Config
    graph: nx.Graph
    peers: dict[int, PeerState] = field(default_factory=dict)
    current_time: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    # Cached file-related quantities for convenience
    num_pieces: int = field(init=False)
    piece_size_bits: int = field(init=False)

    def __post_init__(self):
        file_cfg = self.config.file
        self.num_pieces = file_cfg.num_pieces
        # Convert MB to bits: MB -> bytes -> bits
        self.piece_size_bits = file_cfg.piece_size_mb * 1024 * 1024 * 8

    # ---- Peer manipulation helpers ----

    def add_peer(self, peer: PeerState) -> None:
        """
        Register a new peer in the swarm's peer dictionary.

        Note: the graph topology (node + edges) is managed elsewhere.
        (topology module). This only TRACKS the peer state.
        """
        if peer.peer_id in self.peers:
            raise ValueError(f"Peer {peer.peer_id} already exists in swarm")
        self.peers[peer.peer_id] = peer

    def get_peer(self, peer_id: int) -> PeerState:
        """Retrieve the PeerState for the given peer_id."""
        return self.peers[peer_id]

    def iter_peers(self) -> list[PeerState]:
        """Iterator over all PeerState objects in the swarm."""
        return self.peers.values()

    def active_peers(self) -> Iterator[PeerState]:
        """
        Peers that have joined ( join_time <= current_time ).
        """
        t = self.current_time
        return (p for p in self.peers.values() if p.join_time <= t)

    def neighbors(self, peer_id: int):
        """
        Iterate over neighbor peer ODs in the graph.
        """
        return self.graph.neighbors(peer_id)

    # ---- Seed helper ----

    def initialize_seed(self, seed_bandwidth: BandwidthProfile) -> None:
        """
        Create and register the seed peer ( peer_id=SEED_PEER_ID ) with all pieces.
        """
        seed_peer = PeerState(
            peer_id=SEED_PEER_ID,
            join_time=0.0,
            bandwidth=seed_bandwidth,
        )
        # Seed starts with the full file
        seed_peer.owned_pieces = set(range(self.num_pieces))
        self.add_peer(seed_peer)


__all__ = [
    "DownloadTask",
    "PeerState",
    "SwarmState",
]
