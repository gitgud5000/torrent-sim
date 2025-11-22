from __future__ import annotations

import copy
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .arrival import (
    ArrivalConfig,
    TimeConfig,
    arrivals_up_to_time,
    generate_poisson_arrivals,
)
from .bandwidth import BandwidthProfile, sample_bandwidth_profile
from .config import Config
from .model import DownloadTask, PeerState, SwarmState
from .topology import SEED_PEER_ID, add_peer_with_topology, create_initial_graph


# Metrics & Results containers
@dataclass
class SimulationMetrics:
    """
    Time series metrics collected during the simulation.
    """

    times: list[float] = field(default_factory=list)
    num_peers: list[int] = field(default_factory=list)
    num_completed_peers: list[int] = field(default_factory=list)
    avg_completion_fraction: list[float] = field(default_factory=list)


@dataclass
class SimulationResult:
    """
    Final result of a simulation run.
    """

    config: Config
    swarm: SwarmState
    metrics: SimulationMetrics


@dataclass
class SimulationFrame:
    time: float
    swarm: SwarmState


@dataclass
class SimulationTrajectory:
    result: SimulationResult
    frames: list[SimulationFrame] = field(default_factory=list)


# Helper: spawn a new peer
# When a new peer arrives, we:
# sample its bandwidth profile,
# build its PeerState,
# add to SwarmState,
# wire into graph via topology rules.
def _spawn_peer(
    swarm: SwarmState,
    peer_id: int,
    join_time: float,
) -> None:
    """
    Create new PeerState, assign bandwidth, add to swarm and graph.
    """
    cfg = swarm.config
    rng = swarm.rng

    bw: BandwidthProfile = sample_bandwidth_profile(cfg.bandwidth, rng)

    peer = PeerState(
        peer_id=peer_id,
        join_time=join_time,
        bandwidth=bw,
    )
    swarm.add_peer(peer)

    # Topology:  add node + edges  to the underlying graph
    add_peer_with_topology(swarm.graph, peer_id, cfg.graph, rng)


# Helper: start new downloads (very simple policy for MVP)
# For each active peer:
#   if not complete, and has free slots:
#       find neighbors that:
#          exist as peers,
#          are active,
#          have at least one piece the peer lacks.
#       choose a random neighbor.
#       choose a random piece that neighbor has & peer lacks.
#       create DownloadTask with remaining_bits = piece_size_bits.


def _start_new_downloads(swarm: SwarmState) -> None:
    """
    For each active peer, start new downloads up to the configured
    max_concurrent_downloads if neighbors have pieces they lack.
    """

    rng = swarm.rng
    num_pieces = swarm.num_pieces
    piece_size_bits = swarm.piece_size_bits
    max_concurrent = swarm.config.bandwidth.max_concurrent_downloads

    for peer in swarm.active_peers():
        # Skip seed (it already has all piece; treat it as pure uploader)
        if peer.peer_id == SEED_PEER_ID:
            continue

        # Skip peers that already complete
        if peer.is_complete(num_pieces):
            continue

        # Count how many downloads are currently in-flight
        active_downloads = [t for t in peer.active_downloads if t.remaining_bits > 0]
        peer.active_downloads = active_downloads  # clean up completed downloads

        free_slots = max_concurrent - len(active_downloads)
        if free_slots <= 0:
            continue

        # Find neighbors who can supply pieces
        neighbor_ids = list(swarm.neighbors(peer.peer_id))
        rng.shuffle(neighbor_ids)  # randomize order

        # Compute set of pieces this peer already has or is already downloading
        downloading_pieces = {t.piece_index for t in peer.active_downloads}
        unavailable = peer.owned_pieces | downloading_pieces

        for neighbor_id in neighbor_ids:
            if free_slots <= 0:
                break

            if neighbor_id not in swarm.peers:
                continue  # neighbor not yet a peer

            neighbor = swarm.get_peer(neighbor_id)
            if neighbor.join_time > swarm.current_time:
                continue  # neighbor not yet active

            # Pieces neighbor hast that peer doesn't and isn't already downloading
            candidate_pieces = list(neighbor.owned_pieces - unavailable)
            if not candidate_pieces:
                continue  # no pieces to offer

            # Choose random piece to download (MVP:L random; later: rarest-first etc)
            piece_idx = rng.choice(candidate_pieces)

            # Create DownloadTask
            task = DownloadTask(
                from_peer_id=neighbor_id,
                piece_index=piece_idx,
                remaining_bits=piece_size_bits,
            )
            peer.active_downloads.append(task)

            # Mark this piece as "in progress" so we don't request it twice
            unavailable.add(piece_idx)
            free_slots -= 1


# Helper: update downloads for one timestep
# We need to:
# compute upload load per sender (how many tasks they’re serving),
# compute download load per receiver (how many tasks they’re receiving),
# for each task, compute:
# sender_share = sender.up_bps / max(1, num_uploads[sender])
# recv_share   = recv.down_bps / max(1, num_downloads[receiver])
# rate = min(sender_share, recv_share)
# transferred_bits = rate * dt
# remaining_bits -= transferred_bits
def _build_transfer_loads(swarm: SwarmState) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build dictionaries:
    - upload_counts[peer_id]: numbner of active uploads this peer is serving
    - download_counts[peer_id]: number of active downloads this peer is receiving
    """

    upload_counts: dict[int, int] = {}
    download_counts: dict[int, int] = {}

    for peer in swarm.active_peers():
        for task in peer.active_downloads:
            if task.remaining_bits <= 0:
                continue  # already completed

            # Sender side
            upload_counts[task.from_peer_id] = (
                upload_counts.get(task.from_peer_id, 0) + 1
            )

            # Receiver side
            download_counts[peer.peer_id] = download_counts.get(peer.peer_id, 0) + 1

    return upload_counts, download_counts


def _step_downloads(swarm: SwarmState, dt: float) -> None:
    """
    Advance all active downloads by one timestep of the length dt (seconds).
    """

    upload_counts, download_counts = _build_transfer_loads(swarm)

    for peer in swarm.active_peers():
        receiver = peer
        recv_bw = receiver.bandwidth

        for task in peer.active_downloads:
            if task.remaining_bits <= 0:
                continue  # already completed

            # Sender info
            if task.from_peer_id not in swarm.peers:
                # Sender peer not found (should not happen in practice)
                continue
            sender = swarm.get_peer(task.from_peer_id)
            send_bw = sender.bandwidth

            num_uploads = upload_counts.get(sender.peer_id, 1)
            num_downloads = download_counts.get(receiver.peer_id, 1)

            senbder_share = send_bw.up_bps / max(1, num_uploads)
            recv_share = recv_bw.down_bps / max(1, num_downloads)

            rate = min(senbder_share, recv_share)  # bits per second
            transferred_bits = rate * dt

            task.remaining_bits -= transferred_bits
            if task.remaining_bits < 0:
                task.remaining_bits = 0  # clamp


# 7️⃣ Helper: finalize completed pieces and completion times
# After updating remaining_bits, we need to:
#   For each peer, find tasks that reached 0,
#   Add those pieces to owned_pieces,
#   Optionally clear completed tasks,
#   Set completed_time if peer now owns all pieces.
def _finalize_completed_downloads(swarm: SwarmState) -> None:
    """
    For tasks that have finished (remaining_bits <= 0), grant the piece to the receiver peer
    and record completion time if they now have the full file.
    """

    num_pieces = swarm.num_pieces
    t_now = swarm.current_time

    for peer in swarm.active_peers():
        if peer.peer_id == SEED_PEER_ID:
            peer.completed_time = 0.0  # seed has full file at t=0
            continue  # seed already has all pieces

        new_pieces: list[int] = []
        remaining_tasks: list[DownloadTask] = []

        for task in peer.active_downloads:
            if task.remaining_bits <= 0:
                # Piece fully received
                if task.piece_index not in peer.owned_pieces:
                    peer.owned_pieces.add(task.piece_index)
            else:
                remaining_tasks.append(task)

        # Update owned pieces and active downloads
        if new_pieces:
            peer.owned_pieces.update(new_pieces)
        peer.active_downloads = remaining_tasks

        # set complted_time if this is the first time peer gets the full file
        if peer.completed_time is None and peer.is_complete(num_pieces):
            peer.completed_time = t_now


# 8️⃣ Helper: log metrics
def _log_metrics(swarm: SwarmState, metrics: SimulationMetrics) -> None:
    """
    Record aggregate metrics at the current time
    """

    t = swarm.current_time
    num_peers = swarm.num_pieces

    peers = list(swarm.peers.values())
    if not peers:
        return

    num_peers = len(peers)
    completed = sum(1 for p in peers if p.is_complete(swarm.num_pieces))
    avg_completion = (
        sum(p.completion_fraction(swarm.num_pieces) for p in peers) / num_peers
    )

    metrics.times.append(t)
    metrics.num_peers.append(num_peers)
    metrics.num_completed_peers.append(completed)
    metrics.avg_completion_fraction.append(avg_completion)


# 9️⃣ Main entrypoint: run_timestep_sim
def run_timestep_sim(config: Config) -> SimulationResult:
    """
    Run the 0->MVP timestep-based torrent simulation.

    - Uses Poisson peer arrivals.
    - Uses simple Barabási-Albert-like or Erdős-Rényi-like topology (per `GraphConfig`).
    - Uses fixed per-peer bandwidth from `BandwidthConfig`.
    - no choking, no rarest-first, no ramp-up (all off for MVP).
    """

    # 1) Initialization
    G: nx.Graph = create_initial_graph() # pylint: disable=invalid-name
    swarm = SwarmState(config=config, graph=G)
    rng = swarm.rng

    # Initialize seed with full file and its own bandwidth
    seed_bw = sample_bandwidth_profile(config.bandwidth, rng)
    swarm.initialize_seed(seed_bw)

    # Arrival schedule for all non-seed peers
    schedule = generate_poisson_arrivals(
        arrival_cfg=config.arrival, time_cfg=config.time, rng=rng
    )
    arrival_index = 0
    next_peer_id = SEED_PEER_ID + 1  # start after seed

    metrics = SimulationMetrics()
    t = 0.0
    dt = config.time.dt
    max_time = config.time.max_time

    next_log_time = 0.0
    log_interval = config.logging.log_interval

    # 2) Main simulation loop
    while t < max_time:
        swarm.current_time = t

        # 2.1) Activate new peers whose join_time <= t
        new_join_times, arrival_index = arrivals_up_to_time(schedule, t, arrival_index)
        for join_time in new_join_times:
            _spawn_peer(swarm, next_peer_id, join_time)
            next_peer_id += 1

        # 2.2) Start new downloads for peers with free slots
        _start_new_downloads(swarm)

        # 2.3) Step downloads forward by dt
        _step_downloads(swarm, dt)

        # 2.4) Finalize completed pieces and completion times
        _finalize_completed_downloads(swarm)

        # 2.5) Log metrics at intervals
        if t >= next_log_time:
            _log_metrics(swarm, metrics)
            next_log_time += log_interval

        # Advance time
        t += dt

    # Final log at end of simulation
    swarm.current_time = t
    _log_metrics(swarm, metrics)

    return SimulationResult(
        config=config,
        swarm=swarm,
        metrics=metrics,
    )


def run_timestep_sim_with_frames(
    config: Config,
    snapshot_interval: float = 5.0,
) -> SimulationTrajectory:
    """
    Run the timestep simulation and record `SwarmState` snapshots at regular intervals.
    NOTE: this uses deepcopy on SwarmState, which can be memory-intensive for large swarms.
    """

    # 1) Initialization (same as run_timestep_sim)
    G: nx.Graph = create_initial_graph()  # pylint: disable=invalid-name
    swarm = SwarmState(config=config, graph=G)
    rng = swarm.rng

    seed_bw = sample_bandwidth_profile(config.bandwidth, rng)
    swarm.initialize_seed(seed_bw)

    schedule = generate_poisson_arrivals(config.arrival, config.time, rng)
    arrival_index = 0
    next_peer_id = SEED_PEER_ID + 1

    metrics = SimulationMetrics()
    frames: list[SimulationFrame] = []

    t = 0.0
    dt = config.time.dt
    max_time = config.time.max_time

    next_log_time = 0.0
    log_interval = config.logging.log_interval

    next_snapshot_time = 0.0

    # 2) Main simulation loop
    while t < max_time:
        swarm.current_time = t

        # arrivals
        new_join_times, arrival_index = arrivals_up_to_time(schedule, t, arrival_index)
        for join_time in new_join_times:
            _spawn_peer(swarm, next_peer_id, join_time)
            next_peer_id += 1

        # downloads
        _start_new_downloads(swarm)
        _step_downloads(swarm, dt)
        _finalize_completed_downloads(swarm)

        # log metrics
        if t >= next_snapshot_time:
            _log_metrics(swarm, metrics)
            next_log_time += log_interval

        # snapshot for animation
        if t >= next_snapshot_time:
            # deepcopy swarm state
            frames.append(SimulationFrame(time=t, swarm=copy.deepcopy(swarm)))
            next_snapshot_time += snapshot_interval

        # Advance time
        t += dt

    swarm.current_time = t
    _log_metrics(swarm, metrics)

    result = SimulationResult(
        config=config,
        swarm=swarm,
        metrics=metrics,
    )
    return SimulationTrajectory(result=result, frames=frames)
