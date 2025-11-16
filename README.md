# **Torrent Swarm Simulation**

# **🏗️WORK IN PROGRESS**

This project develops a **Python-based simulation of BitTorrent-style swarm dynamics**, starting simple and progressively adding realism. The goal is to model how a file spreads through a P2P network under various bandwidth, topology, and arrival-rate conditions, and to visualize that diffusion over time.

### **Core Simulation Goals**

- Simulate **peer arrivals** using statistical distributions (e.g., Poisson).
- Assign **asymmetric or symmetric upload/download bandwidths** from configurable distributions.
- Model **network topology** using graph structures (e.g., random, preferential attachment, constrained multi-hop layouts). 
- Perform **piece-level transfers** using simplified rules at first:
    - No choking/unchoking initially.
    - No rarest-first initially.
    - Connections limited by `min(sender_share_bw, receiver_free_download_bw)`.
- Use a **fixed-timestep MVP engine**, then optionally upgrade to **SimPy** for event-driven accuracy.
- Add visualization: swarm progress over time, network snapshots, and completion distributions.

### **Advanced Features Planned (Post-MVP)**
- Choking/unchoking (BitTorrent tit-for-tat).
- Rarest-first piece selection.
- Bandwidth ramp-up, request overhead, and time-varying connection speeds.
- More realistic network models or optional integration with ns.py / Tribler simulators.
### **Project Environment**
- Python **3.13**
- Project managed with **uv**
- Standard structure: `src/`, `tests/`, `notebooks/`, clear configuration layer.