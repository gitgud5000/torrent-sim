# **Torrent Swarm Simulation**
# **🏗️WORK IN PROGRESS**

This project builds a **Python-based simulation of BitTorrent-style swarm dynamics**, starting simple and gradually adding realism. The aim is to model how a file propagates through a P2P network under configurable bandwidth distributions and network topologies, and to visualize how it diffuses through the network over time.  

The simulator tracks piece-level transfers with concurrent download limits and bandwidth-constrained connections, collecting metrics on completion rates and swarm evolution. The broader goal is to build an intuitive (and quantitative) feel for how swarms behave, while also using the project as a way to explore discrete-event simulation techniques, network graph structures, and the mechanics that drive real P2P systems.

> **Note on authorship:**  
> This project is developed by me, with help from AI-assisted tools for brainstorming, code generation, and refinement. All architectural decisions, testing, and integration are still guided and validated manually.

### **Core Simulation Goals 0->1**
- Model **peer arrivals** as a simple Poisson process: sample exponential inter-arrival times to build a join schedule, with room to plug in richer arrival patterns later.
- Sample **peer bandwidth configurations** by starting with two simple connection types (symmetric vs. asymmetric) and drawing upload/download speeds from lightweight distributions—keeping it basic at first, with room to add more realistic bandwidth models later.
- Explore **network topology** by generating and comparing different graph structures (e.g., random, preferential attachment, simple multi-hop setups) to get a feel for how each shape influences swarm behavior.
- Perform **piece-level transfers** using simplified rules at first:
    - No choking/unchoking initially.
    - No rarest-first initially.
    - Static connections limited by `min(sender_share_bw, receiver_free_download_bw)`.
- Start with a **simple fixed-timestep engine**, with the plan to upgrade to **SimPy** or some other framework.
- Add visualization: swarm progress over time, network snapshots, and completion distributions.

### **Advanced Features Planned (Post-MVP)**
- [ ] Choking/unchoking (BitTorrent tit-for-tat), to explore how different reciprocity and collaboration strategies influence swarm throughput and overall system efficiency.
- [x] Record periodic snapshots of the swarm state during the simulation for offline visualization and later animation
- [x] Move from interactive `FuncAnimation` to offloading parallel frame rendering (multiprocessing + ffmpeg), connecting the simulation snapshots to a faster offline pipeline.
- [ ] Rarest-first piece selection, with room to experiment with alternative piece-request strategies—e.g., random-first, sequential-first, or balanced hybrid approaches—and compare how each one shapes swarm flow and completion dynamics.
- [ ] Add bandwidth variability, including ramp-up effects, protocol overhead, and time-varying connection speeds.
- [ ] Explore richer network models and peer‑discovery rules—different topology growth processes, attachment heuristics, and joining behaviors—to study how these choices shape swarm structure and performance.
