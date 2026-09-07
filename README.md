# Federated Fleet Learning for Embodied Agents under IoT Constraints

Federated learning across a simulated fleet of heterogeneous robots running a vision-language-action policy, with a systematic study of four realistic IoT stress axes — bandwidth caps, staleness, node dropout, and finite per-round energy — that the current literature identifies as open but has never plotted together at controlled scale.

> **Status:** Active research. Independent project, 2026.
> **Runs on:** MacBook Air (Apple Silicon), CPU-only inference; no NVIDIA GPU required.

---

## What this project builds

A simulated fleet of `N = 4 / 8 / 16` robots in MuJoCo + robosuite runs [SmolVLA](https://huggingface.co/lerobot/smolvla) — a 450M-parameter VLA policy that natively supports macOS — on manipulation tasks. Fleet updates aggregate via a **hand-written FedAvg** (no framework), and each experiment sweeps one IoT stressor at a time to isolate its cost.

The scientific contribution is the **payload-versus-accuracy tradeoff curves** these sweeps produce. The [July 2025 CRML survey](https://arxiv.org/abs/2507.17903) states these tradeoffs exist and are unstudied at controlled small scale; this project fills that gap and extends split-inference across the VLA action head (a scoped extension of [COHORT](https://arxiv.org/abs/2603.10436)'s perception splitting) as the workshop-paper deliverable.

---

## Method — four milestones

| # | Milestone | Duration | Deliverable |
|---|---|---|---|
| **M1** | Single-robot baseline | 3–4 weeks | Reproducible per-robot success rates on 3–5 manipulation tasks, seeded, in a public repo |
| **M2** | Heterogeneous fleet, centralised | 3–4 weeks | Centralised-pooled ceiling + characterisation of how non-IID-ness alone degrades it |
| **M3** | Federated layer under IoT constraints | 5–6 weeks | Payload-vs-accuracy curves under bandwidth caps / staleness / dropout / energy budgets |
| **M4** | Split-inference on the VLA action head | 4–6 weeks | Workshop-paper-sized result (CoRL / ICRA / IROS target) |

Instrumentation is standard from day one: wall-clock per step, memory, and — critically — parameter payload size (the currency in M3).

---

## Stack

| Component | Choice | Why on Mac |
|---|---|---|
| Physics + sim | MuJoCo 3.9 + robosuite 1.5.2 | Universal macOS binaries; official macOS support |
| Policy | SmolVLA (450M) | Only mainstream VLA that runs on CPU / Apple Silicon end-to-end |
| Robot framework | LeRobot (HuggingFace) | Native SmolVLA support; dataset tooling and training loops already written |
| Federated layer | Hand-rolled FedAvg | ~300 lines; instrumenting is worth more than a framework abstraction here |

**Not on the stack (with reasons):** PyBullet (last release Apr 2022, dead); Habitat-sim (Meta docs state "no longer actively maintained" beyond v0.3.4); Isaac Sim/Lab (NVIDIA-only, no ARM); π₀, GR00T-N1, OpenVLA (all CUDA-only or require 8× A100).

---

## Repository layout

```
federated-fleet-vla-iot/
├── src/
│   ├── fleet/          # MuJoCo + robosuite wrappers, non-IID task/morphology assignment
│   ├── policy/         # SmolVLA loader (LeRobot), action head
│   ├── federated/      # FedAvg from scratch, IoT stressors (bandwidth / stale / dropout / energy)
│   ├── evaluate/       # Centralised ceiling, metrics
│   └── utils/          # Seeding, logging
├── configs/            # fleet_4.yaml, fleet_8.yaml, fleet_16.yaml
├── scripts/            # run_baseline.py, run_fedavg.py, run_iot_stress.py
├── notebooks/          # Exploratory analysis
├── tests/              # Unit tests for FedAvg + fleet simulator
└── docs/               # method.md, evidence.md (literature anchors)
```

---

## Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Reproduce M1 single-robot baseline
python scripts/run_baseline.py --config configs/single_robot.yaml

# 3. Run centralised heterogeneous fleet (M2)
python scripts/run_baseline.py --config configs/fleet_8.yaml --centralised

# 4. Run FedAvg with one IoT stressor (M3)
python scripts/run_iot_stress.py --config configs/fleet_8.yaml --stressor bandwidth --cap-bytes 1e6
```

Every result in the paper reproduces with a single command from `scripts/`, fixed seeds, and versions pinned in `requirements.txt`.

---

## Papers this builds on

1. **CRML survey** — Wang, Li, Xie et al. (Jul 2025). *Federated Learning for Large-Scale Cloud Robotic Manipulation: Opportunities and Challenges.* [arXiv:2507.17903](https://arxiv.org/abs/2507.17903) — the problem statement.
2. **SmolVLA** — Shukor et al. (2026). *A vision-language-action model for affordable and efficient robotics.* [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) — the policy.
3. **COHORT** — Atreya et al. (Mar 2026). *Hybrid RL for Collaborative Large DNN Inference on Multi-Robot Systems Under Real-Time Constraints.* [arXiv:2603.10436](https://arxiv.org/abs/2603.10436) — the closest existing work; this project's M4 extends its split-inference idea from perception backbones to the VLA action head.
4. **CHORUS (Stanford)** — Doshi, Gao, Chen, Finn, Bohg (Jun 2026). *Decentralized Multi-Embodiment Collaboration with One VLA Policy.* [arXiv:2606.12352](https://arxiv.org/abs/2606.12352) — the natural comparison arm (does explicit federation beat implicit coordination?).
5. **Learning While Deploying** — Wang, Li, Xie et al. (May 2026). *Fleet-Scale Reinforcement Learning for Generalist Robot Policies.* [arXiv:2605.00416](https://arxiv.org/abs/2605.00416) — the industrial-scale analogue that this simulation-scale study parallels.

Full reference list in `docs/references.md`.

---

## Roadmap

30-day tactical plan in [`ROADMAP.md`](ROADMAP.md); milestone timeline above.

---

## Citation

```bibtex
@misc{mina2026fedvla,
  author       = {Ravindra Mina},
  title        = {Federated Fleet Learning for Embodied Agents under IoT Constraints},
  year         = {2026},
  howpublished = {\url{https://github.com/rvnmina/federated-fleet-vla-iot}},
  note         = {Work in progress}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

*Maintained by [Ravindra Mina](https://github.com/rvnmina) · M.Tech (Artificial Intelligence), IIT Kharagpur*
