# Roadmap — Federated Fleet VLA · 30-day daily plan

Every checkbox is one commit's worth of work. Push at the end of each day.

## Week 1 — Environment & baseline setup (M1 kickoff)

- [x] Day 01 — Repo bootstrap: LICENSE, .gitignore, requirements.txt, empty test hits
- [ ] Day 02 — Install MuJoCo 3.9 + robosuite 1.5.2 on Mac; verify with viewer smoke test
- [ ] Day 03 — Add `src/fleet/simulator.py` — wrap robosuite `SingleArmEnv` with seedable reset
- [x] Day 04 — Add `src/utils/seed.py` and `src/utils/run_logger.py` — fixed seeds; JSONL metrics (wall-clock, memory, payload bytes)
- [ ] Day 05 — Add `configs/single_robot.yaml` + `scripts/run_baseline.py` — CLI that loads one task
- [ ] Day 06 — Install LeRobot + SmolVLA; `src/policy/smolvla_wrapper.py` — load, forward, save
- [ ] Day 07 — First end-to-end rollout on one manipulation task, log success rate

## Week 2 — Heterogeneous fleet, centralised (M2)

- [ ] Day 08 — `src/fleet/heterogeneity.py` — non-IID object-class distribution per robot
- [ ] Day 09 — `configs/fleet_4.yaml`, `configs/fleet_8.yaml` — N-robot fleet configs
- [ ] Day 10 — `scripts/run_baseline.py --centralised` — pool all robots' data, train
- [ ] Day 11 — First 4-robot centralised run; log per-robot and pooled success rates
- [ ] Day 12 — Add embodiment variation: different gripper types in `heterogeneity.py`
- [ ] Day 13 — First 8-robot centralised run; compare against 4-robot ceiling
- [ ] Day 14 — Write `docs/method.md` M2 section; commit weekly summary

## Week 3 — Federated layer + first IoT stressor (M3 part 1)

- [ ] Day 15 — `src/federated/fedavg.py` — hand-rolled FedAvg, ~200 lines, tested
- [ ] Day 16 — `tests/test_fedavg.py` — parity test: FedAvg over 1 client = centralised
- [ ] Day 17 — `scripts/run_fedavg.py` — CLI wrapper, log per-round convergence
- [ ] Day 18 — First 4-robot FedAvg run; measure gap to centralised ceiling
- [ ] Day 19 — `src/federated/compress.py` — top-k gradient sparsification (bandwidth cap)
- [ ] Day 20 — `scripts/run_iot_stress.py --stressor bandwidth` — first bandwidth-cap sweep
- [ ] Day 21 — Plot bandwidth-vs-accuracy curve for N=4; commit to `benchmarks/`

## Week 4 — Remaining stressors & writeup

- [ ] Day 22 — `src/federated/stale.py` — async delayed updates with configurable delay
- [ ] Day 23 — Staleness sweep on N=4; add to `benchmarks/`
- [ ] Day 24 — `src/federated/dropout.py` — random node dropout mid-round
- [ ] Day 25 — Dropout sweep on N=4
- [ ] Day 26 — `src/federated/energy.py` — finite compute budget per round
- [ ] Day 27 — Energy sweep on N=4
- [ ] Day 28 — Combined 4-panel plot: all four stressors on one figure
- [ ] Day 29 — Write `docs/evidence.md` — cite every result and its data-generating command
- [ ] Day 30 — README overhaul with results section; tag v0.1.0

## Beyond Day 30 (M4 — the workshop paper)

- Split-inference across the VLA action head; scaled runs to N=16; comparison to CHORUS baseline; paper draft.

---

**Commit cadence:** aim for 1–3 commits/day. Small, focused commits are better than end-of-week bundles — the graph reads as sustained engineering, not a burst.

**Commit message convention:**
```
[area] short imperative — one line
```
e.g., `[federated] FedAvg parity test — trivial case matches centralised`
