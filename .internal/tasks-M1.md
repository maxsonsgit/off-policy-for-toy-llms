## Milestone: M1 — Synthetic NLI dataset + RL harness

**T = 2026-09-01**

Infrastructure complete — validated synthetic NLI generator; replay-buffer RL harness with actor/learner pipeline, logging, and buffer operational. No RQs answered yet — representation milestone enabling the empirical work.

### Task table

| ID | Task name | Status | Duration (weeks) | Spec ref | Notes |
|----|-----------|--------|------------------|----------|-------|
| 01 | Generate synthetic NLI dataset (train/eval splits) | Doing | 2 | 02-synthetic-nli-data-spec.md | Generate premise/hypothesis/label with controllable difficulty; validate label correctness, hop/distractor balance, OOD splits |
| 02 | On-policy bandit RL baseline (REINFORCE) | Todo | 3 | | Simple policy gradient on synthetic NLI; confirm reward parsing, log-prob computation, gradient flow, and model learns to produce correct labels |
| 03 | Replay buffer + data collection loop (single-process first) | Todo | 2 | | Store (x, completion, parsed_label, r, logp_behavior, behavior_policy_id, timestamps); deterministic replay sampling |
| 04 | Learner objectives for bandits: naive off-policy, IS-clipped, IS+KL | Todo | 3 | | Implement loss variants; compute logp_target; track IS stats (ESS, clip fraction) and KL diagnostics |
| 05 | Staleness mechanism: frozen snapshots with controllable K_sync gap (synchronous) | Todo | 3 | | Collect batch with frozen θ_old, then run K_sync learner steps; staleness Δ known by construction; see Notes below |

- **Status**: `Todo`, `Doing`, `Done`
- **Spec ref**: filename in `.internal/specs/`
- **Duration**: planned (for status `Todo` and `Doing`) or factual (for status `Done`) duration in weeks; can be 1, 2 or 3 (no more)

---

### Notes

**Task 05 — staleness mechanism (synchronous, no asynchrony).**

Staleness does not require asynchrony — it requires a policy gap. The training loop alternates two phases:

1. **Collection phase.** Freeze the current policy as θ_old. Generate a batch of bandit interactions using θ_old as the behavior policy, logging `logp_behavior` and `behavior_policy_id` per sample.
2. **Learning phase.** Run exactly K_sync learner updates on a mixture of fresh and replayed samples.

Staleness Δ for every sample is then known by construction (≈ K_sync, with the actual per-sample Δ logged anyway). This replaces the originally planned "async actor/learner with checkpoint sync": a staleness knob that is *set* (K_sync) is deterministic, reproducible, and debuggable, as required by the validation philosophy (constitution §8), whereas asynchronously produced staleness is noisy and scheduling-dependent. Additionally, asynchrony classically exists to decouple collection from learning across hardware; under the compute constraint (§2.3, single consumer GPU) actor and learner would time-share one device — all of the concurrency cost, none of the throughput benefit. The other two off-policiness knobs (replay intensity, behavior-policy mismatch) never required asynchrony.

Deferred: true async actor/learner (separate processes, concurrent buffer access, checkpoint sync I/O) moves to the backlog, tentatively M2, where it doubles as a validation that async-produced staleness matches the synchronous Δ = K_sync approximation.

### Backlog tasks

| Task name | Notes |
|-----------|-------|
| Experiment runner + config for off-policiness knobs + logging | Sweep K_sync, replay intensity U, fresh-vs-replay mix, age cap; structured logs for plotting |
| Main controlled sweeps (K_sync x U) across objectives | Run grid; produce first plots: accuracy vs time, stability markers, ESS/weight histograms; identify failure regimes |
| Additional ablations: behavior mismatch + buffer policies | Actor temp/top-p mismatch; fresh/replay ratio; max-age cutoff; show which knobs dominate instability |
| Results analysis + "next dataset" decision memo | Consolidate findings, failure modes, compute cost; propose next 4-week plan toward real NLI and/or freer generation |
| True async actor/learner (separate processes) | Deferred from M1; revisit in M2 as validation that async staleness matches the synchronous Δ = K_sync approximation |
