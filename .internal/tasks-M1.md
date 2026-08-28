## Milestone: M1 — Synthetic NLI dataset + RL harness

**T = 2026-06-01**

Infrastructure complete — validated synthetic NLI generator; replay-buffer RL harness with actor/learner pipeline, logging, and buffer operational. No RQs answered yet — representation milestone enabling the empirical work.

### Task table

| ID | Task name | Status | Duration (weeks) | Spec ref | Notes |
|----|-----------|--------|------------------|----------|-------|
| 01 | Generate synthetic NLI dataset (train/eval splits) | Todo | 1 | 02-synthetic-nli-data-spec.md | Generate premise/hypothesis/label with controllable difficulty; validate label correctness, hop/distractor balance, OOD splits |
| 02 | On-policy bandit RL baseline (REINFORCE) | Todo | 1 | | Simple policy gradient on synthetic NLI; confirm reward parsing, log-prob computation, gradient flow, and model learns to produce correct labels |
| 03 | Replay buffer + data collection loop (single-process first) | Todo | 1 | | Store (x, completion, parsed_label, r, logp_behavior, behavior_policy_id, timestamps); deterministic replay sampling |
| 04 | Learner objectives for bandits: naive off-policy, IS-clipped, IS+KL | Todo | 1 | | Implement loss variants; compute logp_target; track IS stats (ESS, clip fraction) and KL diagnostics |
| 05 | Async actor/learner with checkpoint sync + controllable staleness | Todo | 2 | | Separate processes/threads; actor uses frozen snapshot updated every K_sync; log staleness Δ per sample |

- **Status**: `Todo`, `Doing`, `Done`
- **Spec ref**: filename in `.internal/specs/`
- **Duration**: planned (for status `Todo` and `Doing`) or factual (for status `Done`) duration in weeks; can be 1, 2 or 3 (no more)

---

### Backlog tasks

| Task name | Notes |
|-----------|-------|
| Experiment runner + config for off-policiness knobs + logging | Sweep K_sync, replay intensity U, fresh-vs-replay mix, age cap; structured logs for plotting |
| Main controlled sweeps (K_sync x U) across objectives | Run grid; produce first plots: accuracy vs time, stability markers, ESS/weight histograms; identify failure regimes |
| Additional ablations: behavior mismatch + buffer policies | Actor temp/top-p mismatch; fresh/replay ratio; max-age cutoff; show which knobs dominate instability |
| Results analysis + "next dataset" decision memo | Consolidate findings, failure modes, compute cost; propose next 4-week plan toward real NLI and/or freer generation |
