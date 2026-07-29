# Tasks

| Task | Start date | End date | Status | Comment |
|------|------------|----------|--------|---------|
| 10: Results analysis + “next dataset” decision memo (SNLI vs MNLI vs keep synthetic) | 2026-06-15 | 2026-06-21 |  | Consolidate findings, failure modes, compute cost; propose next 4-week plan toward real NLI and/or freer generation |
| 09: Additional ablations: behavior mismatch + buffer policies | 2026-06-08 | 2026-06-14 |  | Actor temp/top-p mismatch; fresh/replay ratio; max-age cutoff; show which knobs dominate instability |
| 08: Main controlled sweeps (K_sync x U) across objectives | 2026-05-25 | 2026-06-07 |  | Run grid; produce first plots: accuracy vs time, stability markers, ESS/weight histograms; identify failure regimes |
| 07: Experiment runner + config for off-policiness knobs + logging | 2026-05-18 | 2026-05-24 |  | Sweep K_sync, replay intensity U, fresh-vs-replay mix, age cap; structured logs for plotting |
| 06: Async actor/learner with checkpoint sync + controllable staleness | 2026-05-04 | 2026-05-17 |  | Separate processes/threads; actor uses frozen snapshot updated every K_sync; log staleness Δ per sample |
| 05: Learner objectives for bandits: naive off-policy, IS-clipped, IS+KL | 2026-04-27 | 2026-05-03 |  | Implement loss variants; compute logp_target; track IS stats (ESS, clip fraction) and KL diagnostics |
| 04: Replay buffer + data collection loop (single-process first) | 2026-04-20 | 2026-04-26 |  | Store (x, completion, parsed_label, r, logp_behavior, behavior_policy_id, timestamps); deterministic replay sampling |
| 03: Baseline SFT/eval harness for SmolLM2-135M on synthetic NLI | 2026-04-13 | 2026-04-19 |  | Full finetune pipeline + heldout eval; confirms model/tokenizer/prompting; establishes “ceiling” and sanity metrics |
| 02: Synthetic NLI dataset (separate task): spec + generator + validation | 2026-03-30 | 2026-04-12 |  | Generate premise/hypothesis/label with controllable difficulty; build train/eval splits; check for leakage/artifacts |
| 01: Lock bandit-NLI spec (prompt, parsing, reward, logging schema) | 2026-03-23 | 2026-03-29 |  | Define label set, robust parser, reward, action=full completion, what to store in replay (logp_behavior etc.) |
