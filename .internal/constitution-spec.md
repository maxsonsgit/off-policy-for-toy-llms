## off-policy-for-toy-llms constitution spec

### 1. Executive summary

#### 1.1 Research topic

Off-policy reinforcement learning for toy LLMs

#### 1.2 Abstract

This project studies off-policy effects in asynchronous reinforcement learning for small decoder-only LLMs under tight compute constraints. We focus on a replay-buffer setup where fresh trajectories are continuously collected by an actor and appended to a buffer while a learner updates the policy from a mixture of fresh and replayed data. The initial task family is contextual bandits framed as unconstrained generation with parsing, starting from NLI label prediction (entailment / neutral / contradiction). We run controlled experiments that vary off-policiness (policy lag, replay intensity, behavior-policy mismatch) and compare objectives such as naive replay-based updates vs importance sampling (with clipping) and KL-regularized variants, measuring stability, final accuracy, and compute cost.

#### 1.3 Motivation

Replay-buffer and asynchronous data collection are common in practical RL systems, but off-policy effects are often confounded by long-horizon credit assignment and noisy rewards. Contextual bandits with deterministic rewards derived from parsing model outputs provide a clean, cheap testbed where off-policiness can be precisely controlled and measured, enabling clear conclusions about stability regimes, failure modes, and the cost/benefit of off-policy corrections for small LLM training.

### 2. Research object

#### 2.1 Object layer

Computational model

#### 2.2 Description

A small decoder-only LLM trained through an asynchronous replay-buffer reinforcement learning pipeline on contextual bandit tasks. The upper layer is an LLM itself. The lower level is a computer program executing the model.

#### 2.3 Assumptions / Restrictions

- Compute is tight — models fit and can be trained on a single consumer GPU
- Model family: small decoder-only transformer LLMs
- Task domain is restricted to contextual bandits (single-step decisions) — no long-horizon credit assignment
- Rewards are deterministic, derived from parsing the generated tokens against a fixed label set

### 3. Knowledge gap

Off-policy reinforcement learning with replay buffers is a standard technique in classical RL and deep RL for games, with well-studied corrections such as importance sampling (with clipping) and KL regularization. Existing RL work on LLMs predominantly uses on-policy methods (e.g., PPO) or focuses on long-horizon credit assignment (RLHF, tool-use agents), where off-policy effects are confounded by trajectory length, sparse rewards, and reward model noise. Specifically, it is unknown how controllable sources of off-policiness — policy staleness, replay intensity, behavior-policy mismatch etc. — individually and jointly affect training stability, sample efficiency, and final accuracy at toy scale for different hyperparameter values. A clean, single-step testbed where these factors can be isolated has not been systematically applied to small LLMs.

### 4. Research questions

**RQ1**: How do controllable sources of off-policiness — policy staleness, replay intensity, behavior-policy mismatch, etc. — individually and jointly affect training stability, sample efficiency, and final accuracy for small decoder-only LLMs in a contextual bandit setting?

**RQ2**: How does the hyperparameter landscape (learning rate, batch size, replay ratio) modulate sensitivity to off-policiness — which configurations are robust, and which collapse under moderate staleness?

**RQ3**: Which off-policy corrections — naive replay, importance sampling with clipping, and KL regularization — most reliably prevent degradation under high staleness, and what is the compute cost trade-off?

### 5. Research value

Answering these RQs requires building a controlled replay-buffer RL testbed for LLMs with deterministic rewards and isolatable off-policiness knobs — something that does not currently exist in the public literature. Most LLM+RL research operates in regimes where confounders (long-horizon credit assignment, noisy reward models, human feedback) prevent isolating the contribution of individual off-policiness factors.

The barrier is primarily methodological: designing a task that is genuinely LLM-like (autoregressive generation, text parsing) while being simple enough for systematic sweeps over off-policiness parameters. The compute cost of running such sweeps across hyperparameter configurations, even at toy scale, is also non-trivial. Results from this program would produce reusable infrastructure and diagnostic metrics for off-policy analysis that generalize beyond the initial bandit setting.

### 6. Study type

Empirical characterization — discovering the properties of small LLMs trained under off-policy RL through controlled observation sweeps over staleness, replay intensity, and behavior-policy mismatch, measuring their impact on stability, efficiency, and accuracy. A secondary component is representation — building the replay-buffer RL framework itself as a computational tool for producing these observations.

### 7. Resources & constraints

- **Compute**: single consumer GPU (e.g., RTX 3090/4090 class); no multi-GPU or cloud cluster access

### 8. Validation philosophy

Results are validated through controlled experimental sweeps where off-policiness knobs are systematically varied. A valid finding must demonstrate consistent patterns across multiple hyperparameter configurations — not a single lucky seed. Stability claims require showing absence of NaN/Inf events and bounded gradient norms across the sweep. Accuracy improvements must be reproducible across at least 3 random seeds with statistically significant separation (non-overlapping confidence intervals). Diagnostic metrics (ESS, importance weight variance, staleness) must be logged and reported as evidence connecting observed behavior to measured causes.

### 9. Roadmap

#### 9.1 Milestones

| ID | Name | Expected result | Time index | Strong scaling efficiency |
|-----------|-------|-----------------|------------|---------------------------|
| M1 | Synthetic NLI dataset + RL harness | Infrastructure complete — validated synthetic NLI generator; replay-buffer RL harness with actor/learner pipeline, logging, and buffer operational. No RQs answered yet — representation milestone enabling the empirical work. | T+3 | 0.7 |
| M2 | Initial off-policy characterization | RQ1 partially answered — demonstrate that controllable off-policiness sources produce measurable effects on training stability and accuracy; identify baseline instability regimes and which corrections (if any) prevent collapse. | T+5 | 0.3 |
| M3 | Hyperparameter sensitivity and correction comparison | RQ2 answered — map hyperparameter regions (learning rate, batch size, replay ratio) that are robust vs fragile; RQ3 answered — rank corrections by reliability under high staleness and quantify compute cost trade-offs. | T+9 | 0.3 |

#### 9.2 Long-term vision

Extend the off-policy diagnostic framework beyond single-step bandits to short- and medium-horizon decision tasks, building a reusable characterization toolkit applicable across model scales and task families. A longer-term goal is to produce practical guidance for practitioners choosing between on-policy and off-policy methods at different compute budgets, and to understand whether corrections that stabilize toy-scale models generalize to larger LLMs or break down as model capacity increases.
