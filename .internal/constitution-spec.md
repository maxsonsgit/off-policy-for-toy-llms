## off-policy-for-toy-llms constitution spec

### 1. Executive summary

#### 1.1 Research topic

Off-policy reinforcement learning for toy LLMs

#### 1.2 Abstract

This project studies off-policy effects in reinforcement learning for small decoder-only LLMs under tight compute constraints. We focus on a replay-buffer setup where trajectories are appended to a buffer while a learner updates the policy from a mixture of fresh and replayed data, with policy lag between data collection and learning controlled explicitly (e.g., via synchronous frozen-snapshot gaps, later true asynchronous collection). The initial task family is contextual bandits framed as unconstrained generation with parsing, starting from NLI label prediction (entailment / neutral / contradiction). We run controlled experiments that vary off-policiness (policy lag, replay intensity, behavior-policy mismatch, etc.) and compare objectives such as naive replay-based updates vs importance sampling (with clipping) and KL-regularized variants, measuring stability, final accuracy, and compute cost.

#### 1.3 Motivation

Replay-buffer and asynchronous data collection are common in practical RL systems, but off-policy effects are often confounded by long-horizon credit assignment and noisy rewards. Contextual bandits with deterministic rewards derived from parsing model outputs provide a clean, cheap testbed where off-policiness can be precisely controlled and measured, enabling clear conclusions about stability regimes, failure modes, and the cost/benefit of off-policy corrections for small LLM training.

### 2. Research object

#### 2.1 Object layer

Computational model

#### 2.2 Description

A small decoder-only LLM trained through a replay-buffer reinforcement learning pipeline with controllable policy lag between data collection and learning, on contextual bandit tasks. The upper layer is an LLM itself. The lower level is a computer program executing the model.

#### 2.3 Assumptions / Restrictions

- Compute is tight — models fit and can be trained on a single consumer GPU
- Model family: small decoder-only transformer LLMs
- Task domain is restricted to contextual bandits (single-step decisions) — no long-horizon credit assignment
- Rewards are deterministic and programmatically computable from the generated tokens (initially via parsing against a label set)
- Task difficulty is calibrated such that the on-policy baseline reaches a non-saturating accuracy band, leaving measurable headroom for degradation effects
- Not part of the research object: the pretrained base checkpoint, tokenizer, and inference stack are fixed infrastructure

### 3. Knowledge gap

Off-policy reinforcement learning with replay buffers is a standard technique in classical RL and deep RL for games, with well-studied corrections such as importance sampling (with clipping) and KL regularization. Existing RL work on LLMs predominantly uses on-policy methods (e.g., PPO) or focuses on long-horizon credit assignment (RLHF, tool-use agents), where off-policy effects are confounded by trajectory length, sparse rewards, and reward model noise. The specific gaps this program aims to fill:

- **Gap 1**: It is unknown how controllable sources of off-policiness — policy staleness, replay intensity, behavior-policy mismatch, etc. — individually and jointly affect training stability, sample efficiency, and final accuracy at toy scale for small LLMs. A clean, single-step testbed where these factors can be isolated has not been systematically applied to small LLMs.
- **Gap 2**: It is unknown whether hyperparameter recipes tuned for on-policy LLM training (learning rate, batch size, etc.) remain valid when training becomes off-policy — i.e., where the robust and fragile regions of the hyperparameter landscape lie under off-policiness.
- **Gap 3**: Off-policy corrections proven in classical and deep RL have unverified transfer to autoregressive generation, where the action is a full token sequence and sequence-level importance ratios are products over tokens, with qualitatively different variance behavior than per-action ratios. Whether and why these corrections work in this regime is unexplored.

### 4. Research questions

**RQ1**: How do controllable sources of off-policiness — policy staleness, replay intensity, behavior-policy mismatch, etc. — individually and jointly affect training stability, sample efficiency, and final accuracy for small decoder-only LLMs in a contextual bandit setting?

**RQ2**: Do hyperparameter recipes tuned for on-policy LLM training — learning rate, batch size, replay ratio, etc. — remain valid under off-policiness? Which configurations are robust, and which collapse under moderate staleness?

**RQ3**: Under what conditions does each off-policy correction — naive replay, importance sampling with clipping, KL regularization, etc. — succeed or fail under high staleness, and what mechanism (e.g., the bias–variance trade-off in sequence-level importance ratios) explains the boundary? What is the compute cost trade-off of each correction?

### 5. Research value

Answering these RQs requires building a controlled replay-buffer RL testbed for LLMs with deterministic rewards and isolatable off-policiness knobs — something that does not currently exist in the public literature. Most LLM+RL research operates in regimes where confounders (long-horizon credit assignment, noisy reward models, human feedback) prevent isolating the contribution of individual off-policiness factors. The core methodological barrier is that the LLM regime differs from classical RL in a specific, testable way: importance ratios are products over tokens of an autoregressive sequence, so variance grows with sequence length and the classical correction taxonomy may not transfer unchanged.

The barrier is primarily methodological: designing a task that is genuinely LLM-like (autoregressive generation, text parsing) while being simple enough for systematic sweeps over off-policiness parameters. The compute cost of running such sweeps across hyperparameter configurations, even at toy scale, is also non-trivial. Results from this program would produce reusable infrastructure and diagnostic metrics for off-policy analysis; whether these findings generalize beyond the initial bandit setting and model scale is outside the scope of the initial roadmap (see §9.2).

### 6. Study type

Empirical characterization — discovering the properties of small LLMs trained under off-policy RL through controlled observation sweeps over staleness, replay intensity, and behavior-policy mismatch, measuring their impact on stability, efficiency, and accuracy. A deliberately secondary component is representation — building the replay-buffer RL framework itself as a computational tool for producing these observations; it is a prerequisite, not a goal.

### 7. Resources & constraints

- **Compute**: single consumer GPU (e.g., RTX 3090/4090 class); no multi-GPU or cloud cluster access

### 8. Validation philosophy

Results are validated through controlled experimental sweeps where off-policiness knobs are systematically varied. A valid finding must demonstrate consistent patterns across multiple hyperparameter configurations — not a single lucky seed. Stability is defined operationally: accuracy remains within a bounded margin of its peak for the remainder of training, policy entropy stays above a floor, effective sample size stays above a threshold, and KL to the reference policy stays bounded — absence of NaN/Inf events and bounded gradient norms are necessary but not sufficient conditions. Accuracy improvements must be reproducible across at least 3 random seeds with statistically significant separation (non-overlapping confidence intervals). Diagnostic metrics (ESS, importance weight variance, staleness) must be logged and reported as evidence connecting observed behavior to measured causes.

Two calibration gates must be passed before any conclusions are drawn: (a) the on-policy baseline reaches a non-saturating accuracy band on the task (leaving measurable headroom); (b) the testbed reproduces at least one known qualitative result from the off-policy RL literature (e.g., IS variance growth with staleness, or clipping restoring stability), establishing that the toy setup behaves consistently with established knowledge.

### 9. Roadmap

#### 9.1 Milestones

| ID | Name | Expected result | Time index | Strong scaling efficiency |
|-----------|-------|-----------------|------------|---------------------------|
| M1 | Synthetic NLI dataset + RL harness | Testbed capability established — on-policy REINFORCE baseline reaches a non-saturating accuracy band on synthetic NLI, and staleness is demonstrably controllable and measurable, establishing that off-policiness effects are observable in this setup. No RQs answered yet — representation milestone enabling the empirical work. | T+3 | 0.7 |
| M2 | Initial off-policy characterization | RQ1 partially answered — demonstrate that controllable off-policiness sources produce measurable effects on training stability and accuracy; identify baseline instability regimes and which corrections (if any) prevent collapse. | T+5 | 0.3 |
| M3 | Hyperparameter sensitivity and correction comparison | RQ1 answered — individual and joint effects of off-policiness sources fully characterized; RQ2 answered — map hyperparameter regions (learning rate, batch size, replay ratio) that are robust vs fragile; RQ3 answered — establish the conditions under which each correction succeeds or fails under high staleness, the mechanism explaining the boundary, and the compute cost trade-offs. | T+9 | 0.3 |

#### 9.2 Long-term vision

Extend the off-policy diagnostic framework beyond single-step bandits to short- and medium-horizon decision tasks, building a reusable characterization toolkit applicable across model scales and task families. A longer-term goal is to produce practical guidance for practitioners choosing between on-policy and off-policy methods at different compute budgets, and to understand whether corrections that stabilize toy-scale models generalize to larger LLMs or break down as model capacity increases.
