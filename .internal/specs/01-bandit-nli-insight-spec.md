## Bandit-NLI off-policy landscape

### 1. Executive summary

#### 1.1 Spec description

This insight spec maps the existing literature on off-policy reinforcement learning with replay buffers, focusing on methods designed to handle distributional mismatch (importance sampling, clipped IS, KL regularization) and their documented behavior under varying degrees of staleness and replay intensity. It relates to the constitution's knowledge gap by establishing what is already known about off-policy corrections in classical RL domains, and where the gap lies when moving to small autoregressive LLMs. It directly supports RQ1 (off-policiness effects), RQ2 (hyperparameter sensitivity), and RQ3 (correction comparison) by identifying which correction mechanisms have proven effective in other domains and which remain untested for LLM-scale generation.

#### 1.2 Spec motivation

The knowledge gap identified in Section 3 of the constitution spec hinges on the claim that off-policy corrections have not been systematically characterized for small LLMs. Before running experiments, it is necessary to survey what classical RL and deep RL literature already says about importance sampling variance, staleness tolerance, and KL penalties — to avoid rediscovering known results and to isolate what is genuinely novel about the LLM regime. This survey will also surface terminology, diagnostics (ESS, drift metrics), and experimental design patterns that can be directly adopted.

### 2. Research questions

This spec addresses RQ1, RQ2, and RQ3 from the constitution spec. It maps the existing knowledge landscape around each: what is already known about off-policiness sources and their effects (RQ1), how hyperparameter choices modulate sensitivity to off-policiness (RQ2), and which correction methods and diagnostics have proven effective in non-LLM domains (RQ3).

### 3. Scope

**Inclusion**: Peer-reviewed papers and preprints on off-policy RL with replay buffers, importance sampling methods, KL-regularized RL, staleness/async RL, and policy gradient variance reduction. Domains: classical RL, deep RL for games, continuous control, and early LLM+RL work (RLHF, PPO-for-LLMs). Venues: NeurIPS, ICML, ICLR, AAAI, JMLR, arXiv (cs.LG, cs.AI, cs.CL). Time range: foundational methods (1990s–2010s) through current work (2024+).

**Exclusion**: On-policy-only methods without off-policy analysis, model-based RL without replay, RLHF work that does not report off-policy diagnostics (ESS, KL, staleness), and work exclusively on tabular/small-state MDPs where generation is not a factor.

### 4. Search strategy

**Databases**: arXiv (cs.LG, cs.AI, cs.CL).

**Keyword queries**:
- "off-policy reinforcement learning" AND ("replay buffer" OR "experience replay")
- "importance sampling" AND ("clipped" OR "weight clipping") AND "reinforcement learning"
- "KL regularization" AND "policy optimization" AND "off-policy"
- "staleness" AND "asynchronous" AND "reinforcement learning"
- "effective sample size" AND "importance weights" AND RL
- "off-policy" AND ("large language model" OR "language model" OR "RLHF")
- "policy gradient variance reduction" AND ("reinforcement learning" OR "actor-critic")

**Procedure**: Start with keyword search on arXiv (2015–2025). Filter by relevance to replay-based off-policy methods. Snowball forward/backward from key papers (e.g., IMPALA, R2D2, PPO, CPO, Retrace, V-trace, SAC). Include foundational importance sampling papers (precis, per-decision IS) for RQ3 context. Record search dates and result counts in the extraction table.

### 5. Analysis framework

**Extraction dimensions** (one row per paper in the extraction table):

1. **Method category**: off-policy correction type (naive replay, IS, clipped IS, KL regularization, trust region, other)
2. **Staleness model**: how staleness/off-policiness is introduced (async delay, replay buffer age, explicit behavior mismatch)
3. **Diagnostic metrics reported**: ESS, KL/JS divergence, weight variance, gradient norm, return/accuracy
4. **Domain**: tabular, Atari, MuJoCo, robotics, LLM, other
5. **Empirical claim**: what the paper claims about the correction's effectiveness (e.g., "clipped IS stabilizes training at high staleness")
6. **Failure conditions**: when the correction breaks down (if reported)
7. **Hyperparameter sensitivity**: whether the paper studies LR/batch-size interactions with off-policiness
8. **Compute constraints**: compute budget mentioned (relevant to our tight-compute setting)

**Coding rules**:
- If a paper uses multiple correction methods, create one row per method-dimension pair.
- If a metric is mentioned but not reported with values, record as "mentioned, not quantified."
- If results are ambiguous or conflicting within the paper, record the primary claim and note ambiguity in a "notes" column.

**Format**: Structured extraction table (CSV or spreadsheet) with columns matching the dimensions above, plus paper title, year, arXiv ID, and a free-text "notes" column.

### 6. Synthesis framework

**Organization**: Papers will be grouped by correction method (naive replay, IS-based, KL-based, trust region, hybrid) and further stratified by domain (classical RL vs LLM). Within each group, papers are compared on: (a) reported effectiveness of the correction, (b) staleness tolerance, (c) diagnostic metrics used, (d) failure conditions identified.

**Gap identification**: For each correction method, count the number of papers that study it in LLM-relevant settings (autoregressive generation, text output, compute-constrained). Flag correction methods with zero or one LLM-domain studies as gaps. Also flag diagnostic metrics that are standard in classical RL but never reported in LLM+RL work.

**Contradiction resolution**: When papers disagree on the same correction's effectiveness, record both results with full citations, note differences in domain, task, and compute setup, and mark as "unresolved discrepancy — requires experimental resolution."

**Narrative structure**: The final document follows: (1) taxonomy of off-policy corrections and how they address distributional mismatch, (2) empirical comparison matrix across domains and conditions, (3) gap analysis highlighting what is unknown for small LLMs, (4) synthesis of actionable design recommendations for the experimental program (which corrections to test first, which diagnostics to log, which hyperparameter ranges to prioritize).

### 7. Implementation plan

#### 7.1 Implementation repos

- `off-policy-for-toy-llms` (management repo; output goes to `progress/`)

#### 7.2 Deliverables

- Literature survey document in `progress/` with taxonomy, extraction table, gap analysis, and design recommendations

#### 7.3 Todo list

1. Run arXiv searches per Section 4 keywords, collect candidate papers
2. Apply inclusion/exclusion criteria from Section 3 to produce final paper set
3. Extract data from each paper per Section 5 dimensions into a structured table
4. Group, compare, and synthesize per Section 6 framework
5. Write the final survey document with taxonomy, gap analysis, and recommendations
