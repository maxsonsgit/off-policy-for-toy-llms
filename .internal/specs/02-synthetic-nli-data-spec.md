## Synthetic NLI dataset

### 1. Executive summary

#### 1.1 Spec description

This data spec produces a synthetic NLI dataset with a hidden symbolic world model, supporting controlled difficulty axes (hop difficulty 0–2, distractor difficulty 1–3, background knowledge explicit/implicit). It relates to the constitution's RQ1 by providing the training and evaluation data on which off-policy RL experiments are run — a dataset where labels are correct by construction and difficulty can be systematically varied, enabling clean isolation of off-policiness effects from dataset noise.

#### 1.2 Spec motivation

Existing NLI datasets (SNLI, MNLI) are unsuitable for our tight-compute, controlled-experiment setting: they have noisy human annotations, no controllable difficulty axes, and no OOD split guarantees. A synthetic dataset with a hidden symbolic world provides exact labels by construction, known reasoning depth, configurable OOD regimes, and full reproducibility — all essential for clean off-policy RL experiments where confounders must be minimized.

### 2. Data specification

#### 2.1 Schema

Each example is a JSON object with required fields:
- `id`: unique example identifier
- `premise`: rendered premise text (multiple sentences)
- `hypothesis`: rendered hypothesis text (one sentence)
- `gold_label`: one of `entailment`, `neutral`, `contradiction`
- `split`: `train` or `eval` (optional: `test`)
- `hop_difficulty`: integer in `{0,1,2}`
- `distractor_difficulty`: integer in `{1,2,3}` (number of irrelevant premise facts)
- `background_knowledge`: one of `explicit`, `implicit`

Optional but recommended metadata fields:
- `world_seed`: seed used to generate the hidden world (for reproducibility)
- `templates_version`: identifier of the rendering template set
- `ood_tag`: optional string describing the split regime (e.g., `iid`, `entity_heldout`, ...)

Output format: JSONL

#### 2.2 Volume and scale

Approximate estimates (to be finalized during generation):
- Training split: ~100K–500K examples
- Eval split: ~5K–10K examples
- Optional test split: ~5K examples
- Target label balance: roughly uniform across entailment/neutral/contradiction
- Target hop difficulty distribution: configurable, default mixture over {0, 1, 2}
- Target distractor difficulty distribution: configurable, default mixture over {1, 2, 3}

#### 2.3 Sampling strategy

Deterministic generation from a hidden symbolic world model with configurable parameters:
- Number of entities (`n_entities`), fact families (attributes, taxonomy, binary relations)
- Taxonomy DAG structure (synthetic type names `t0`, `t1`, ...)
- Attribute types (color, location, shape — single-valued per entity)
- Relation type: `left_of` with transitivity inference rule
- Rendering templates per fact type (multiple templates for variety, deterministic given seed)
- Inclusion criteria: examples are included if they pass validation checks (label correctness, hop difficulty enforcement, distractor correctness)

### 3. Collection / generation protocol

**Generation tool**: `slam-datagen` (https://github.com/anton-pershin/slam-datagen)

**Protocol**:
1. Initialize a hidden symbolic world: sample entities, assign attributes (color, location, shape), sample taxonomy DAG edges, sample binary relation facts (`left_of` pairs)
2. For each example:
   - Select supporting facts from the world consistent with the target `hop_difficulty` and `background_knowledge` mode
   - Sample `distractor_difficulty` irrelevant facts from the world
   - Construct hypothesis and assign `gold_label` by construction (entailment/neutral/contradiction per the label rules)
   - Render premise and hypothesis text using deterministic templates
3. Write output as JSONL to the specified output file

### 4. Quality criteria

**Dataset-level validation**:
- Label proportions match configured targets (within tolerance)
- Balance across hop difficulty buckets {0, 1, 2}
- Balance across distractor difficulty buckets {1, 2, 3}
- No duplicate examples (same premise + hypothesis pair)

### 5. Annotation / labeling plan

#### 5.1 Labeling scheme

Labels are assigned by construction during generation, not by human annotation:
- `entailment`: hypothesis is provable from the premise using the allowed inference rules (for hop 1/2, exactly the intended number of reasoning steps)
- `contradiction`: hypothesis contradicts the premise by violating a mutually-exclusive, single-valued attribute constraint (no explicit negation — e.g., premise implies `color(e)=red`, hypothesis states `color(e)=blue`)
- `neutral`: hypothesis is not provable from the premise and does not contradict it; the premise does not determine the queried attribute/relation

#### 5.2 Annotation process

N/A — labels are deterministically assigned by the generator from the symbolic world model. Each example's label is derived from the underlying facts and inference rules before rendering.

### 6. Implementation plan

#### 6.1 Implementation repos

- `slam-datagen` — dataset generator module, script, and Hydra config
- `off-policy-for-toy-llms` — consumes the generated JSONL files for RL experiments

#### 6.2 Deliverables

- Synthetic NLI dataset generator module in `slam-datagen`
- Generated JSONL files (train/eval/test splits) stored via DVC in `off-policy-for-toy-llms/data/`
- Generation script and Hydra config for reproducibility (in `slam-datagen`)

#### 6.3 Todo list

1. Implement `synthetic_nli.py` module in `slam-datagen` with world model, example construction, and rendering
2. Add Hydra config and generation script
3. Generate dataset splits per Section 2.2 volume targets
4. Run dataset-level validation per Section 4
5. Store generated files via DVC in `off-policy-for-toy-llms/data/`
