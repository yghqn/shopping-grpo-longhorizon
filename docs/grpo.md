# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
  memory_runtime.py     public-observation candidate-memory lifecycle
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | enabled (logging only) |

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

Each run also appends `training_diagnostics.jsonl` under its output directory.
`generation_batch` records contain every generated rollout, its public tool
sequence, terminal result, reward breakdown, Guard rejection reasons and group
keep/drop decision. `optimizer_step` records preserve the scalar veRL metrics,
including entropy, PPO KL, clip fractions, response lengths and effective-group
rates. `skipped_update` records make zero-signal attempts visible even though
they do not advance the optimizer step.

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Budget-aware training arms

The paired checkpoint evaluation selects the 24,576-token SFT-e3 model as
the initial policy and `grpo_budget_only_v1` as the primary training arm. The
arm keeps candidate memory disabled and adds a bounded cost for cumulative
actor input above 60,000 tokens:

```text
min(max(cumulative_input_tokens - 60000, 0) / 1000 * 0.001, 0.1)
```

The 40-task evaluation used a 100,000-token hard trajectory ceiling, while
GRPO uses this soft, capped reward cost. They target the same resource metric
but are not equivalent interventions. The evaluation result motivates testing
the soft cost; it does not establish its training benefit in advance.

`grpo_budget_memory_v1` remains available as an ablation. It additionally
injects a 1,024-token candidate-memory view built from public observation v2.
On the SFT-e1 development run this reduced strict success from 18/40 to 14/40,
so it is not the recommended primary arm. The online AgentLoop integration and
audit records remain covered by tests for future redesign work.

Resolve either command without starting training:

```bash
python scripts/run_experiment.py grpo_budget_only_v1 \
  --model outputs/models/sft-merged-e3-24576 \
  --dry-run
python scripts/run_experiment.py grpo_budget_memory_v1 \
  --model outputs/models/sft-merged-e3-24576 \
  --dry-run
```

The branch contains the training integration and dry-run configurations, but
no local-modification GRPO checkpoint has been trained yet. See
[`sft-checkpoint-selection.md`](sft-checkpoint-selection.md) for checkpoint
selection, budget evidence and the holdout-reuse boundary.

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.
