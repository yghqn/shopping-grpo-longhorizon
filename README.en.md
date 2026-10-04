# Reproducible Post-training and Evaluation for Long-horizon Shopping Agents

[简体中文](README.md) · **English**

An **Agentic RL reproduction and extension** built on Qwen3.5-2B and
[Shopping GRPO](https://github.com/YYHDBL/shopping-grpo-longhorizon).
The supported workflow is **Baseline → SFT → GRPO → Evaluation**.

This branch studies task success, trajectory quality and cumulative input-token cost.
The environment, teacher data and basic training/evaluation design come from upstream.
The work here covers single-GPU adaptation, training/export reliability, budget and
candidate-memory ablations, and evaluation of the resulting checkpoints.
See [UPSTREAM.json](UPSTREAM.json) for the pinned source and asset hashes.

## Results from this branch

All three models were evaluated on the same 200 Final-200 Clean tasks.
Strict success requires a complete `gold_purchase` terminal result with
`reward_valid=true`. Errors and missing tasks remain in the denominator.

| Model | Strict successes | Success rate | Mean terminal utility | Mean environment steps | Total input tokens |
|---|---:|---:|---:|---:|---:|
| SFT-e3 | 139/200 | 69.5% | 0.6024 | 9.825 | 12,081,725 |
| Standard GRPO · step 450 | **142/200** | **71.0%** | **0.6279** | 10.095 | 13,446,494 |
| Budget-only GRPO · step 450 | 139/200 | 69.5% | 0.5935 | 9.945 | 12,938,476 |

- Standard GRPO gains 12 tasks and loses 9 relative to SFT: +3 net successes,
  +11.30% input tokens, exact McNemar `p=0.6636`. The success difference is
  **not statistically significant**.
- Budget-only uses 3.78% fewer input tokens than standard GRPO, but loses
  1.5 percentage points of success and 0.0344 mean utility.
- Relative to SFT, budget-only has the same success rate and uses 7.09% more tokens.

These results do not establish a superior budget-aware policy or superiority over
upstream results. Aggregates, paired tests and artifact hashes are available in
[the result summary](experiments/final200-comparison.json).

### Scope and limitations

Both GRPO runs reached step 500 from the same SFT-e3 starting policy; the table
evaluates step 450. Standard GRPO's highest-validation checkpoint, step 300, was
removed by retention. Step 450 was chosen among the remaining candidates, and is
not the best checkpoint across the entire standard run.

The budget run used staged resumes and a different scheduling history. There are
no multi-seed replications, so this is a paired checkpoint comparison rather than
a controlled causal estimate of the token penalty. All final evaluations disable
candidate memory and cumulative-input hard stops. Development-time hard stops
and training-time soft penalties are different interventions.

Earlier evaluations of incompletely merged GRPO exports are excluded.

## Contribution boundary

| Area | Upstream foundation | Work in this branch |
|---|---|---|
| Environment and data | ShopSimulator, Reward v3, teacher trajectories, observation projection | Pinned runtime/data hashes and task-disjointness checks |
| SFT | Action-only LoRA supervision | 24GB QLoRA/Liger adaptation and development-set checkpoint comparison |
| GRPO | veRL, tool integration and group-relative optimization | Output-head/entropy memory fixes, sequence alignment, sampling diagnostics, resume/export fixes |
| Budget and memory | Context truncation and existing length penalties | Candidate evidence memory, action isolation and cumulative-input cost ablations |
| Evaluation | Deterministic reward and Curator/Judge design | Branch-specific extraction/prompt revisions, validation, frozen shared rubrics and three-model evaluation |

Budget refers to **cumulative actor input tokens**, not the user's shopping price
limit. No separate reward model was trained. LLM-as-Judge is an offline evaluation
component, not the reward used for these GRPO runs.

## Training design

The actor searches, inspects evidence, selects product variants and purchases through
multi-turn tool calls. The runtime contract is **Environment v2.1 / Reward v3 /
observation v2 / tool schema v2**.

| Setting | Executed configuration |
|---|---|
| Base model | Qwen/Qwen3.5-2B |
| SFT data | 800 train / 200 validation; 799 train trajectories retained at 24,576 tokens |
| SFT recipe | QLoRA, Liger, bf16, gradient checkpointing; 3 epochs; batch 1, accumulation 8 |
| SFT hardware | RTX 4090 24GB; 18.2 GiB peak; approximately 4 hours |
| GRPO data | 1,000 train tasks / 50 validation tasks |
| Sampling | 4 trajectories per prompt; prompt batch 2; at most 3 generation batches per attempt |
| Optimization | LoRA, learning rate `1e-6`, symmetric PPO clipping; KL loss/reward disabled |
| Memory controls | Torch fused output head, chunked entropy, checkpointing, remove padding, microbatch 1 |
| GRPO hardware | RTX PRO 6000 Blackwell 96GB; both runs reached step 500 |
| Budget penalty | Above 60,000 cumulative input tokens: 0.001 per 1,000 tokens, capped at 0.1 |
| Final evaluation | 200 tasks; temperature 0; 35 environment steps; 24,576-token context |

SFT masks user and tool-observation tokens. Dynamic sampling rejects equal-reward
and sampling-invalid groups and records resampling and skipped attempts.
Skipped attempts are distinct from optimizer updates.

The 40-task development set was used for checkpoint and method selection; it is
not an unseen test set. Candidate memory reduced SFT-e1 success from 18/40 to 14/40
and was disabled in the two final GRPO arms.
See [SFT selection](docs/sft-checkpoint-selection.md),
[memory ablation](docs/sft-e1-ablation.md) and [data isolation](docs/evaluation-dataset.md).

## LLM-as-Judge

Following the upstream two-role design:

1. **Rubric Curator (DeepSeek Flash)** selects constraints from code-extracted
   candidates and freezes 1,347 hard/soft rubrics for 200 tasks, shared across actors.
2. **Trajectory Judge (DeepSeek V4 Pro)** evaluates constraints and five dimensions
   on a 0/1/2 scale using actor-visible trajectories, neutral terminal state and
   whitelisted metrics. Private Gold fields, reward, strict-success labels and
   other actors' results are excluded.
3. **Code validation and aggregation** check structured outputs and evidence
   references, keeping reward, rubric, trajectory and deterministic panels separate.

The run made 200 Curator and 592 Judge requests. Eight infrastructure-invalid
trajectories were marked `not_judged` locally and remain in the fixed denominators.
Dimension means below are conditional on judged trajectories.

| Actor | Judge coverage | Search | Candidate use | Evidence | Decision | Termination |
|---|---:|---:|---:|---:|---:|---:|
| SFT-e3 | 198/200 | 1.793 | 1.783 | 1.662 | 1.545 | 1.697 |
| Standard GRPO | 197/200 | 1.822 | 1.797 | 1.695 | 1.548 | 1.726 |
| Budget-only GRPO | 197/200 | 1.802 | 1.746 | 1.635 | 1.508 | 1.660 |

The most common primary error for all actors is `wrong_option`.
Judge scores supplement environment success; they have not been independently
validated against human annotations. Prompts, versions and aggregation rules are
in [the evaluation guide](docs/evaluation.md).

## Reproduction entry points

Training requires Linux, an NVIDIA GPU, Python 3.12 for the main environment
and Python 3.10 for ShopSimulator. Disk use scales with checkpoint retention:
each GRPO checkpoint in these runs was approximately 11GB.
Weights, full trajectories and logs are not included in Git; a local artifact
path does not imply a public download.

Installation, training, merging, environment evaluation and paid Judge calls are
separate actions. Cloning the repository does not start them.

```bash
bash scripts/setup.sh
uv sync --extra sft --extra grpo --extra sft-accelerated
bash scripts/start_environment.sh
```

Keep the environment service running. Run the model server in a separate terminal
and evaluation in another; stop the standalone model server before training.

| Stage | Entry point |
|---|---|
| Baseline | `bash scripts/serve_model.sh Qwen/Qwen3.5-2B`; evaluate separately with `bash scripts/baseline.sh` |
| SFT | Explicit 24GB QLoRA recipe in [the Chinese README](README.md#baseline--sft--grpo--evaluation); see [recorded configuration](docs/sft-checkpoint-selection.md) |
| GRPO | `scripts/run_experiment.py`: `grpo_baseline` or `grpo_budget_only_v1`; see [GRPO guide](docs/grpo.md) |
| Evaluation | Serve a correctly merged checkpoint; run `bash scripts/evaluate.sh NAME` in another terminal |
| LLM-as-Judge | Prepare saved trajectories and invoke the API separately; see [evaluation guide](docs/evaluation.md) |

The plain `scripts/sft.sh` entry point does not automatically enable QLoRA/Liger.
The published 24GB result uses the explicit flags in the Chinese README.

Inspect a budget experiment without starting training:

```bash
.venv/bin/python scripts/run_experiment.py grpo_budget_only_v1 \
  --model outputs/models/sft-merged-e3-24576 --dry-run
```

Merge the SFT adapter before starting GRPO. Choose the GRPO checkpoint on validation
data and retain candidates until selection is complete. Export with
[scripts/export_grpo.sh](scripts/export_grpo.sh), which merges the separate LoRA
adapter into the top-level weights. Raw veRL model-merger output is an intermediate
artifact, not the final GRPO model to serve.

## Documentation and code

- [Project status and contribution boundary](PROJECT.md)
- [Data provenance](docs/data-collection.md) and [Final-200 Clean](docs/evaluation-dataset.md)
- [SFT configuration and selection](docs/sft-checkpoint-selection.md)
- [GRPO training/export](docs/grpo.md) and [Reward v3](docs/reward-v3.md)
- [Evaluation protocol](docs/evaluation.md) and [aggregate results](experiments/final200-comparison.json)
- [Training code](src/shopping_grpo/training/), [candidate memory](src/shopping_grpo/memory/),
  [evaluation code](src/shopping_grpo/evaluation/) and [tests](tests/)

Retained upstream experiment material is not a result of this branch.
Reproduction settings and artifact identity do not guarantee bit-identical results
across hardware, dependency versions or repeated runs.

## Attribution and license status

Based on [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)
at `449bba2686a6ff07327cb3ea77e5f0f91ebb4239`.
Thanks to [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator),
[veRL](https://github.com/verl-project/verl), [Qwen](https://github.com/QwenLM/Qwen3),
[PEFT](https://github.com/huggingface/peft) and
[Liger Kernel](https://github.com/linkedin/Liger-Kernel).

The pinned upstream source has no LICENSE/COPYING file; see [UPSTREAM.json](UPSTREAM.json).
This branch preserves attribution and does not assign a new open-source license to
upstream code or embedded data.
