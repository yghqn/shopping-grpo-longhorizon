# Experiments

This directory contains the compact, reviewable artifacts behind the README
result table. Large checkpoints and complete trajectories are intentionally not
stored in Git.

```text
baseline/   base-model evaluation config and summary
sft/        SFT training/evaluation config and summary
grpo/       GRPO training/evaluation config and summary
budget_memory/  local candidate-memory development holdout summary
sft_e3_4090/  local 24GB SFT checkpoint selection and budget summary
comparison.md
```

All reported models use the same 200 held-out tasks, Environment v2.1 and
Reward v3. See [comparison.md](comparison.md) for interpretation and protocol
limitations.

`budget_memory/` and `sft_e3_4090/` use the separate 40-task development set.
They are not part of the upstream Final-200 model comparison; the latter records
the local 4090 SFT checkpoint selection and budget-only decision.
