# Data

Only the datasets used by the tutorial are kept here.

| Stage | Files | Rows |
|---|---|---:|
| SFT | `sft/train.jsonl`, `sft/validation.jsonl` | 800 / 200 |
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` | 1000 / 50 |
| Budget-memory development | `budget_memory/dev_smoke.jsonl`, `budget_memory/dev_holdout_40.jsonl` | 10 / 40 |
| Evaluation | `evaluation/tasks.jsonl` (Final-200 Clean) | 200 |

Adjacent `metadata.json` files record SHA256 checksums and collection
provenance. All SFT, GRPO and evaluation splits are task-disjoint. Generated
trajectories belong under `outputs/`, never under `data/`. Use
`scripts/collect_sft_data.py` to create a new audited SFT dataset before
promoting its train/validation files into this directory.

The budget-memory files partition the existing 50-task GRPO validation split;
they are not additional training data. The 10-task smoke partition was used for
method selection, while the remaining 40 tasks are frozen as an unseen
development holdout.
