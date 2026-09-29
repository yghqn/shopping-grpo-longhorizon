# 40-task unseen development holdout

This protocol validates the candidate-memory design selected on the 10-task
smoke set. It remains a development evaluation and does not touch Final-200
Clean.

## Data boundary

`data/grpo/validation.jsonl` contains 50 tasks. Ten were selected and repeatedly
used during smoke debugging, so reporting all 50 as unseen would leak tuning
tasks into validation. The holdout therefore contains every remaining task:

- tuning: `data/budget_memory/dev_smoke.jsonl`, 10 tasks;
- unseen holdout: `data/budget_memory/dev_holdout_40.jsonl`, 40 tasks;
- their intersection is empty and their union is the complete GRPO validation
  split;
- the holdout has zero task-ID overlap with GRPO training, SFT training, SFT
  validation and Final-200 Clean.

The frozen holdout SHA256 is
`5c80820f3a0f692bac8b5241c138a1b3e5bef90fa027c8e818acc142e10a7af6`.

## Frozen arms

| Arm | Candidate memory | Cumulative input-token ceiling | Purpose |
| --- | --- | ---: | --- |
| `baseline` | off | none | Original actor behavior |
| `budget-only` | off | 100,000 | Isolate the resource guardrail |
| `memory-v4` | 1,024-token view | 100,000 | Test memory with the same guardrail |

All arms use Qwen3.5-2B, temperature 0, top-p 1, 35 environment steps, a
24,576-token context window, the same observation projection, tool schemas,
environment and Reward v3 contract. Missing and failed tasks remain in the
fixed denominator of 40.

## Cloud command

Start ShopSimulator on port 5700 and the `shopping-agent` vLLM service on port
8000, then run:

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
time bash scripts/dev_holdout.sh all
```

The command is resumable. It writes each arm under
`outputs/budget-memory/dev-holdout-40/` and then creates:

- `comparison.json`: machine-readable aggregates, paired task transitions,
  Wilson intervals and deterministic paired-bootstrap intervals;
- `comparison.md`: reviewer-readable result tables.

To resume or inspect one stage:

```bash
bash scripts/dev_holdout.sh baseline
bash scripts/dev_holdout.sh budget-only
bash scripts/dev_holdout.sh memory-v4
bash scripts/dev_holdout.sh compare
```

## Decision rule

The holdout is descriptive because 40 tasks still provide a wide success-rate
interval for a 2B model. Memory v4 is retained for the next stage only if it
does not introduce strict-success regressions relative to both controls and its
cost/legality tradeoff remains explainable. The report must show confidence
intervals and paired task transitions; a single percentage is insufficient.

Do not tune v4 on these 40 tasks and rerun them as if they remained unseen. Any
new method choice after inspecting holdout outcomes requires a new development
split before making a fresh comparative claim.
