# Candidate memory dev-smoke analysis

This note records the first paired cloud run of the Qwen3.5-2B shopping actor.
It is a development diagnostic, not a benchmark claim. The run used the same
10-task `data/budget_memory/dev_smoke.jsonl` slice, temperature 0, a 24,576-token
context window and a 1,024-token candidate-memory budget.

The archived raw artifacts are stored locally as
`outputs/cloud-runs/2026-09-29-dev-smoke.tar.gz`. Its SHA256 is
`47253d9826753fb00066ccd636075d577bf92e335b6ee11738f7ce1be868e307`.

## Aggregate comparison

| Metric | Baseline | Candidate memory v1 | Change |
| --- | ---: | ---: | ---: |
| Strict success | 0/10 | 0/10 | no change |
| Purchase success | 0/10 | 0/10 | no change |
| Environment `done` | 1/10 | 4/10 | +3 tasks |
| `invalid_action_limit` | 9/10 | 4/10 | -5 tasks (-55.6%) |
| Guard rejections | 40 | 24 | -16 (-40.0%) |
| Executed steps | 53 | 51 | -2 (-3.8%) |
| Cumulative input tokens | 445,848 | 425,995 | -19,853 (-4.5%) |
| Mean cumulative input tokens/task | 44,584.8 | 42,599.5 | -4.5% |
| Maximum single-turn input tokens | 8,227 | 10,187 | +23.8% |
| Mean final reward | -0.065 | -0.181 | -0.116 |

The process metrics improved, but the task objective did not. The lower mean
reward is not directly comparable to the baseline's many unscored guard aborts:
the memory arm reached four reward-valid terminal outcomes while the baseline
reached one. Three of those additional terminal outcomes were `repeat_loop`
penalties rather than successful purchases.

## Paired outcomes

| Task | Baseline | Memory v1 | Guard calls B/M | Input tokens B/M | Interpretation |
| ---: | --- | --- | ---: | ---: | --- |
| 643 | invalid action | invalid action | 8 / 4 | 112,316 / 39,142 | fewer retries, still used stale controls |
| 18176 | invalid action | partial alternative purchase | 4 / 1 | 31,822 / 41,864 | only substantive task-level improvement |
| 4199 | invalid action | invalid action | 3 / 3 | 29,049 / 22,655 | historical ASIN chosen off-page |
| 16715 | invalid action | assistant final | 5 / 1 | 77,543 / 28,189 | stopped early without environment termination |
| 4128 | repeat loop | repeat loop | 1 / 3 | 36,228 / 69,573 | worse search/back loop and higher cost |
| 2511 | invalid action | invalid action | 4 / 4 | 33,341 / 40,614 | no legality improvement |
| 20304 | invalid action | repeat loop | 4 / 3 | 32,588 / 61,022 | converted guard abort into penalized loop |
| 19430 | invalid action | repeat loop | 4 / 0 | 32,232 / 39,838 | legal actions improved, goal did not |
| 2718 | invalid action | invalid action | 3 / 4 | 27,546 / 52,963 | repeated stale purchase/option actions |
| 16263 | invalid action | assistant final | 4 / 1 | 33,183 / 30,135 | stopped early without environment termination |

## Diagnosis

Candidate memory reduced repeated guard failures and total input-token spend by
shortening several failed trajectories. It also retained enough product evidence
for task 18176 to re-search an 英氏-brand stroller and complete a partially
matching purchase.

The v1 prompt exposes historical ASINs as `候选 <asin>`. Qwen3.5-2B often copies
one of those identifiers into `open_product` after leaving the search-result
page. That conflicts with the environment contract, where only targets in the
latest observation are actionable. A similar state error occurs after an
information subpage: the actor selects `view_features`, `select_option` or
`buy_now` instead of first returning to the product-detail page.

The next development iteration should therefore:

1. keep exact ASINs and provenance in the saved audit snapshot;
2. replace prompt-visible ASINs with stable non-actionable labels such as H01;
3. add a small dynamic current-page action section listing the only legal ASINs
   and buttons for the next tool call;
4. prefer `prev_page` over `back_to_search` when leaving an information subpage;
5. repeat the paired 10-task smoke run before any SFT, GRPO or 200-task evaluation.

These results support a narrow claim: provenance-preserving candidate memory
improved interaction legality and reduced cumulative input-token use on this
small deterministic smoke set. They do not yet support a success-rate claim.

## Prepared v2 ablation

The next local revision implements items 1-4 above. The model-visible memory now
uses stable labels (`H01`, `H02`, ...) while the trajectory snapshot retains the
real ASINs. A separate `[CURRENT_ACTION_STATE_V1]` section is refreshed after
every environment step and lists the only legal ASINs and buttons for the next
call. This revision has passed the focused memory, rollout and CLI unit tests;
its effect still requires another paired cloud smoke run. Use
`bash scripts/dev_smoke.sh memory-v2` to keep the new trajectories separate
from the archived v1 output.

## Memory v2 cloud outcome

The v2 smoke run produced the first strict success on task 19430: the actor
searched once, opened the gold product, selected the 400g option and purchased
it in four steps. Task 16715 also reached a partial-alternative purchase.

| Metric | Baseline | Memory v1 | Memory v2 |
| --- | ---: | ---: | ---: |
| Strict success | 0/10 | 0/10 | 1/10 |
| Environment `done` | 1/10 | 4/10 | 6/10 |
| `invalid_action_limit` | 9/10 | 4/10 | 2/10 |
| Guard rejections | 40 | 24 | 33 |
| Cumulative input tokens | 445,848 | 425,995 | 1,276,508 |
| Context-overflow tasks | 0 | 0 | 2 |

The live-action capsule improved decision quality on two tasks, but a page-state
classification bug made it over-explore other tasks. The capsule inferred an
information subpage from the presence of `< prev`, while product-detail pages
also expose that button. This encouraged premature returns to search results.
Tasks 4128 and 16263 then repeated open/return or search cycles until their
single-turn context exceeded the 23,552-token input limit; together they
consumed 813,739 cumulative input tokens.

The prepared v3 reads the explicit `page_type` before giving navigation advice.
It also sets a 100,000 cumulative-input-token ceiling for the v3 smoke arm. The
two successful v2 trajectories used 17,703 and 90,185 cumulative tokens, so this
ceiling preserves the observed successes while bounding pathological loops.

## v3 diagnosis and selected v4

The v3 run removed context overflows and reduced total input to 587,123 tokens,
but strict success returned to 0/10. The token ceiling did not cause this
regression: task 19430 abandoned the already selected gold product after 60,357
tokens because the changed navigation text encouraged more comparison.

The final v4 ablation therefore restores the exact v2 decision prompt and keeps
only the 100,000-token trajectory guardrail. This isolates the resource control
from the policy prompt.

| Metric | Baseline | Memory v1 | Memory v2 | Memory v3 | Memory v4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict success | 0/10 | 0/10 | 1/10 | 0/10 | **1/10** |
| Gold purchase | 0 | 0 | 1 | 0 | **1** |
| Partial alternative purchase | 0 | 1 | 1 | 0 | **1** |
| Environment `done` | 1 | 4 | 6 | 4 | **5** |
| Invalid-action termination | 9 | 4 | 2 | 4 | **2** |
| Guard rejections | 40 | 24 | 33 | 24 | **23** |
| Context-overflow tasks | 0 | 0 | 2 | 0 | **0** |
| Budget-exhausted tasks | 0 | 0 | 0 | 2 | **3** |
| Cumulative input tokens | 445,848 | 425,995 | 1,276,508 | 587,123 | **640,131** |
| Wall time | 50.3s | 64.3s | 133.0s | 73.0s | **84.6s** |

V4 retains v2's observed strict and partial purchases while cutting cumulative
input tokens by 49.9%, guard rejections by 30.3%, and wall time by 36.4%. It is
the selected development configuration. The three budget-exhausted tasks remain
counted as failures; the ceiling is a cost and reliability guardrail, not a way
to improve the success denominator.

Raw cloud archives and checksums:

| Run | Archive | SHA256 |
| --- | --- | --- |
| Baseline + v1 | `2026-09-29-dev-smoke.tar.gz` | `47253d9826753fb00066ccd636075d577bf92e335b6ee11738f7ce1be868e307` |
| v2 | `2026-09-29-memory-v2.tar.gz` | `40f35cbdf4d7b9dd1cf6eaac3fb9637e9840e538cdbfc13fb9ed929dae79ce48` |
| v3 | `2026-09-29-memory-v3.tar.gz` | `379f2de470739bfe16f8702af7fe54dc9d3c9f3bb373456494aa21cfc0e70a6` |
| v4 | `2026-09-29-memory-v4.tar.gz` | `339b19a5044ef0c22952aaa5cc7682b18643fe62a21e8f9f46d8b0239ac6675c` |

These are small deterministic development runs. The defensible claim is that
v4 preserved the observed v2 success while bounding pathological token spend;
the 10% strict-success figure is not a formal benchmark estimate.
