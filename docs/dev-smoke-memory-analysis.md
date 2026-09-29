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
