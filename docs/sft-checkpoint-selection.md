# SFT checkpoint 选择与预算消融

本页记录在单张 RTX 4090 24GB 上完成的 Qwen3.5-2B SFT-e3，以及用于选择 GRPO 初始策略的 40-task 配对开发评测。该评测集已经用于 checkpoint 和方法选择，不能再作为新方案的未见测试集。

## 训练配置与资源

训练从 `Qwen/Qwen3.5-2B` 开始，使用历史 `data/sft/` 教师轨迹：

- QLoRA + Liger kernel，BF16 计算；
- 最大序列长度 24,576；
- batch size 1，梯度累积 8；
- LoRA rank 16 / alpha 32；
- learning rate `1e-4`；
- 799/800 条训练样本保留；
- 3 epochs，300 optimizer steps；
- 总训练约 4 小时，峰值显存 18.2 GiB；
- 最终 train loss 0.3172。

三个 epoch 的 validation loss 分别为 0.3521、0.3396、0.3415。Epoch 2 的 loss 最低，但 checkpoint 选择同时使用真实环境开发任务，避免把 token-level loss 当作最终策略质量。

## Checkpoint 配对比较

| Checkpoint | 严格成功 | Done | 平均终局效用 | 平均步骤 | Guard 拒绝 | 输入 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-e1，12,288 | 18/40 | 95.0% | 0.3397 | 8.675 | 13 | 2,208,741 |
| SFT-e2，24,576 | 17/40 | 92.5% | 0.3279 | 10.100 | 2 | 2,703,609 |
| SFT-e3，24,576 | **19/40** | **100%** | **0.4136** | 9.225 | 6 | 2,253,459 |

相对 e1，e3 新增 task 8107、10077、19606，丢失 task 2898、18534，净增加一个严格成功。Epoch 2 虽有最低 validation loss，但环境成功率、效用、完成率和成本均弱于 epoch 3，因此选择 e3，而不是机械选择最低 loss checkpoint。

## E3 budget-only 消融

| Arm | 严格成功 | Done | 平均终局效用 | 平均步骤 | Guard 拒绝 | 输入 token | Budget exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e3 baseline | 19/40 | 100% | 0.4136 | 9.225 | 6 | 2,253,459 | 0 |
| e3 + 100k hard ceiling | **19/40** | 92.5% | **0.4413** | **8.350** | **4** | **1,686,792** | 3 |

Budget-only 保留完全相同的 19 个严格成功任务，同时减少 566,667 输入 token（-25.15%）。三条触发上限的轨迹原本都不是严格成功，所以这次硬上限截断的是高成本失败，没有误伤成功任务。

评测使用 100,000 token 硬停止；命名 GRPO 实验 `grpo_budget_only_v1` 使用 60,000 token 后开始、最大 0.1 的封顶软奖励成本。二者针对同一成本指标，但机制不同。开发结果支持训练这个方向，不能预先等同为 GRPO 收益。

## 最终选择

- GRPO 初始策略：`outputs/models/sft-merged-e3-24576`
- 主实验：`grpo_budget_only_v1`
- Candidate Memory：默认关闭，`grpo_budget_memory_v1` 仅保留为负向消融
- Final-200 Clean：已在配置冻结后完成，SFT / 标准 GRPO / Budget-only GRPO 分别为 139 / 142 / 139 个严格成功；见[正式结果摘要](../experiments/final200-comparison.json)

Candidate Memory 的失败分析见 [`sft-e1-ablation.md`](sft-e1-ablation.md)：在 SFT-e1 上从 18/40 降到 14/40，说明简单的搜索排名/已核验优先记忆会干扰已学到的商品与规格决策。

## 归档

云端产物保存在 `outputs/cloud-runs/`：

- `2026-09-30-sft-e3-24576-results.tar.gz`，SHA256 `e87bf412ffd45bc10b4cfc8869dc65efd49d54074787d32c45de3043d1f335f1`；
- `2026-09-30-sft-e3-24576-adapter.tar.gz`，SHA256 `bd5bff75db503c058408cf172b4c3f6ab04e7bdfd1099600ed52edef3ef7f638`。

Adapter 归档包含 epoch 1/2/3 checkpoint 与最终 adapter。合并后的 4.2GB BF16 模型不重复归档，可用基础模型和 adapter 重建。
