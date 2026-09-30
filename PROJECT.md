# 预算感知购物 Agent

本目录是基于 Shopping GRPO 的独立改造项目，遵循 `Baseline → SFT → GRPO → Evaluation`。当前已在单张 RTX 4090 24GB 上完成 Qwen3.5-2B 三轮 QLoRA SFT、checkpoint 选择和预算消融；下一阶段配置已冻结为 budget-only GRPO，但尚未启动正式 GRPO 或 Final-200 Clean。

研究主题：在长程购物任务中比较成功率与累计输入 token、工具调用成本之间的关系。计算预算与用户的商品价格预算分开统计。

## 已完成

- 固定上游提交，来源和核心资产哈希见 `UPSTREAM.json`。
- 实现可追溯候选商品记忆、当前页面动作隔离、累计输入 token 统计与硬上限。
- 将候选记忆和累计输入 token 软惩罚接入 veRL 在线 rollout，并保持 token、mask 与 logprob 数组对齐。
- 将原先按 96GB GPU 验证的长上下文 SFT 适配到 RTX 4090：QLoRA + Liger、24,576 token、3 epochs、300 steps、约 4 小时、峰值显存 18.2 GiB。
- 配对比较 e1/e2/e3，在 40-task 开发集上选择 e3：19/40 严格成功、100% 完成率。
- e3 budget-only 保留相同 19 个成功任务，将总输入 token 降低 25.15%。完整证据见 `docs/sft-checkpoint-selection.md`。
- 增加 `grpo_budget_only_v1` 和 `grpo_budget_memory_v1` 两个显式训练 arm；前者是当前主实验，后者保留为负向消融。

## 关键结果

同一 40 个开发任务上，原始 Qwen3.5-2B 为 0/40 严格成功；SFT-e1 为 18/40，SFT-e2 为 17/40，SFT-e3 为 19/40。E3 的完成率为 100%，平均终局效用为 0.4136。最低 validation loss 出现在 e2，但真实环境指标由 e3 胜出，体现了 checkpoint 选择不能只看 token-level loss。

在 e3 上加入每轨迹 100,000 累计输入 token 硬上限后，19 个成功 task ID 完全不变，总输入 token 从 2,253,459 降到 1,686,792（-25.15%），平均步骤从 9.225 降到 8.35。三条触发上限的轨迹原本均为失败。

Candidate Memory 在弱 base model 上曾提供正向开发信号，但在 SFT-e1 上从 18/40 降到 14/40。失败覆盖错误商品和错误规格选择，因此默认关闭，不进入当前主实验。

GRPO 主实验 `grpo_budget_only_v1` 使用 e3 合并模型，关闭候选记忆，并对超过 60,000 的累计输入 token 施加最大 0.1 的封顶软成本。评测中的 100k 硬停止与 GRPO 软惩罚机制不同，当前结果只支持该方向值得训练验证，不能预先宣称 GRPO 收益。

## 数据边界

- `data/budget_memory/dev_smoke.jsonl` 的 10 条任务用于开发调试。
- `data/budget_memory/dev_holdout_40.jsonl` 已用于记忆设计、SFT checkpoint 和预算方案选择，因此属于开发数据。
- 40-task 与 GRPO train、SFT train/validation、Final-200 Clean 的 task ID 仍为零重叠。
- Final-200 Clean 只能在 GRPO 配置冻结后执行，且不用于调参。

## 贡献边界

购物环境、教师数据、SFT/GRPO 流程、已有奖励与评测来自上游。上游已经实现工具上下文截断、observation 投影和可选长度惩罚；这些不属于本项目新增贡献。

本项目新增的是候选证据记忆及其训练接入、动作合法性隔离、累计输入成本统计/约束、24GB 训练适配、配对消融和失败诊断。当前结论是：SFT 显著恢复了 2B 模型的工具策略；简单候选记忆会干扰 SFT 后策略；预算约束可在开发任务上保留成功集合并削减失败轨迹成本。

上游当前提交未发现 LICENSE/COPYING 文件。保留来源信息；对外发布代码前需要核实源码与嵌入数据的再分发条款。
