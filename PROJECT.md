# 面向长程购物 Agent 的可复现后训练与评测（Agentic RL）

本目录是基于 Shopping GRPO 的独立改造项目，遵循 `Baseline → SFT → GRPO → Evaluation`。当前已完成 Qwen3.5-2B 三轮 QLoRA SFT、标准与预算感知 GRPO、checkpoint 选择和 Final-200 Clean 配对评测。SFT 在单张 RTX 4090 24GB 上完成，24K 长轨迹 GRPO 在单张 RTX PRO 6000 96GB 上完成。

研究主题：在长程购物任务中比较成功率与累计输入 token、工具调用成本之间的关系。计算预算与用户的商品价格预算分开统计。

## 已完成

- 固定上游提交，来源和核心资产哈希见 `UPSTREAM.json`。
- 实现可追溯候选商品记忆、当前页面动作隔离、累计输入 token 统计与硬上限。
- 将候选记忆和累计输入 token 软惩罚接入 veRL 在线 rollout，并保持 token、mask 与 logprob 数组对齐。
- 将原先按 96GB GPU 验证的长上下文 SFT 适配到 RTX 4090：QLoRA + Liger、24,576 token、3 epochs、300 steps、约 4 小时、峰值显存 18.2 GiB。
- 配对比较 e1/e2/e3，在 40-task 开发集上选择 e3：19/40 严格成功、100% 完成率。
- e3 budget-only 保留相同 19 个成功任务，将总输入 token 降低 25.15%。完整证据见 `docs/sft-checkpoint-selection.md`。
- 增加 `grpo_budget_only_v1` 和 `grpo_budget_memory_v1` 两个显式训练 arm；前者是当前主实验，后者保留为负向消融。
- 在相同 SFT-e3 起点上完成 500 个有效更新的标准 GRPO 与 budget-only GRPO，并用独立的 Final-200 Clean 做最终比较。
- 修复 Qwen3.5 大词表 PPO 输出头、entropy 显存、checkpoint resume 和 veRL 分离 LoRA 导出问题；最终可服务模型会真正合并 GRPO adapter。
- 沿用上游 Curator / Judge 评测设计，完成本分支规则提取、输入校验与正式运行：200 题共享 1,347 条冻结 Rubric，三组轨迹共调用 Judge 592 次。

## 关键结果

同一 40 个开发任务上，原始 Qwen3.5-2B 为 0/40 严格成功；SFT-e1 为 18/40，SFT-e2 为 17/40，SFT-e3 为 19/40。E3 的完成率为 100%，平均终局效用为 0.4136。最低 validation loss 出现在 e2，但真实环境指标由 e3 胜出，体现了 checkpoint 选择不能只看 token-level loss。

在 e3 上加入每轨迹 100,000 累计输入 token 硬上限后，19 个成功 task ID 完全不变，总输入 token 从 2,253,459 降到 1,686,792（-25.15%），平均步骤从 9.225 降到 8.35。三条触发上限的轨迹原本均为失败。

Candidate Memory 在弱 base model 上曾提供正向开发信号，但在 SFT-e1 上从 18/40 降到 14/40。失败覆盖错误商品和错误规格选择，因此默认关闭，不进入当前主实验。

Final-200 Clean 的固定分母结果为：SFT-e3 139/200（69.5%），标准 GRPO step 450 为 142/200（71.0%），budget-only GRPO step 450 为 139/200（69.5%）。标准 GRPO 相对 SFT 净增 3 个成功任务、平均终局效用增加 0.0255，但输入 token 增加 11.30%；配对 McNemar `p=0.6636`，只支持小幅正向趋势。

Budget-only 相对标准 GRPO 减少 3.78% 输入 token 和 31 次 Guard 拒绝，但丢失 3 个净成功任务，终局效用下降 0.0344；相对 SFT 则成功数不变、输入 token 增加 7.09%。因此简单的封顶 token 软成本没有形成更优 Pareto 点，当前最佳成功率模型是标准 GRPO，最佳成本效益基线仍是 SFT-e3。

完整聚合指标、逐任务配对检验与文件哈希见 [正式结果摘要](experiments/final200-comparison.json)。表中 GRPO 均评估 step 450；标准组验证分最高的 step 300 未保留，step 450 仅为剩余候选中的较优检查点。Budget 组经历分段续训，且未开展多随机种子复验，因此不能将当前比较解释为预算惩罚的严格因果效应。

## 数据边界

- `data/budget_memory/dev_smoke.jsonl` 的 10 条任务用于开发调试。
- `data/budget_memory/dev_holdout_40.jsonl` 已用于记忆设计、SFT checkpoint 和预算方案选择，因此属于开发数据。
- 40-task 与 GRPO train、SFT train/validation、Final-200 Clean 的 task ID 仍为零重叠。
- Final-200 Clean 只能在 GRPO 配置冻结后执行，且不用于调参。

## 贡献边界

购物环境、教师数据、SFT/GRPO 流程、已有奖励与评测来自上游。上游已经实现工具上下文截断、observation 投影和可选长度惩罚；这些不属于本项目新增贡献。

本项目新增的是候选证据记忆及其训练接入、动作合法性隔离、累计输入成本统计/约束、24GB SFT 与 24K GRPO 适配、配对消融和失败诊断。最终结论是：SFT 恢复了 2B 模型的大部分工具策略；标准 GRPO 以更高推理成本换取小幅成功与效用增益；简单候选记忆会干扰策略，而当前 token 软成本只能收回部分 GRPO 成本，无法保留其成功率收益。

上游当前提交未发现 LICENSE/COPYING 文件。保留来源信息；对外发布代码前需要核实源码与嵌入数据的再分发条款。
