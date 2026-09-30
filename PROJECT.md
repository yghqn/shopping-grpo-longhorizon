# 预算感知购物 Agent

本目录是基于 Shopping GRPO 的独立改造项目，遵循 `Baseline → SFT → GRPO → Evaluation`。当前已完成 Qwen3.5-2B 的本地改造、单卡 SFT-e1 和四臂开发集消融；下一阶段配置已冻结为 budget-only GRPO，但尚未启动 GRPO 或 Final-200 Clean。

研究主题：在长程购物任务中比较成功率与累计输入 token、工具调用成本之间的关系。计算预算与用户的商品价格预算分开统计。

## 已完成

- 固定上游提交，来源和核心资产哈希见 `UPSTREAM.json`。
- 实现可追溯候选商品记忆、当前页面动作隔离、累计输入 token 统计与硬上限。
- 将候选记忆和累计输入 token 软惩罚接入 veRL 在线 rollout，并保持 token、mask 与 logprob 数组对齐。
- 在单张 RTX 4090 24GB 上完成 Qwen3.5-2B SFT-e1：792 条训练样本、99 steps、约 76 分钟、峰值显存 10.5 GiB、train loss 0.3971、eval loss 0.3566。
- 完成 10-task smoke 和 40-task 四臂配对评测，结果见 `docs/sft-e1-ablation.md`。
- 增加 `grpo_budget_only_v1` 和 `grpo_budget_memory_v1` 两个显式训练 arm；前者是当前主实验，后者保留为消融。

## 关键结果

在同一批 40 个开发任务上，原始 Qwen3.5-2B baseline 为 0/40 严格成功；SFT-e1 baseline 为 18/40（45%），完成率从 25% 提升到 95%，Guard 拒绝从 155 降到 13。

SFT-e1 的 budget-only arm 保持 18/40 严格成功，并将总输入 token 从 2,208,741 降到 1,573,200（-28.77%）。Candidate Memory 则降到 14/40：新增 1 个成功、丢失 5 个成功。逐任务诊断显示它既会把模型引向搜索高排名但约束不完整的商品，也会干扰正确商品上的颜色和套餐选择。

因此下一阶段不继续沿用“memory + budget”作为默认方案。GRPO 主实验 `grpo_budget_only_v1` 关闭候选记忆，启用超过 60,000 累计输入 token 后的封顶软惩罚。评测中的 100k 硬停止与 GRPO 软惩罚机制不同，当前结果只支持这个方向值得训练验证，不能预先宣称训练收益。

## 数据边界

- `data/budget_memory/dev_smoke.jsonl` 的 10 条任务用于开发调试。
- `data/budget_memory/dev_holdout_40.jsonl` 最初用于 base-model v4 留出验证，现又用于 SFT-e1 四臂决策，因此从现在起属于开发数据。
- 40-task 与 GRPO train、SFT train/validation、Final-200 Clean 的 task ID 仍为零重叠。
- Final-200 Clean 只能在配置冻结后执行，且不用于调参。

## 贡献边界

购物环境、教师数据、SFT/GRPO 流程、已有奖励与评测来自上游。上游已经实现工具上下文截断、observation 投影和可选长度惩罚；这些不属于本项目新增贡献。

本项目新增的是候选证据记忆及其训练接入、动作合法性隔离、累计输入成本统计/约束、配对消融与失败诊断。当前最有价值的实验结论是：SFT 显著恢复了 2B 模型的工具策略；简单候选记忆对弱 base model 有帮助，但会干扰 SFT 后策略；预算约束可在这组开发任务上保持成功集合并削减失败轨迹成本。

上游当前提交未发现 LICENSE/COPYING 文件。保留来源信息；对外发布代码前需要核实源码与嵌入数据的再分发条款。
