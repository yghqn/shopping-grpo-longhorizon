# 预算感知购物 Agent：简历与面试讲解

## 一句话介绍

在 Shopping GRPO 上改造长程购物 Agent，将 96GB 长上下文训练配方适配到单张 RTX 4090 24GB，并通过配对消融选择 SFT checkpoint 与推理预算策略：Qwen3.5-2B 从 0/40 提升到 19/40 严格成功，100k token 上限保留全部成功并减少 25.15% 输入成本。

## 简历项目名称

**Budget-Aware Shopping Agent｜长程购物智能体的后训练与推理成本优化**

## 简历描述

- 将 Qwen3.5-2B 的长上下文 SFT 配方适配到单张 RTX 4090，采用 QLoRA、Liger、梯度检查点和 24,576-token 上下文完成 3-epoch 训练，峰值显存 18.2 GiB。
- 在固定 40-task 开发集上按 task ID 配对比较 base、SFT-e1/e2/e3，发现最低 validation loss 的 e2 并非最佳策略，最终选择达到 19/40 严格成功、100% 完成率的 e3。
- 实现累计输入 token 计量和轨迹级预算控制；e3 上的 100k 硬上限保留完全相同的 19 个成功任务，将输入 token 从 2.25M 降到 1.69M（-25.15%）。
- 实现带来源审计的 Candidate Memory，并通过消融发现其在 SFT 后造成 18/40 到 14/40 的退化；基于错误商品和错误规格案例否决该方案，保留为负向实验。
- 将 budget-only 软奖励塑形接入 veRL 在线 GRPO，保留固定分母、数据哈希、任务重叠审计和逐任务轨迹；正式 GRPO 与 Final-200 尚未执行，不把上游结果算作本项目成果。

## 技术栈

Python、PyTorch、Transformers、PEFT/QLoRA、Liger、vLLM、veRL、Qwen3.5-2B、ShopSimulator、结构化工具调用、长上下文训练、配对评测、Git、AutoDL、RTX 4090。

## 1 分钟面试讲解

这个项目研究长程购物 Agent 的成功率与推理成本。原项目的长上下文训练在 96GB GPU 上验证，我先用 QLoRA、Liger 和梯度检查点把 24,576-token、3-epoch SFT 跑到单张 24GB RTX 4090，峰值显存 18.2 GiB。

我没有只看 validation loss，而是在固定 40 个开发任务上配对比较三个 checkpoint。E2 的 loss 最低，但环境成功只有 17/40；E3 达到 19/40、完成率 100%，因此选 e3 作为 GRPO 起点。然后我加入累计输入 token 上限，保持相同 19 个成功任务，同时把总输入量降低 25.15%，三条被截断的都是原本失败轨迹。

我也实现了可追溯 Candidate Memory，但它在 SFT 后把成功数从 18 降到 14。逐任务分析发现它会偏向搜索高排名但约束不完整的商品，也会干扰颜色和套餐选择，所以没有为了故事好看保留该功能，而是把它记录为负向消融。下一步是用 e3 启动 budget-only GRPO，再在配置冻结后运行 Final-200 Clean。

## 3 分钟讲解顺序

1. **问题**：购物 Agent 需要多轮搜索、核验规格和购买，失败既来自策略错误，也来自不断增长的输入成本。
2. **资源约束**：上游配方在 96GB GPU 上验证；本项目需要在可租用的 24GB 4090 上完成真实训练。
3. **训练适配**：QLoRA + Liger + gradient checkpointing，完整 24,576 上下文，3 epochs，峰值 18.2 GiB。
4. **Checkpoint 选择**：e1/e2/e3 分别为 18/17/19 个严格成功，证明 validation loss 不能替代环境评测。
5. **预算消融**：100k 上限保持 19/40，输入 token 减少 566,667，平均步骤从 9.225 降到 8.35。
6. **负向结果**：Candidate Memory 对弱 base model 有帮助，但干扰 SFT 策略；根据五个退化案例将其从主方案移除。
7. **下一步**：在 80/96GB GPU 上运行 budget-only GRPO，冻结后再做 Final-200。

## 常见追问

### 为什么 QLoRA 还能保留长上下文？

量化主要降低基础权重和优化器状态占用，Liger 与梯度检查点进一步控制激活显存。最长样本单步压力测试峰值 18.0 GiB，完整训练峰值 18.2 GiB，因此不是简单把上下文砍短来换显存。

### 为什么不用 validation loss 最低的 epoch 2？

Validation loss 衡量教师动作 token 的平均预测误差，不直接衡量商品、规格、终止和工具交互是否正确。E2 的 loss 为 0.3396，低于 e3 的 0.3415，但环境严格成功是 17/40 对 19/40，且成本更高，所以选择 e3。

### 预算上限是不是靠提前终止来虚假降低 token？

它确实通过提前停止三条超过 100k 的轨迹节省成本，所以报告明确披露 budget-exhausted 数量。关键是配对结果显示 19 个成功 task ID 完全不变，被截断的三条原本都是失败。GRPO 中使用的是软惩罚，不把硬上限结果直接冒充训练收益。

### Candidate Memory 为什么失败？

当前记忆按已核验程度、最近出现和搜索排名排序，没有直接优化用户约束匹配；1024-token 视图还会重复当前搜索证据。失败案例既包括选错商品，也包括正确商品上选错颜色或套餐，说明候选排序和提示分布偏移都可能干扰 SFT 策略。

### 19/40 能否和上游 60.5% 比较？

不能直接比较。19/40 来自已经用于开发选择的 40-task 集；上游 60.5% 来自不同的 Final-200 协议。只有在 GRPO 配置冻结后运行同一 Final-200，才能做公平横向比较。

## 不能写进简历的说法

- “达到或超过原项目 60.5%”：当前尚未运行同一 Final-200。
- “Candidate Memory 提升了 SFT”：实际消融显示它降低成功率。
- “训练预算惩罚已经提升模型”：目前只验证了评测期硬上限，GRPO 软惩罚尚未训练。
- “完成了 GRPO 优化”：训练接入已完成，但还没有本项目的 GRPO checkpoint。
- “节省 25.15% 且完全没有代价”：应同时披露三条失败轨迹被硬上限终止。

## 可展示证据

- 当前状态：[`PROJECT.md`](../PROJECT.md)
- SFT checkpoint 与预算消融：[`docs/sft-checkpoint-selection.md`](sft-checkpoint-selection.md)
- Candidate Memory 负向分析：[`docs/sft-e1-ablation.md`](sft-e1-ablation.md)
- 命名训练实验：`configs/experiments.json` 中的 `grpo_budget_only_v1`
- 核心实现：`src/shopping_grpo/training/grpo/`、`src/shopping_grpo/memory/`
- 配对报告：`scripts/compare_budget_memory_runs.py`
