# 预算感知购物 Agent：简历与面试讲解

## 一句话介绍

基于 Shopping GRPO 完成 Qwen3.5-2B 的长程购物 Agent 后训练与可审计评测：先在单张 RTX 4090 上完成 24,576-token QLoRA SFT，再在单张 RTX PRO 6000 上完成标准与预算感知 GRPO，并构建 Rubric Curator + Trajectory Judge 流水线；Final-200 Clean 中，标准 GRPO 将严格成功率从 69.5% 提升到 71.0%，而预算软惩罚减少部分推理成本但未保留成功率增益。

## 简历项目名称

**Budget-Aware Shopping Agent｜长程购物智能体后训练与推理成本优化**

## 推荐简历描述

**项目简介：** 面向长程商品搜索、证据核验、规格选择与购买决策，构建从 SFT、在线 GRPO 到固定盲测集与 LLM-as-Judge 的完整 Agent 后训练流水线，并研究任务成功率与累计输入 token 成本之间的权衡。

- 将 Qwen3.5-2B 的 24,576-token SFT 配方适配到单张 RTX 4090，采用 QLoRA、Liger、梯度检查点和 bf16 完成 3-epoch 训练，峰值显存 18.2 GiB；通过环境评测而非最低 validation loss 选择 checkpoint。
- 基于 veRL 实现标准与 budget-only GRPO，各完成 500 个有效更新；针对 Qwen3.5 的 248,320 词表修复 PPO 输出头和 entropy 显存问题，并补齐动态采样、checkpoint resume、训练诊断与 LoRA 导出合并链路。
- 通过 SHA-256 权重审计发现 veRL 导出的顶层权重仍为 SFT base、GRPO 更新保存在独立 LoRA Adapter；修复 Adapter 合并链路并重跑 Final-200，避免将 SFT 结果误报为 GRPO。
- 构建“代码硬检查 + DeepSeek Flash Rubric Curator + DeepSeek V4 Pro Trajectory Judge”评测：为 200 个任务冻结 1,347 条 hard/soft Rubric，以 Actor 可见轨迹完成逐约束判定、五维评分和错误归因，并采用固定分母与逐任务配对统计防止评测泄漏。
- 在严格隔离的 Final-200 Clean 上评测：SFT、标准 GRPO、budget-only GRPO 的严格成功率分别为 69.5%、71.0%、69.5%；标准 GRPO 平均效用提升 0.0255，Judge 五维均值整体提高，记录的 hard 违反由 64 降至 48，但配对差异不显著（McNemar `p=0.6636`）。
- 设计累计输入 token 软成本与 Candidate Memory 消融：budget-only 相对标准 GRPO 减少 3.78% 输入 token 和 31 次 Guard 拒绝，却损失 3 个净成功任务；据此否决“预算方法整体胜出”的假设，并将 `wrong_option` 定位为三组模型共同的主要错误。

## 技术栈

Python、PyTorch、Transformers、PEFT/QLoRA、Liger、vLLM、veRL、Ray、Qwen3.5-2B、ShopSimulator、结构化工具调用、长上下文训练、GRPO、LLM-as-Judge、Rubric Curator、轨迹评测、配对统计、Git、AutoDL、RTX 4090、RTX PRO 6000。

## 最终结果表

| 模型 | 严格成功 | 完成率 | 平均终局效用 | 平均加权分 | 平均步骤 | 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| SFT-e3 | 139/200（69.5%） | 98.5% | 0.6024 | 0.7750 | 9.825 | 12,081,725 |
| 标准 GRPO step 450 | 142/200（71.0%） | 97.5% | 0.6279 | 0.7913 | 10.095 | 13,446,494 |
| Budget-only GRPO step 450 | 139/200（69.5%） | 98.0% | 0.5935 | 0.7681 | 9.945 | 12,938,476 |

配对比较：

- 标准 GRPO 相对 SFT：12 题由失败变成功，9 题由成功变失败，净增 3 题；输入 token 增加 11.30%，exact McNemar `p=0.6636`。
- Budget-only 相对标准 GRPO：8 题改善、11 题回退，净减 3 题；输入 token 减少 3.78%，Guard 拒绝减少 31 次，exact McNemar `p=0.6476`。
- 统计检验不支持显著成功率差异，因此应表述为“小幅趋势”和“成本权衡”，不能宣称稳定提升或预算方法胜出。

LLM-as-Judge 补充结果：

| Actor | Judge 覆盖率 | 搜索 | 候选利用 | 证据核验 | 决策质量 | 终止效率 | hard 满足 / 未知 / 违反 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-e3 | 99.0% | 1.793 | 1.783 | 1.662 | 1.545 | 1.697 | 1099 / 47 / 64 |
| 标准 GRPO step 450 | 98.5% | 1.822 | 1.797 | 1.695 | 1.548 | 1.726 | 1106 / 65 / 48 |
| Budget-only GRPO step 450 | 98.5% | 1.802 | 1.746 | 1.635 | 1.508 | 1.660 | 1073 / 66 / 76 |

标准 GRPO 的记录 hard 违反减少，但 unknown 增加，因此不能只用 violation 数量宣称全面改善。三组最常见 primary error 均为 `wrong_option`（29 / 26 / 24）。

## 1 分钟面试讲解

这个项目研究长程购物 Agent 的任务成功率和推理成本。我先把 Qwen3.5-2B 的 24K 长上下文 SFT 适配到单张 24GB RTX 4090，使用 QLoRA、Liger 和梯度检查点完成 3 个 epoch，峰值显存 18.2 GiB。Checkpoint 选择没有只看 validation loss：e2 的 loss 最低，但固定开发集上 e3 的真实环境成功率更高，所以最终选 e3。

然后我在 96GB 单卡上分别训练标准 GRPO 和带累计输入 token 软成本的 budget-only GRPO，各完成 500 个有效更新。训练过程中解决了 Qwen3.5 大词表导致的 PPO 输出头显存问题、entropy 峰值、动态采样无信号批次、断点续训和 LoRA 导出合并问题。

最终在完全隔离的 200 题上，SFT 是 69.5%，标准 GRPO 是 71.0%，budget-only 是 69.5%。除 Reward 外，我还让 Flash 从代码候选中冻结 1,347 条逐任务 Rubric，再由 Pro 只依据 Actor 可见轨迹评价搜索、候选利用、证据核验、决策和终止效率。标准 GRPO 的五维均值整体更高，但输入 token 增加 11.3%；预算方案相对标准 GRPO 节省 3.78% token，却丢失了 3 个净成功任务。我的结论不是“预算方法成功”，而是当前软惩罚没有形成更优 Pareto 点，主要失败根因集中在错误规格选择。

## 3 分钟讲解顺序

1. **问题定义**：长程购物需要反复搜索、核验商品与规格并完成购买，轨迹越长，输入 token 和错误累积越严重。
2. **数据边界**：SFT、GRPO train/validation、40-task 开发集和 Final-200 Clean 按 task ID 零重叠；Final-200 只在方案冻结后使用。
3. **SFT 适配**：在 RTX 4090 上以 QLoRA、Liger、梯度检查点完成 24,576-token、3-epoch 训练；峰值显存 18.2 GiB。
4. **Checkpoint 选择**：e1/e2/e3 在 40-task 开发集分别取得 18/17/19 个严格成功，说明 token-level validation loss 不能代替环境成功率。
5. **GRPO 工程**：在 RTX PRO 6000 上完成两个 500-step arm；用 fused PPO output head、entropy chunking/checkpointing、remove padding 和单轨迹 microbatch 稳定训练。
6. **训练可靠性**：为动态采样记录接受/丢弃原因，为每个 optimizer step 保存诊断；修复恢复训练被 epoch 上限提前截断的问题。
7. **导出审计**：发现 veRL 中间导出的顶层权重仍是 SFT base，GRPO 更新位于独立 LoRA Adapter；修复导出脚本并重新合并、哈希和评测。
8. **评测可靠性**：用代码候选约束 Curator，冻结 Rubric 后让 Judge 只看 Actor-visible trajectory；基础设施无效轨迹记为 `not_judged` 但保留在 200 题分母。
9. **最终结论**：标准 GRPO 有小幅正向趋势但更贵；budget-only 收回部分成本却失去净成功增益；Candidate Memory 是负向消融；主要错误是 `wrong_option`。

## 常见追问

### 为什么使用 QLoRA，24K 上下文是否被缩短？

没有缩短。QLoRA 降低基础权重和优化器状态占用，Liger 与梯度检查点控制激活显存。最长样本单步压力测试峰值 18.0 GiB，完整训练峰值 18.2 GiB，正式训练上限仍为 24,576 token。

### 为什么不用 validation loss 最低的 epoch 2？

Validation loss 衡量教师动作 token 的平均预测误差，不直接衡量商品、规格、终止和工具交互是否正确。E2 的 loss 为 0.3396，低于 e3 的 0.3415，但环境严格成功为 17/40 对 19/40，因此选择 e3。

### GRPO 是否真的提升了模型？

标准 GRPO 从 139/200 提升到 142/200，平均终局效用增加 0.0255，重复循环从 25 降到 15，说明存在正向信号。但逐任务配对为 12 个改善、9 个回退，`p=0.6636`，没有统计显著性；准确说法是“小幅正向趋势”，不是已经证明稳定提升。

### Budget-only 的作用是什么？

它在每条训练轨迹累计输入超过 60,000 token 后施加线性软惩罚，惩罚最高封顶 0.1。最终它相对标准 GRPO 减少 3.78% 输入 token 和 31 次 Guard 拒绝，但成功率从 71.0% 回到 69.5%。这说明当前惩罚强度与 credit assignment 没有形成更优的成功率—成本 Pareto 点。

### 为什么 GRPO 训练使用 96GB，但峰值看起来不高？

早期实现会在长序列和 248,320 词表上物化巨大的 logits/entropy 张量并多次 OOM。修复 fused output head、entropy chunking/checkpointing 和 microbatch 后显存显著下降。96GB 机器承担了问题定位和稳定性验证；最终配置有机会缩到更便宜的 GPU，但需要重新做完整压力测试，不能只根据某一次监控值断言。

### 动态采样是什么意思？

每个 prompt 生成 4 条 rollout。若一组奖励完全相同，GRPO 组内优势为零，无法提供有效梯度；动态采样会丢弃这类组并继续采样，最多补采三批。达到上限仍没有足够有效组时跳过本次更新，同时记录原因，避免无限采样。

### Candidate Memory 为什么失败？

当前记忆按已核验程度、最近出现和搜索排名组织，但没有直接优化全部用户约束；额外记忆还会改变 SFT 已学到的提示分布。配对消融显示它造成错误商品和错误颜色、套餐选择，因此没有进入最终主方案。

### 导出模型时发现了什么问题？

veRL 的模型合并工具导出了 SFT 顶层权重和单独的 GRPO LoRA Adapter。如果直接服务顶层权重，实际评测的仍是 SFT。项目通过文件哈希发现三个模型权重完全相同，随后修复 `scripts/export_grpo.sh`，将 GRPO Adapter 真正 merge 到基础模型，并重新运行两组 Final-200。

### Rubric Curator 和 Trajectory Judge 为什么要分开？

Curator 只处理“用户到底要求了什么”：它只能从代码生成的候选中选择、去重并标注 hard/soft，不能看任何模型轨迹。Judge 只处理“Actor 做得怎么样”：它读取冻结 Rubric 和 Actor 实际看到的轨迹，逐项判断满足状态并完成五维评分。拆分后，一份 Rubric 可以被三个 Actor 共享，也避免同一个模型一边定义标准、一边根据结果修改标准。

### 如何防止 LLM Judge 看到答案后倒推分数？

Judge 输入中删除 raw Observation、Gold 商品字段、Reward、strict-success 标签、基础设施有效性和其他模型结果，只保留 Actor 当时可见的投影 Observation、中性终局状态以及 legality、repetition、context 等白名单指标。代码还会验证 rubric ID、event ID、分值范围与 hard violation/decision score 的一致性；无效轨迹由代码生成 `not_judged`，不会要求 Judge 猜测。

### 为什么正式运行是 592 次 Judge 请求，不是 600 次？

三个 Actor 理论上共有 600 条轨迹，其中 8 条被代码判定为基础设施无效，直接生成 `not_judged` 并继续计入固定 200 题分母，因此只向 Pro 发送 592 次请求。Curator 与轨迹无关，每个任务只生成一次，所以是 200 次请求。

### LLM-as-Judge 能替代 Environment Reward 吗？

不能。Reward 用结构化 Gold 与终局事实判断严格成功；Rubric 面板解释具体要求是否满足；Trajectory Judge 评价过程质量；确定性面板负责 Guard、重复、上下文和基础设施。四个面板并列保留，不合成一个总分，也不允许 Judge 覆盖 Reward。

### 能否声称超过原项目？

不能仅凭百分比直接声称。不同提交、模型、数据清理和评测协议可能不同。本项目能可靠声明的是：在自己的固定 Final-200 Clean、相同运行时和相同 SFT 起点上，完成了 SFT、标准 GRPO、budget-only GRPO 的公平配对比较。

## 不能写进简历的说法

- “GRPO 显著提升成功率”：配对检验不显著，只能报告小幅正向趋势。
- “Budget-aware GRPO 同时提高成功率并降低成本”：最终实验没有支持这一结论。
- “Candidate Memory 提升了 SFT”：实际消融显示它降低成功率。
- “GRPO 后达到 71.5%”：该数字来自导出错误时实际服务 SFT 权重的无效评测；修复后的标准 GRPO 是 71.0%。
- “训练节省 96GB 到 27GB”：训练优化后的观察值不等于经完整压力测试确认的最低显存需求。
- “达到或超过原项目”：除非确认双方使用完全相同的模型、任务文件、环境和奖励协议。

## 可展示证据

- 项目状态与贡献边界：[`PROJECT.md`](../PROJECT.md)
- GRPO 配置、训练与导出：[`docs/grpo.md`](grpo.md)
- SFT checkpoint 与开发集预算消融：[`docs/sft-checkpoint-selection.md`](sft-checkpoint-selection.md)
- Candidate Memory 负向分析：[`docs/sft-e1-ablation.md`](sft-e1-ablation.md)
- 正式评测协议：[`docs/evaluation.md`](evaluation.md)
- Final-200 Clean 数据边界：[`docs/evaluation-dataset.md`](evaluation-dataset.md)
- 最终本地产物：`outputs/evaluation/`、`outputs/models/*/lora_adapter/`、`outputs/cloud-runs/2026-10-03-final-model-manifest.sha256`
- LLM-as-Judge 归档：`outputs/cloud-runs/2026-10-04-final200-llm-judge-results.tar.gz`
- 核心实现：`src/shopping_grpo/training/grpo/`、`src/shopping_grpo/memory/`、`scripts/export_grpo.sh`
