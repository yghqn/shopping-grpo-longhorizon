# 预算感知购物 Agent

本目录是基于 Shopping GRPO 的独立改造项目。当前已完成 Qwen3.5-2B
真实环境基线、四轮候选记忆开发集消融和 40-task 未见开发留出评测；尚未执行
本地改造的 SFT、GRPO 或 Final-200 Clean。

研究主题：在长程购物任务中，用可追溯的候选商品记忆保留约束和核验证据，比较成功率与累计 token、工具调用成本之间的关系。计算预算与用户的商品价格预算分开统计。

## 已准备

- 固定上游提交，来源和核心资产哈希见 `UPSTREAM.json`。
- 本地分支 `codex/budget-aware-memory`，原仓库保留为 `upstream`。
- 10 个工程验证任务，来自 GRPO validation；与训练数据和 Final-200 Clean 无 task_id 重叠。
- 对照设计见 `docs/budget-memory-plan.md`，云端首次运行步骤见 `docs/cloud-first-run.md`。
- `configs/budget_memory_protocol.json` 记录实验设计；候选记忆已通过显式开关接入评测运行时。
- 首轮云端分析、逐任务诊断和最终 v4 选择见
  `docs/dev-smoke-memory-analysis.md`；原始轨迹已按轮次归档到
  `outputs/cloud-runs/`（该目录不进入 Git）。
- 未参与调参的剩余 40 条 GRPO validation 已冻结并完成三臂配对评测，数据、协议、
  结果和统计限制见 `docs/dev-holdout-40.md`。

## 贡献边界

购物环境、教师数据、SFT/GRPO 流程、已有奖励与评测来自上游。上游已经实现完整工具组上下文截断、observation 投影和可选长度惩罚；这些不属于本项目新增贡献。原 README 及 experiments 中的结果均为上游结果。

本项目已实现候选证据记忆：只读取公开 observation v2，保留字段冲突与来源，
在独立 token 预算下优先选择已核验候选。真实 ASIN 保留在审计轨迹中，模型只看到
稳定的历史标签和独立的当前页面动作区，避免把历史商品误作当前控件。评测侧记录
累计输入 token，并可设置轨迹级硬上限；训练侧接入仍未实现。

## 10-task 开发集结果

RTX 4090/4090D、Qwen3.5-2B、temperature 0 下的最终 v4 保留了 v2 首次出现的
1/10 Gold 严格成功和 1/10 部分匹配购买，同时用每轨迹 100,000 累计输入 token
上限把总输入量从 1,276,508 降到 640,131（-49.9%），并把上下文溢出从 2 条降到
0。相较无记忆基线，严格成功从 0/10 到 1/10，非法动作终止从 9/10 到 2/10。

这是固定 10 条开发任务上的工程诊断，不能外推为正式成功率结论。

## 40-task 未见开发留出结果

在相同 Qwen3.5-2B、temperature 0 和环境协议下，baseline、budget-only、memory-v4
分别完成 40 条配对轨迹。Memory-v4 严格成功 2/40，对照均为 0/40；相对 baseline，
总输入 token 从 2,334,845 降到 1,867,081（-20.0%），非法动作终止从 28 降到 21，
Guard 拒绝从 155 降到 105。两个新增成功任务为 17825 和 19090，没有 baseline-only
成功。仅预算上限将 token 降低 10.0%，但严格成功仍为 0/40。

该结果支持候选记忆继续进入下一阶段，但仍是单次开发留出实验。成功率差的配对
bootstrap 95% 区间包含 0，不能表述为统计显著，也不能替代 Final-200 Clean。
候选记忆及运行路径的 42 个针对性测试已通过；云端首次环境检查另有 74 tests、
2 subtests 通过。加入 holdout 完整性和配对报告检查后，本地相关测试集合为 83 tests
通过。

上游当前提交未发现 LICENSE/COPYING 文件。保留来源信息；对外发布代码前需要核实源码与嵌入数据的再分发条款。
