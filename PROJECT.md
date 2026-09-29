# 预算感知购物 Agent

本目录是基于 Shopping GRPO 的独立改造项目。当前阶段是复现准备，尚未完成模型推理、SFT、GRPO 或效果提升验证。

研究主题：在长程购物任务中，用可追溯的候选商品记忆保留约束和核验证据，比较成功率与累计 token、工具调用成本之间的关系。计算预算与用户的商品价格预算分开统计。

## 已准备

- 固定上游提交，来源和核心资产哈希见 `UPSTREAM.json`。
- 本地分支 `codex/budget-aware-memory`，原仓库保留为 `upstream`。
- 10 个工程验证任务，来自 GRPO validation；与训练数据和 Final-200 Clean 无 task_id 重叠。
- 对照设计见 `docs/budget-memory-plan.md`，云端首次运行步骤见 `docs/cloud-first-run.md`。
- `configs/budget_memory_protocol.json` 记录实验设计；候选记忆已通过显式开关接入评测运行时。

## 贡献边界

购物环境、教师数据、SFT/GRPO 流程、已有奖励与评测来自上游。上游已经实现完整工具组上下文截断、observation 投影和可选长度惩罚；这些不属于本项目新增贡献。原 README 及 experiments 中的结果均为上游结果。

本项目已实现第一版候选证据记忆：只读取公开 observation v2，保留字段冲突与来源，并在独立 token 预算下优先选择已核验候选。评测侧还支持累计输入 token 统计与可选硬上限；训练侧接入仍未实现。

## 当前基线检查

候选记忆及相关评测路径已有离线测试覆盖。本地当前可运行的主项目集合为 177 passed、1 skipped、15 subtests passed；另有训练依赖和 Windows 路径相关测试需在 Linux 云端固定环境验证。已将旧搜索页面测试夹具对齐到 observation v2，并修复结构化搜索页面容量校验遗漏（包括无需压缩的分支）。这不是模型评测；真实环境、模型运行和训练仍待云端验证。

上游当前提交未发现 LICENSE/COPYING 文件。保留来源信息；对外发布代码前需要核实源码与嵌入数据的再分发条款。
