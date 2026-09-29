# 首次云 GPU 验证

算力方式：AutoDL 按量租用。首次 10 题基线建议 RTX 4090 24GB、至少 8 vCPU / 64GB RAM、PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 镜像与免费 50GB 数据盘。完整操作见 [AutoDL 首次基线](autodl-baseline.md)。当前没有启动租赁或付费模型调用。

上游主训练环境使用 Linux / Python 3.12，ShopSimulator 使用隔离 Python 3.10。原始 GRPO 配置按 96GB GPU 验证；缩小上下文、并发或采样数后必须重新验证，不能直接视为等价复现。

## 本地离线检查

设置 `PYTHONPATH=src` 与 `PYTHONUTF8=1` 后执行：

```bash
python -m pytest -q tests/test_context_window.py tests/test_observation_projection.py tests/test_rollout.py tests/test_environment_manifest.py tests/test_evaluation_dataset.py tests/test_action_validation.py tests/test_structured_observation.py
```

Windows 克隆应使用 `git -c core.autocrlf=false clone ...`，避免冻结数据的原始字节哈希被 CRLF 转换改变。当前本地副本已恢复为上游原始字节。

## 云端顺序

先传输本项目及本地改造，再安装固定依赖；安装与模型下载可能耗时，需纳入租赁预算。依次使用上游入口：

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

在另一个终端启动模型：

```bash
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

模型和环境就绪后，10 题工程验证可直接运行：

```bash
bash scripts/dev_smoke.sh baseline
```

其展开命令如下。该命令会实际调用云端模型；本地准备阶段尚未执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_shop_benchmark.py   --benchmark data/budget_memory/dev_smoke.jsonl   --output outputs/budget-memory/dev-smoke/trajectories.jsonl   --summary outputs/budget-memory/dev-smoke/summary.json   --base-url http://127.0.0.1:5700   --model shopping-agent   --llm-base-url http://127.0.0.1:8000/v1   --api-key EMPTY   --max-steps 35   --temperature 0.0 --top-p 1.0   --context-window 24576 --context-safety-margin 512 --max-tokens 512
```

这只是上游默认推理 smoke；对比已有上下文裁剪时增加 `--context-compaction` 并使用不同输出目录。

候选记忆组使用独立输出目录，可直接运行 `bash scripts/dev_smoke.sh memory`。其显式增加：

```bash
  --candidate-memory \
  --candidate-memory-token-budget 1024
```

该开关使用同一个 vLLM tokenizer 计算记忆 token，轨迹保存每轮选择统计和最终可追溯记忆。首次 10 题运行只验证工程链路，不据此宣称效果提升。

首次 smoke 保持 `--cumulative-input-token-budget 0`，先记录每题真实累计输入 token。根据开发集分布预先固定后续硬上限，再用非零值重跑所有对照组；预算耗尽会标记为 `cumulative_input_budget_exceeded`。

## 继续条件

- 确认环境契约、模型 checkpoint、tokenizer 与 tools 一致。
- 确认模型能够合法交互，失败轨迹和服务错误可区分；Base 成功率低本身不等于服务失败。
- 记录峰值显存、每题耗时和实际 GPU 费用，再估计后续实验预算。
- 已有 SFT checkpoint 的来源与加载需核实；没有 checkpoint 时才安排 SFT，不能把基础模型用于 GRPO 初始化却称为上游复现。
- 正式训练、模型合并和 Final-200 评测尚未执行；开始训练前释放推理服务占用的 GPU。

## 本地尚未覆盖的验证

Windows 本地已运行的主项目集合为 177 passed、1 skipped、15 subtests passed。完整收集还需要 ShopSimulator 独立环境、Hydra、veRL、NumPy 等固定依赖；上游两项 curriculum 测试还把 Linux 路径分隔符写入断言。首次云端安装后应分别运行主项目与 ShopSimulator 测试，不能把本地缺依赖误报为功能回归。
