# AutoDL 首次基线

本页只覆盖第一阶段：在 10 个开发任务上运行原始 `Qwen/Qwen3.5-2B`，确认真实环境、模型服务、工具调用和指标记录全部连通。它不是正式结果，也不使用 Final-200。

## 实例选择

建议首次使用以下配置：

| 项目 | 选择 |
| --- | --- |
| 计费 | 按量计费 |
| GPU | RTX 4090 24GB；若 3090 24GB 明显更便宜也可 |
| CPU / 内存 | 至少 8 vCPU / 64GB RAM |
| 镜像 | PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 |
| 数据盘 | 先用免费 50GB |
| 项目目录 | `/root/autodl-tmp/budget-aware-shopping-agent` |

2B 模型的首次推理不需要 80GB 或 96GB GPU。后续 SFT、GRPO 的配置要根据本次实测重新估算，不能沿用本页选型。

## 上传项目

本地交付文件为 Git bundle，它同时保存固定的上游提交和本地改造提交。把
`budget-aware-shopping-agent.bundle` 上传到 `/root/autodl-tmp/`，然后在 AutoDL 终端执行：

```bash
cd /root/autodl-tmp
git clone -b codex/budget-aware-memory \
  budget-aware-shopping-agent.bundle budget-aware-shopping-agent
cd budget-aware-shopping-agent
git status --short
git log -1 --oneline
```

`git status --short` 应无输出。记录最后一条提交，后续结果才能和代码版本一一对应。

## 安装

所有缓存都放在数据盘，避免系统盘重置后重新下载：

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
mkdir -p /root/autodl-tmp/cache/huggingface /root/autodl-tmp/cache/uv
export HF_HOME=/root/autodl-tmp/cache/huggingface
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
export PATH="$HOME/.local/bin:$PATH"

source /etc/network_turbo
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10 3.12
bash scripts/setup.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

安装会创建主项目 Python 3.12 环境和隔离的 ShopSimulator Python 3.10 环境。基础镜像里的 PyTorch 只是启动环境；项目最终使用 `pyproject.toml` 固定的依赖版本。

先跑离线检查：

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_context_window.py \
  tests/test_observation_projection.py \
  tests/test_rollout.py \
  tests/test_environment_manifest.py \
  tests/test_evaluation_dataset.py \
  tests/test_action_validation.py \
  tests/test_structured_observation.py \
  tests/test_candidate_memory.py
```

## 三个终端

终端 1 启动购物环境：

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
bash scripts/start_environment.sh
```

终端 2 启动基础模型，同时观察首次模型下载是否完成：

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
export HF_HOME=/root/autodl-tmp/cache/huggingface
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

终端 3 确认两个服务就绪后运行 10 条无记忆基线：

```bash
cd /root/autodl-tmp/budget-aware-shopping-agent
bash scripts/dev_smoke.sh baseline
```

运行期间可在第四个终端使用 `nvidia-smi` 记录峰值显存。结果写入：

```text
outputs/budget-memory/dev-smoke/baseline/trajectories.jsonl
outputs/budget-memory/dev-smoke/baseline/summary.json
```

这轮通过的标准是：10 条任务都有轨迹；没有模型服务或环境服务错误；合法工具调用可以完成；summary 中有累计输入 token 指标。成功率可以很低，因为这里测的是原始 2B 基础模型。

## 结束后

保存 `summary.json`、`trajectories.jsonl`、安装日志、模型服务日志和峰值显存。停止模型与环境进程，然后在 AutoDL 控制台关机，避免继续按开机时间计费。候选记忆组将在基线通过后使用相同实例和参数运行：

```bash
bash scripts/dev_smoke.sh memory
```
