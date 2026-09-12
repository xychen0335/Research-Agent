# 数据与运行环境

本项目已经固定数据版本、模型 revision、训练框架 commit 和单卡环境。实施配置以 [configs/data.json](configs/data.json)、[configs/upstreams.json](configs/upstreams.json) 和 [configs/environment.json](configs/environment.json) 为准。修改任一版本时，需要重新完成环境检查、数据清单生成、单步 SFT 和单步 GRPO。

## 数据集

训练数据使用 BIRD 团队发布的 [birdsql/bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered)。固定 revision 为 `4068469807b255fcfc0816bdd520946fe460d256`，共 6,601 条记录，许可证为 CC-BY-SA-4.0。记录文件、字段说明和数据库压缩包地址都写在 `configs/data.json`。

最终测试使用 BIRD 团队发布的 [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev)。固定 revision 为 `f65faf4ae3b638c1fa6df1d3370c8d92c8366301`，共 500 条 SQLite 任务和 11 个数据库，许可证为 CC-BY-SA-4.0。mini-dev 在训练开始前冻结，不用于提示词修改、奖励选择或 checkpoint 选择。

训练集内部按 `db_id` 划分训练池和验证池。随机种子固定为 `20260911`。实际数据包含 69 个数据库，其中 58 个进入训练池，11 个进入验证池。验证数据库名单固定在 `configs/data.json`，验证池包含 1,005 条记录。训练池和验证池不会共享数据库。

| 输出 | 数量 | 用途 |
|---|---:|---|
| `sft_train.json` | 1,500 | 生成教师轨迹的训练来源池 |
| `sft_val.json` | 150 | SFT 验证轨迹来源池 |
| `rl_train.json` | 2,000 | GRPO 训练 |
| `rl_val.json` | 200 | checkpoint 选择和训练诊断 |
| `frozen_test.json` | 500 | 最终 Base、SFT、SFT + GRPO 对照 |

SFT 文件的实际条数由教师轨迹的执行正确率决定。转换器只保留通过执行评测的轨迹，因此不会把来源池数量当成最终 SFT 数量。

运行以下命令会下载固定 revision 的记录、数据库压缩包和模型，规范化 SQLite 目录，生成确定性划分，并把下载文件的 SHA256 写入 `data/SHA256SUMS`：

```bash
bash scripts/download_assets.sh
```

下载结果位于 `data/raw/`，数据库位于 `data/databases/`，划分及其清单位于 `data/splits/`。这些大文件目录不提交到 Git。首次下载后应提交 `data/SHA256SUMS`，让后续运行可以核对同一数据库归档快照。

## 模型与上游代码

底座为 `Qwen/Qwen3.5-4B`，固定 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。模型下载到 `models/Qwen3.5-4B-851bf6e8`。Qwen 的[模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)说明该模型支持 vLLM、`qwen3` reasoning parser 和 `qwen3_coder` tool parser。

BIRD-RL 固定在 `b26c4285ef69dfb6c096b076a7499018c6b370ca`。setup 会应用仓库内的 `patches/bird-rl-readonly.patch`，把教师轨迹执行与评测连接改为 SQLite 只读 URI。verl 固定在 `10db40d0da4d59150bb389960b77585f81a89b8d`。setup 脚本按 commit 检出代码，后续命令不会跟随上游分支移动。

## 训练环境

训练节点固定为以下配置：

| 项目 | 固定值 |
|---|---|
| 平台 | Linux x86_64 |
| 系统 | Ubuntu 24.04 |
| GPU | NVIDIA A100 80GB，显存至少 79,000 MiB |
| NVIDIA driver | 580 或更新版本 |
| CUDA 基础镜像 | `nvidia/cuda:13.0.2-devel-ubuntu24.04` |
| Python | 3.12.x |
| uv | 0.11.16 |
| verl Dockerfile | `upstream/verl/docker/Dockerfile.uv.cu130` |

Python 依赖由固定 verl commit 的 `uv.lock` 决定。该 lock 文件的 SHA256 为 `d30f7e35c9f077c3c27aae1012711a753939db81a9a2cf4bcdbdf03fab59d0a9`。关键包版本如下：

| 包 | 版本 |
|---|---|
| torch | `2.11.0+cu130` |
| vLLM | `0.24.0` |
| transformers | `5.9.0` |
| flash-attn | `2.8.3` |
| Ray | `2.55.1` |
| datasets | `5.0.0` |
| PyArrow | `24.0.0` |
| PEFT | `0.19.1` |
| TensorDict | `0.10.0` |

CUDA 13.x 所需驱动下限来自 [NVIDIA CUDA compatibility 文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。AutoDL 镜像必须满足上述系统和驱动条件。`scripts/setup_autodl.sh` 会在安装依赖前检查系统、GPU、显存和驱动，安装后再导入所有关键包并检查精确版本。

```bash
bash scripts/setup_autodl.sh
```

检查结果写入 `outputs/setup/nvidia-smi.txt` 和 `outputs/setup/versions.json`。也可以基于固定 verl Dockerfile 构建镜像：

```bash
bash scripts/build_training_image.sh
```

当前仓库尚未在 A100 上执行训练，因此 8K prompt、4K response、每题 4 条 rollout 和 LoRA rank 32 是首轮试跑配置，不是已经验证的吞吐或显存结论。正式训练前必须先完成单步 SFT、单步 GRPO 和固定小批量推理，并保存日志。

## 数据预处理检查

数据转换使用训练虚拟环境的 Python。转换完成后，脚本用固定模型 tokenizer 统计每个文件的 token 长度，写入 `data/processed/token_lengths.json`。任何 RL prompt 超过 8,192 token 时，预处理直接失败，训练脚本不会静默过滤样本。

```bash
bash scripts/prepare_bird_data.sh
```

该命令同时生成 `data/processed/bird_mini_dev_tasks.jsonl`，供统一的只读评测 runner 和看板使用。本地 runner 的评分用于工程诊断；`scripts/evaluate_bird_run.sh` 会把结果转换为 BIRD-RL trajectory 格式，再用只读补丁后的固定 evaluator 计算 EX 并保存逐题结果。

## 看板环境

看板使用独立的 Python 3.12 虚拟环境，避免修改 GPU 训练 lock。顶层版本固定为 Streamlit 1.63.0 和 pandas 3.0.5；创建看板环境的 uv 不要求与训练环境同版。

```bash
bash scripts/setup_ui.sh
bash scripts/run_dashboard.sh
```
