# Research Agent

在冻结语料上做科研检索：模型自己决定搜索、阅读和提交。训练、评测和推理走同一个 `research_agent.harness.loop`。verl 只做训练框架，本仓库提供 Agent Loop 与 `compute_score`。

方案：[科研检索训练方案](docs/research/PLAN.md)。目录：[代码目录规划](docs/research/CODE_STRUCTURE.md)。

| 能跑什么 | 证据 |
| --- | --- |
| CPU harness、标签隔离、BM25 | `python -m unittest discover -t . -s tests` |
| PaperSearchQA 官方划分（54907 train + 5000 test） | `prepare --source papersearchqa`；16M 语料要加 `--with-pubmed-corpus` |
| Ollama `qwen3.5:4b` 8 题 Base | `outputs/psqa-qwen35-4b-base/metrics.json`（GGUF，不是 HF bf16） |
| verl GRPO | **未跑**。缺 CUDA、权重或 verl 时写 `not_run`，不填零分 |

## 快速开始

检索服务嵌在进程内（BM25），不用像 Search-R1 那样另起 retriever。

### 1. 环境

Python 3.11。优先 conda，没有 conda 再用 uv。

**CPU（harness、数据准备、看板、单测）** 不装 torch / vLLM / verl：

```bash
conda env create -f environment.yml
conda activate research-agent
pip install -e ".[data]"
python scripts/setup/check_env.py
python -m unittest discover -t . -s tests
```

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[data]"
python scripts/setup/check_env.py
python -m unittest discover -t . -s tests
```

`[data]` 是 pyarrow 与 huggingface-hub，用来读官方 parquet。

**GPU 训练节点** 再装训练栈。`torch`、`vllm`、`verl` 写在 `pyproject.toml` 的 `train` extra 和 `requirements-train.txt` 里，不写进默认 `requirements.txt`，避免 CPU 机器拉 CUDA 轮子：

```bash
pip install -e ".[data,train]"
# 等价：pip install -r requirements-train.txt
bash scripts/setup/install_verl.sh   # 把 verl clone 到 upstream/verl 再 editable 安装
python scripts/setup/check_env.py    # 应看到 torch / cuda / vllm / verl
```

`verl` 按官方方式从源码装（`git clone` + `pip install -e`），本仓库不把训练框架复制进 `research_agent/`。vLLM 需要 Linux + CUDA；macOS CPU 不要装 `.[train]`。

### 2. 数据与模型

工作目录里先放好原始数据和权重，脚本只做转换和 load。把 PaperSearchQA 的 train/test parquet 放到 `data/raw/papersearchqa/`，把 `pubmed.jsonl` 放到 `data/raw/pubmed_bioasq_2022/`。

```text
data/raw/papersearchqa/train-00000-of-00001.parquet
data/raw/papersearchqa/test-00000-of-00001.parquet
data/raw/pubmed_bioasq_2022/pubmed.jsonl
models/Qwen3.5-4B/          # HF 快照：config.json、tokenizer、权重
```

然后转换官方全集（54907 train + 5000 test；16M PubMed 摘要）。test 划分不进入 RL。

```bash
python -m research_agent.cli prepare \
  --source papersearchqa \
  --with-pubmed-corpus \
  --output data/processed/papersearchqa
```

得到 `data/processed/papersearchqa/public/{tasks,corpus}.jsonl` 和 `private/grading.jsonl`。公开文件不含答案。不加 `--with-pubmed-corpus` 时语料只有 gold 摘要，**不能**当成 16M 检索成绩。进程内 BM25 会把语料载入内存；16M 索引需要训练节点上的内存。

CPU 单测用 `tests/fixtures` 里的两篇短文，不进入训练或评测。50 题开发子集是 `--source papersearchqa-dev`，只用于流程冒烟和已测的 8 题冻结评测。

### 3. 推理

数据准备好之后：

```bash
python -m research_agent.cli eval \
  --data data/processed/papersearchqa \
  --baseline no_retrieval \
  --run-id psqa-no-retrieval \
  --output outputs/psqa-no-retrieval

python -m research_agent.cli run \
  --data data/processed/papersearchqa \
  --question "Which gene is mutated in childhood retinoblastoma?" \
  --model scripted
```

本机已装 [Ollama](https://ollama.com) 并 `ollama pull qwen3.5:4b` 时，跑已冻结的 8 题 Base（与 `outputs/psqa-qwen35-4b-base` 同一组 id，开发子集）：

```bash
python -m research_agent.cli eval \
  --data data/processed/papersearchqa-dev \
  --eval-config configs/evaluation/base_mini.yaml \
  --baseline agent \
  --policy ollama \
  --model-name qwen3.5:4b \
  --run-id psqa-qwen35-4b-base \
  --output outputs/psqa-qwen35-4b-base
```

或：`bash scripts/eval/run_frozen.sh`。Ollama GGUF 没有 token logprob，**不能**当作 GRPO 更新。

单题对话把 `--model` 换成 `ollama`：

```bash
python -m research_agent.cli run \
  --data data/processed/papersearchqa \
  --question "Which gene is mutated in childhood retinoblastoma?" \
  --model ollama
```

轨迹看板：

```bash
python -m research_agent.cli dashboard --outputs outputs --port 8765
```

打开 http://127.0.0.1:8765 。没有 GRPO 目录时显示「未运行」，不填零分。

### 4. 训练

对齐 PaperSearchQA / Search-R1：Qwen3.5-4B **直接 GRPO**，不用教师轨迹，也不把 LoRA SFT 当冷启动。要 CUDA 以及 `models/Qwen3.5-4B`。

```bash
bash scripts/train/verl_grpo.sh
```

入口是 `python -m verl.trainer.main_ppo`。Hydra 覆盖在 `configs/training/verl_grpo.yaml`，Agent Loop 在 `research_agent/training/verl/agent_loop.py`。无 CUDA / 无 verl 写 `not_run`。

A100 上一键试跑（先检查 CUDA 和本地 4B，再 GRPO）：

```bash
bash scripts/setup/gpu_node.sh
```

无 verl 时的 HuggingFace 组采集回退：`bash scripts/train/grpo.sh`。`scripts/train/sft.sh` 仍可对现成 messages jsonl 做可选 LoRA，不是默认路径。

训练后冻结评测（adapter 不存在则 `not_run`）：

```bash
python -m research_agent.cli eval-trained \
  --adapter outputs/grpo/adapter \
  --run-id frozen-grpo \
  --output outputs/frozen-grpo

python -m research_agent.cli compare \
  --runs outputs/psqa-qwen35-4b-base,outputs/psqa-no-retrieval,outputs/grpo \
  --output outputs/compare-frozen.json
```

## 数据口径

PaperSearchQA 官方划分约 5.5 万 train + 5 千 test。开发子集成绩不能和 16M 全语料成绩混报。

## 代码入口

1. `research_agent/harness/loop.py`
2. `research_agent/environment/tools.py`
3. `research_agent/grading/reward.py`
4. `research_agent/training/verl/agent_loop.py`
5. `research_agent/evaluation/planning.py`

`research_agent/contracts.py` 不含 gold。标签只在 `research_agent/grading/` 和 `data/*/private/`。
