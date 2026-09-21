# AgenticRL Research Lab

CPU harness 可以在合成开发集和 PaperSearchQA 官方划分的 50 题子集上完成搜索、阅读、提交、评分和事件落盘。本机已用 Ollama `qwen3.5:4b`（Q4_K_M）在 8 道 train 题上跑过 Base 小验证集。SFT / GRPO、16M PubMed 全量索引和 A100/verl 指标尚未运行。

项目把科研检索做成 AutoResearch 流程里的子 Agent：调用方给出问题与预算，Agent 在冻结语料上决定查询、阅读和提交。训练、评测与服务走同一个 `research_agent.harness.loop`。与 AutoTraining 的衔接是公开 JSON 契约，没有接入腾讯生产系统。

方案正文在 [科研检索训练方案](docs/research/PLAN.md)。目录约定在 [代码目录规划](docs/research/CODE_STRUCTURE.md)。

## 方案 / CPU 验证 / 真实模型

| 项目 | 方案 | 当前证据 |
| --- | --- | --- |
| 32 题合成开发集、标签隔离、BM25 | 交付 1–2 | CPU 测试覆盖 |
| PaperSearchQA 官方 50 题子集 | 交付 1 | 已审计；不是 16M 全语料 |
| 无检索 / 固定 RAG 脚本基线 | 对照 | 脚本策略，EM 不能当模型成绩 |
| Qwen3.5-4B Base 小验证集 | 交付 3 | 本机 Ollama 8 题：`outputs/psqa-qwen35-4b-base/metrics.json`，EM 0.5，提交率 0.75，工具成功率 0.97。量化 GGUF，不是 HF bf16，也不是 16M 语料 |
| 教师轨迹 | 交付 4 前置 | 4B：79 条 EM keep 4。9B：138 条 EM keep 10；别名跨度重筛后 `outputs/teacher-qwen35-9b/sft_short.jsonl` 为 31 道短答案（评测 EM 仍不给长句加分） |
| LoRA SFT / GRPO / verl | 交付 4–5 | 短答案教师 31 条已过门槛。本机无 HF 4B、无 CUDA、磁盘约 9 Gi、9B GGUF 仍占内存。SFT 与 HF 推理共用 chat template；A100 入口 `scripts/setup/gpu_node.sh`。缺设备或权重时写 `not_run`，不填零分 |
| 冻结评测入口 | 交付 5 | `scripts/eval/run_frozen.sh` 固定 8 个 train id。训练后对照用 `eval-trained`（无 adapter 则 `not_run`）和 `scripts/eval/compare_frozen.sh` |
| 看板未运行占位 | 交付 5 | SFT / GRPO checkpoint 无目录时显示「未运行」，不填零分 |
| 16M PubMed 索引 | 评测口径 | 未下载 |
| 完整异步 RL | 交付 6 | 队列与策略滞后门控在 `training/async_rl.py`；无 GPU 剖析报告时 `allow_async_rl` 保持关闭 |

## 运行

本机没有 conda 时用 uv（`scripts/setup/check_env.py` 会打印探测结果）。

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
python scripts/setup/check_env.py
python -m unittest discover -t . -s tests
python -m research_agent.cli prepare --source papersearchqa-dev --output data/processed/papersearchqa-dev
python -m research_agent.cli eval --data data/processed/papersearchqa-dev --baseline no_retrieval --run-id psqa-no-retrieval --output outputs/psqa-no-retrieval
python -m research_agent.cli eval --data data/processed/papersearchqa-dev --baseline agent --policy ollama --model-name qwen3.5:4b --split train --limit 8 --run-id psqa-qwen35-4b-base --output outputs/psqa-qwen35-4b-base
python scripts/data/teacher.py --data data/processed/papersearchqa-dev --policy ollama --limit 20
python -m research_agent.cli eval-trained --adapter outputs/sft/adapter --run-id frozen-sft --output outputs/frozen-sft
python -m research_agent.cli compare --runs outputs/psqa-qwen35-4b-base,outputs/psqa-no-retrieval,outputs/sft --output outputs/compare-frozen.json
python -m research_agent.cli plan-review --outputs outputs --task-id cs-005 --output outputs/planning-review.json
python scripts/setup/gpu_node.sh
python scripts/train/gpu_mini.sh
python scripts/train/verl_grpo.sh
python -m research_agent.cli dashboard --outputs outputs --port 8766
```

`scripts/train/sft.sh` 是 LoRA 冷启动。`scripts/train/gpu_mini.sh` 在 CUDA+verl 上会先 SFT，再启动 `verl.trainer.main_ppo`。`scripts/train/grpo.sh` 是无 verl 时的 HuggingFace 组采集 / LoRA 回退。verl GRPO 用 `scripts/train/verl_grpo.sh`，本仓库不实现第二套 trainer。没有 CUDA/MPS 权重时状态为 `not_run`，退出码 2。Ollama 生成没有 token logprob，不能当作 GRPO 更新。Ollama 上的 Qwen3.5 GGUF 不能直接吃 XML 系统提示（会 EOF），本地策略把原生 tools 转成 harness 的 `<tool_call>`；HuggingFace / verl 训练仍用 `prompts/agent.md`。

## 数据

合成开发集在 `research_agent/data/sources/synthetic_dev.py`：16 道生物医学事实题、12 道计算机论文条件题、4 道不可答题。语料是仓库内原创短文，不是 PubMed 或 arXiv 原文。`cs-005` 是面试用的对照题：BoostNet 相对 SyncRet 的 4.2 点提升同时换了 ExtraMix-2M 和 BoostSplit。

PaperSearchQA 官方集：<https://huggingface.co/datasets/jmhb/PaperSearchQA>，Hub revision `563d32ebcf5a8081ed67abe4f7afe0ae614be1e1`，QA 为 MIT。本地 parquet sha256 写在 `data/processed/papersearchqa-dev/manifest.json`。开发子集是 train 40 + test 10，摘要来自 NCBI，外加 40 篇干扰文档。这不是 16M 全语料成绩。测试划分不进入教师 / SFT / RL。

## 代码入口

面试阅读顺序：

1. `research_agent/harness/loop.py`
2. `research_agent/environment/tools.py`
3. `research_agent/grading/reward.py`
4. `research_agent/training/verl/agent_loop.py`
5. `research_agent/evaluation/planning.py`

`research_agent/contracts.py` 不含 gold。评分标签只出现在 `research_agent/grading/` 和 `data/*/private/`。
