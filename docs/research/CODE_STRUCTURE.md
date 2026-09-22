# 代码目录与依赖规划

状态：目标结构。CPU 包路径已按此树落地。GPU 训练目录有 adapter，没有测过的 checkpoint。

## 上游参考

2026-09-18 检查上游 `main` 的目录与核心入口。以下链接指向可变分支，只用于本次结构分析；引入代码前需要记录 commit、许可证和改动。

| 上游实际位置 | 职责与借鉴 | 本项目归属 |
| --- | --- | --- |
| [Search-R1 `search_r1/llm_agent/generation.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py) | 生成与搜索交错的循环 | `research_agent/harness/`，不直接复制其训练耦合 |
| [Search-R1 `search_r1/search/`](https://github.com/PeterGriffinJin/Search-R1/tree/main/search_r1/search) | 检索模块独立于 Agent 循环 | `research_agent/environment/` |
| [Search-R1 根目录](https://github.com/PeterGriffinJin/Search-R1) 的 `verl/`、`train_grpo.sh`、`infer.py` | 训练后端、启动入口与推理入口可定位 | 外部 `upstream/verl/`、薄脚本、统一 CLI |
| [Tongyi `inference/`](https://github.com/Alibaba-NLP/DeepResearch/tree/main/inference) 的 `react_agent.py`、`prompt.py`、`run_multi_react.py` | ReAct 运行、提示词与批量推理有明确入口 | 共用 harness、版本化提示词、评测 runner |

Search-R1 更适合参考搜索训练链路，Tongyi 更适合参考工具 Agent 推理组织。此处借鉴的是职责划分，不推断 Tongyi 的推理代码包含完整训练栈，也不沿用上游旧版本的环境组合。

## 目标目录

```text
Research-Agent/
├── README.md                         # 项目入口、真实状态、最小运行步骤
├── AGENTS.md                         # 开发约定
├── pyproject.toml                    # 本地包元数据、CLI 入口
├── environment.yml                   # Miniconda 环境与 Python 版本
├── requirements.txt                  # 经验证的运行依赖
├── requirements-train.txt            # GPU 训练额外依赖
├── research_agent/                   # 项目唯一 Python 包
│   ├── __init__.py
│   ├── cli.py                        # 参数解析与组件装配，业务逻辑下沉
│   ├── contracts.py                  # 公共 TaskInput、Action、Observation、Result、Event
│   ├── harness/
│   │   ├── loop.py                   # 唯一 Agent 交互循环
│   │   ├── state.py                  # episode 状态、预算与终止原因
│   │   ├── context.py                # 消息构建、观测裁剪、上下文上限
│   │   └── trajectory.py             # 事件与轨迹序列化
│   ├── models/
│   │   ├── base.py                   # 生成接口及 token/logprob 返回契约
│   │   ├── openai_compatible.py      # Ollama 与部署服务调用
│   │   ├── huggingface.py            # 本地 HF 生成，返回 token id 与 logprob
│   │   └── factory.py                # 按名称装配策略，harness 不依赖后端
│   ├── environment/
│   │   ├── tools.py                  # search/open/submit schema 与执行分发
│   │   ├── corpus.py                 # 文档、段落 ID 与快照读取
│   │   ├── retrieval.py              # 首版固定词法检索
│   │   └── server.py                 # 可选独立检索服务入口
│   ├── data/
│   │   ├── prepare.py                # data/raw → data/processed
│   │   ├── papersearchqa.py          # 读本地 parquet 与 pubmed dump
│   │   ├── qasper.py                 # 可选单论文转换
│   │   ├── pubmed.py                 # 读本地 abstracts.json 缓存
│   │   └── validate.py               # 去重、答案证据核验、泄露审计
│   ├── grading/
│   │   ├── contracts.py              # 私有 GradingSpec、Score，与公共输入隔离
│   │   ├── answers.py                # 答案归一化、别名与数据集评分协议
│   │   ├── evidence.py               # 引用合法性、标注匹配与支持率
│   │   └── reward.py                 # RL 奖励组合，复用上述评分函数
│   ├── training/
│   │   ├── export.py                 # 标准任务/轨迹转换为后端数据格式
│   │   ├── sft.py                    # 可选 assistant-only LoRA，不是默认训练路径
│   │   ├── grpo.py                   # 同题组采集与 clipped surrogate
│   │   ├── compat.py                 # GPU / verl / vLLM 探测
│   │   └── verl/
│   │       ├── agent_loop.py         # 注册 AgentLoopBase 子类并调用 harness；不是 trainer
│   │       ├── model_backend.py      # LLMServerClient.generate ↔ PolicyModel
│   │       ├── reward_adapter.py     # custom_reward_function.compute_score → grading/
│   │       └── launch.py             # 组装 Hydra 覆盖项并 exec verl.trainer.main_ppo
│   └── evaluation/
│       ├── runner.py                 # 固定任务集运行、评分、checkpoint 对比
│       ├── baselines.py              # 无检索与固定检索基线
│       ├── metrics.py                # 汇总、成本曲线、配对置信区间
│       ├── failures.py               # 工具失败、无解析、错答分布
│       └── planning.py               # 固定规划模型的下游盲评数据导出
├── prompts/
│   ├── agent.md                      # 训练与部署共用的系统提示词
│   └── verify.md                     # 独立核验提示词
├── configs/
│   ├── data/                         # 来源、划分、语料构建配置
│   ├── harness/                      # 工具、上下文、预算、终止配置
│   ├── models/                       # 模型 revision、服务位置、采样参数
│   ├── training/                     # GRPO 与可选 SFT 配置
│   ├── evaluation/                   # 冻结任务与指标配置
│   ├── experiments/                  # 组合上述配置的具名实验
│   └── upstreams.json                # 仓库 URL、固定 commit、补丁清单
├── scripts/
│   ├── setup/                        # 环境建立、依赖与设备检查
│   ├── data/                         # 准备语料的薄入口
│   ├── train/                        # GRPO、可选 SFT、checkpoint 导出入口
│   └── eval/                         # 批量评测启动入口
├── apps/
│   └── dashboard/                    # FastAPI 轨迹看板，读 outputs/ 事件
├── tests/
│   ├── support.py                    # unittest 共用夹具，不引入 pytest
│   ├── unit/                         # 预算、终止、格式、评分等纯逻辑
│   ├── integration/                  # harness/工具/后端接口及标签隔离
│   ├── gpu/                          # 单步更新、权重同步、导出重载；无 CUDA 时 skip
│   └── fixtures/                     # 单测夹具，不代表性能基准
├── docs/research/
│   ├── PLAN.md
│   └── CODE_STRUCTURE.md
├── upstream/                         # 忽略：固定版本的第三方检出
├── patches/                          # 仅在确需改上游时保存补丁与说明
├── data/                             # 工作目录：raw 已下载，processed 为 prepare 产物
├── models/                           # 工作目录：已下载的 HF 快照，训练只 load
└── outputs/                          # 忽略：运行配置快照、事件、评测、checkpoint
```

`research_agent/data/` 是处理代码，根目录 `data/` 是已下载的原始文件和 prepare 产物，两者不能混放。`models/` 放权重快照，配置里的路径指向该目录。

## 依赖与数据边界

- `contracts.py` 仅定义不含 gold 的公共契约，不导入训练框架、数据库或应用。
- `harness/` 依赖公共契约和模型/工具接口。具体后端由 CLI 或训练入口注入；harness 不导入 `training/`、`grading/`、`evaluation/` 或应用。
- `environment/` 读取公开语料与索引，不读取答案、评分依据和训练奖励。工具返回不携带 gold 标记。
- `grading/` 读取私有评分依据与完成轨迹，不向运行中的策略传回参考答案。目录隔离之外，输入装配也必须显式移除标签。
- `training/verl/` 是 verl 的项目插件（Agent Loop + reward），不是第二个训练框架。调度、GRPO 更新和权重同步由 `verl.trainer.main_ppo` 执行；CPU 数据处理与普通推理无需加载 CUDA、Ray 或 verl。
- `evaluation/runner.py` 调用 `harness/loop.py`，再交给评分器；不另写搜索循环。
- `apps/` 消费公开结果与事件，不读取私有标签、梯度或奖励。
- `scripts/` 只处理启动、参数转发与进程环境；划分、奖励、工具执行逻辑必须在 Python 包内。

## 关键调用链

| 操作 | 调用顺序 |
| --- | --- |
| 准备训练任务 | `prepare → validate → data/processed/` |
| GRPO 组采集（HF 回退） | `training/grpo.collect_group → harness → grading → export_grpo_group` |
| GRPO 训练 | `verl.trainer.main_ppo → VerlResearchAgentLoop.run → harness → model_backend + environment` |
| RL 评分更新 | `完成轨迹 → reward_adapter.compute_score → grading/reward → verl trainer` |
| 在线推理 | `cli → harness → openai_compatible + environment → Result` |
| 批量评测 | `evaluation/runner → harness → grading → metrics → outputs/` |

训练时使用后端返回的真实 token ID、log probability 和生成 mask，普通 API 推理缺少这些字段时明确记为不可用于 RL 更新，不做估算填充。

异步工具执行放在 `harness/loop.py` 和工具实现中，跨 episode 的调度尽量复用 verl。完整异步采样/训练队列仅在完成性能剖析后扩展 `training/`。

## 配置、依赖与产物约定

一个实验配置引用数据、harness、模型、训练及评测配置，命令行覆盖值连同解析后的配置一起写入 `outputs/<run_id>/resolved_config.json`。Manifest 记录代码 commit、脏工作区差异摘要、语料和提示词哈希、模型 revision、上游版本。密钥通过环境变量传入，配置快照必须脱敏。

运行产物统一放 `outputs/<run_id>/` 下的 `events.jsonl`、`episodes.jsonl`、`metrics.json`、`checkpoints/`。源数据、索引与划分放在 `data/`，用内容哈希引用。标签独立保存，部署入口只加载公共任务与语料。每个文件的 schema 随功能实现再固定。

按 AGENTS.md 使用 Miniconda，交付 `environment.yml` 和 `requirements.txt`，仅在无 conda 时降级为 uv。GPU 依赖单独列于 `requirements-train.txt`；不把上游旧版本依赖直接当作 Qwen3.5 的可用组合。当前尚未验证依赖，不填造锁定版本。

第三方代码放 `upstream/` 并在 `configs/upstreams.json` 固定版本，不将上游整个训练框架复制进项目包。确需补丁时附上原因、基线 commit、验证命令与撤销条件。后续引用或复制上游实现时保留许可与归属。

## 按功能逐步建目录

1. 建公共契约、最小 harness、文档工具与 CPU 单测，使一条任务完成搜索、阅读、提交和事件落盘。
2. 读本地 raw 数据、任务验证和评分器，再接模型服务，跑真实 Base 小基线。
3. 接 verl adapter 与 GPU 单步验证。
4. 建批量评测、科研案例、下游规划评测与看板。
5. 测量性能后扩展并发与异步训练。

面试源码阅读顺序是 `harness/loop.py → environment/tools.py → grading/reward.py → training/verl/agent_loop.py → evaluation/planning.py`，分别对应交互、环境、学习信号、训练接入和业务价值。GPU 训练结果仍以实测为准。
