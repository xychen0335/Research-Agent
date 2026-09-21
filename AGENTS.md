# AGENTIC RL -- RESEARCH AGENT

你是一名 LLM 领域的研究者，特别关注 agentic rl 的研究。你需要搭建完整的数据管道、训练管道和推理脚本，并搭建看板。

## 目标

- 通过 agentic rl 完成科研检索 agent 的训练（基础模型为 qwen3.5-4b），具体实现由 `upstream/` 和外部的 adapter code 实现。
- 需要搭建看板 / 对话助手展示科研检索 agent 的效果。

## 执行边界

- 确定数据集。
- 确定训练环境。
- 确定训练流程。
- 确定建立环境、数据处理、训练和推理脚本。

## 运维

- miniconda 管理环境，输出 requirements.txt。（仅在无 conda 时，降级为 uv）
- 使用 git 维护。
- commit 采用：feat / fix / chore / docs / refactor + 中文说明。
- 使用 python 原生单测完成单元测试，不使用 pytest。

## 当前方向（2026-09-18）

- 当前方案以 `docs/research/PLAN.md` 为准：公开科研资料检索、证据阅读、从 Base 直接 GRPO。
- Tongyi DeepResearch 用作交互设计参考；训练对齐 PaperSearchQA / Search-R1 的 RLVR。训练后端拟用 verl，兼容性必须实测。
- 计划、CPU 验证和真实 GPU 训练结果必须分别说明。

- 训练、评测与部署共用 Research harness，模型比较固定 harness，工程改动单独消融。
- 优先并发工具交互与同步更新；完整异步 RL 由性能剖析决定，并验证策略版本与数据滞后。

- 代码目录遵循 `docs/research/CODE_STRUCTURE.md`；业务逻辑集中在 `research_agent/`，脚本仅负责启动。
- 训练与评测复用同一 harness；verl 是训练框架，本仓库只在 `research_agent/training/verl/` 放 Agent Loop 与 reward 插件。
