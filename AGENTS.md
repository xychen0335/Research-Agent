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

- 使用 git 维护。
- commit 采用：feat / fix / chore / docs / refactor + 中文说明

## 当前方向（2026-09-18）

- 用户已决定删除旧 SQL Agent 骨架，不继续扩展 BIRD 路线。
- 当前方案以 `docs/research/PLAN.md` 为准：公开科研资料检索、证据阅读、多轮 SFT 与 RL。
- Tongyi DeepResearch 用作交互设计和教师候选；训练后端拟用 verl，兼容性必须实测。
- 腾讯 AutoTraining 仅作为经历背景。本独立项目使用公开数据，不复制企业代码或运行记录。
- 计划、CPU 验证和真实 GPU 训练结果必须分别说明。
