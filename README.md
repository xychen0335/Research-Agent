# AgenticRL Research Lab

基于 Qwen3.5-4B 的科研检索 Agentic RL 项目。目标是在固定检索环境和工具预算下，比较 Base、SFT、SFT + GRPO 的答案质量、证据支持率和调用成本。

项目作为 AutoResearch 工作流中的科研检索子 Agent，为实验规划提供答案、证据、适用条件和未解决问题。训练、评测与部署共用轻量 harness，模型自主决定搜索、阅读和提交时机。与 AutoTraining 的衔接目前为接口设计，尚未接入生产系统。

## 当前状态

已确定研究方向并编写[科研检索训练方案](docs/research/PLAN.md)。数据审计、检索工具环境、训练脚本和看板尚待实现，暂无训练 checkpoint 或实验指标。

## 实施路线

1. 审计 PaperSearchQA 的数据、许可、划分与检索语料。
2. 实现统一 harness 和搜索、阅读、提交工具，验证标签隔离、预算与训练部署一致性。
3. 建立 Base 和固定检索基线，生成并验证 SFT 轨迹。
4. 验证 Qwen3.5-4B 与 verl 的兼容性，运行 SFT 和多轮 GRPO 小试。
5. 用计算机科研任务完成冻结评测、下游规划盲评与真实轨迹对比看板。
6. 先实现并发交互与同步更新，再依据瓶颈测量决定是否加入完整异步 RL。

Tongyi DeepResearch 用于交互设计参考和教师候选，Search-R1 用于搜索训练实现参考。详细范围、奖励设计和验收标准见方案。

## 代码组织

目录与依赖规则见[代码目录规划](docs/research/CODE_STRUCTURE.md)。核心逻辑集中在 `research_agent/`，训练与推理共用 harness；verl 接入集中在 `training/verl/`，薄脚本只负责启动。目前仍处于设计阶段，代码目录随功能实现逐步创建。
