# AgenticRL Research Lab

基于 Qwen3.5-4B 的科研检索 Agentic RL 项目。目标是在固定检索环境和工具预算下，比较 Base、SFT、SFT + GRPO 的答案质量、证据支持率和调用成本。

## 当前状态

已确定研究方向并编写[科研检索训练方案](docs/research/PLAN.md)。数据审计、检索工具环境、训练脚本和看板尚待实现，暂无训练 checkpoint 或实验指标。

旧 SQL Agent 的实现、配置、脚本、测试、演示界面及本地 BIRD 数据已移除。

## 实施路线

1. 审计 PaperSearchQA 的数据、许可、划分与检索语料。
2. 实现统一的搜索、阅读和答案提交工具，验证标签隔离与预算限制。
3. 建立 Base 和固定检索基线，生成并验证 SFT 轨迹。
4. 验证 Qwen3.5-4B 与 verl 的兼容性，运行 SFT 和多轮 GRPO 小试。
5. 完成冻结评测与真实轨迹对比看板。

Tongyi DeepResearch 用于交互设计参考和教师候选，Search-R1 用于搜索训练实现参考。详细范围、奖励设计和验收标准见方案。
