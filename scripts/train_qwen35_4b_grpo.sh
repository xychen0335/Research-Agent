#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_ROOT="${ROOT}/upstream/verl"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
: "${LORA_ADAPTER_PATH:?Set LORA_ADAPTER_PATH to the exported SFT adapter directory}"
: "${BIRD_DB_DIR:?Set BIRD_DB_DIR to the BIRD database directory}"
TRAIN_FILE=${TRAIN_FILE:-${ROOT}/data/processed/bird_rl_train.jsonl}
VAL_FILE=${VAL_FILE:-${ROOT}/data/processed/bird_rl_val.jsonl}
SAVE_DIR=${SAVE_DIR:-${ROOT}/checkpoints/qwen35-4b-bird-grpo}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
ACTOR_LR=${ACTOR_LR:-3e-6}

[[ -f "${TRAIN_FILE}" ]] || { echo "Missing TRAIN_FILE=${TRAIN_FILE}" >&2; exit 1; }
[[ -f "${VAL_FILE}" ]] || { echo "Missing VAL_FILE=${VAL_FILE}" >&2; exit 1; }
[[ -d "${LORA_ADAPTER_PATH}" ]] || { echo "Missing LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH}" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "Requires an NVIDIA GPU." >&2; exit 1; }

export PYTHONPATH="${ROOT}:${ROOT}/upstream/BIRD-RL:${VERL_ROOT}:${PYTHONPATH:-}"
export BIRD_DB_DIR
cd "${VERL_ROOT}"
uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.return_raw_chat=True \
  +data.need_tools_kwargs=True \
  data.train_batch_size=4 \
  data.max_prompt_length=8192 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=2 \
  data.truncation=error \
  reward.custom_reward_function.path="${ROOT}/upstream/BIRD-RL/bird_rl/rewards/bird_reward_agentic.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=4 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${ROOT}/configs/bird_tools.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.default_agent_loop=bird_sql_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${ROOT}/configs/agent_loop.yaml" \
  trainer.logger='[console,tensorboard]' \
  trainer.project_name=agenticrl-bird \
  trainer.experiment_name=qwen35-4b-sft-grpo \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.val_before_train=True \
  trainer.log_val_generations=8 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${@}"
