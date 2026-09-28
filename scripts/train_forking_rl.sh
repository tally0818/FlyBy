#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/forking_rl.yaml}
PYTHON_BIN=${PYTHON_BIN:-python3}

set +x
[ -f .env ] && set -a && source .env && set +a
set -x

cfgget() { "$PYTHON_BIN" -c "import yaml,sys; c=yaml.safe_load(open('$CFG'))
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(v)" "$1"; }

model_name=$(cfgget model)
model_revision=$(cfgget model_revision)
model_name=$("$PYTHON_BIN" -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "$model_name" "$model_revision")
seed=$(cfgget seed)
export PYTHONHASHSEED=$seed
train_jsonl=$(cfgget data.train)
reward_function=$(cfgget reward.function)
n=$(cfgget rollout.n)
temperature=$(cfgget rollout.temperature)
top_p=$(cfgget rollout.top_p)
top_k=$(cfgget rollout.top_k)
log_prob_micro_batch_size=$(cfgget rollout.log_prob_micro_batch_size_per_gpu)
batch_size=$(cfgget rollout.batch_prompts)
max_prompt_length=$(cfgget rollout.max_prompt_tokens)
max_response_length=$(cfgget rollout.max_response_tokens)
adv=$(cfgget optim.adv)
norm_adv_by_std=$(cfgget optim.norm_adv_by_std_in_grpo)
loss_agg_mode=$(cfgget optim.loss_agg_mode)
entropy_top_ratio=$(cfgget optim.entropy_top_ratio)
lr=$(cfgget optim.lr)
ppo_mini_batch_size=$(cfgget optim.ppo_mini_batch_size)
ppo_micro_batch_size=$(cfgget optim.ppo_micro_batch_size_per_gpu)
ppo_max_token_len=$(cfgget optim.ppo_max_token_len_per_gpu)
ppo_epochs=$(cfgget optim.ppo_epochs)
clip_ratio=$(cfgget optim.clip_ratio)
clip_ratio_low=$(cfgget optim.clip_ratio_low)
eps_high=$(cfgget optim.eps_high)
entropy_coeff=$(cfgget optim.entropy_coeff)
use_kl_loss=$(cfgget optim.use_kl_loss)
kl_coef=$(cfgget optim.kl_coef)
kl_loss_type=$(cfgget optim.kl_loss_type)
grad_ckpt=$(cfgget optim.grad_ckpt)
dynamic_bsz=$(cfgget optim.dynamic_bsz)
optimizer_offload=$(cfgget optim.optimizer_offload)
gpu_mem=$(cfgget optim.gpu_memory_utilization)
tp=$(cfgget optim.tp)
total_steps=$(cfgget run.total_steps)
save_freq=$(cfgget run.save_freq)
val_freq=$(cfgget run.val_freq)
val_before_train=$(cfgget run.val_before_train)
wandb_project=$(cfgget run.wandb_project)
out_root=$(cfgget run.out_root)

[ -f "$train_jsonl" ] || {
  echo "ERROR: $train_jsonl missing — run scripts/build_data.sh first" >&2
  exit 1
}
[ -f "$reward_function" ] || {
  echo "ERROR: reward function not found: $reward_function" >&2
  exit 1
}
"$PYTHON_BIN" -c 'import flash_attn' >/dev/null 2>&1 || {
  echo "ERROR: flash-attn is required with use_remove_padding=True." >&2
  echo "Install it in the active training environment before launching." >&2
  exit 1
}

train_parquet=${train_jsonl%.jsonl}_forking_rl.parquet
if [ ! -f "$train_parquet" ] || [ "$train_jsonl" -nt "$train_parquet" ] || \
   [ src/data/to_verl_parquet.py -nt "$train_parquet" ]; then
  "$PYTHON_BIN" -m src.data.to_verl_parquet \
    --input "$train_jsonl" --out "$train_parquet" --no-tool
fi

val_files_override="[$train_parquet]"

run_id="forking_rl"
out_dir="${out_root%/}/${run_id}"
mkdir -p "$out_dir" outputs/logs
cp "$CFG" "$out_dir/forking_rl.yaml.snapshot"
export VERL_RUN_ID=$run_id

case "${WANDB_ENABLED:-false}" in
  1|true|TRUE|yes|YES) trainer_loggers="['console','wandb']" ;;
  0|false|FALSE|no|NO|'') trainer_loggers="['console']" ;;
  *) echo "ERROR: WANDB_ENABLED must be true or false" >&2; exit 1 ;;
esac

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std \
    data.train_files="$train_parquet" \
    data.val_files="$val_files_override" \
    data.train_batch_size=$batch_size \
    data.seed=$seed \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation=right \
    custom_reward_function.path="$PWD/$reward_function" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.model.enable_gradient_checkpointing=$grad_ckpt \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.entropy_top_ratio=$entropy_top_ratio \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.clip_ratio=$clip_ratio \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$eps_high \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len \
    actor_rollout_ref.actor.ppo_epochs=$ppo_epochs \
    actor_rollout_ref.actor.use_dynamic_bsz=$dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=$seed \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_loggers" \
    trainer.project_name="$wandb_project" \
    trainer.experiment_name="$run_id" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$val_freq \
    trainer.val_before_train=$val_before_train \
    trainer.total_training_steps=$total_steps \
    trainer.default_local_dir="$out_dir" \
    "$@" 2>&1 | tee "outputs/logs/train_${run_id}.log"
