#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/search_r1.yaml}
PYTHON_BIN=${PYTHON_BIN:-python3}
set +x
[ -f .env ] && set -a && source .env && set +a
set -x

cfgget() { "$PYTHON_BIN" -c "import yaml,sys; c=yaml.safe_load(open('$CFG'))
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(v)" "$1"; }
cfgget_csv() { "$PYTHON_BIN" -c "import yaml,sys; c=yaml.safe_load(open('$CFG'))
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(','.join(map(str, v)))" "$1"; }

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
own_tokens_total=$(cfgget rollout.own_tokens_total)
max_turns=$(cfgget rollout.max_turns)
max_obs_length=$(cfgget rollout.max_obs_tokens)
stop_tokens=$(cfgget_csv rollout.stop)
mask_observations=$(cfgget rollout.mask_observations)
tool_type=$(cfgget search.type)
retriever_url=$(cfgget search.retriever_url)
topk=$(cfgget search.topk)
timeout_s=$(cfgget search.timeout_s)
concurrency=$(cfgget search.concurrency)
max_query_chars=$(cfgget search.max_query_chars)
max_result_chars=$(cfgget search.max_result_chars)
router_workers=$(cfgget search.router_workers)
reward_timeout=$(( timeout_s + 15 ))
adv=$(cfgget optim.adv)
norm_adv_by_std=$(cfgget optim.norm_adv_by_std_in_grpo)
loss_agg_mode=$(cfgget optim.loss_agg_mode)
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
max_concurrent=$(cfgget run.max_concurrent_trajectories)
wandb_project=$(cfgget run.wandb_project)
out_root=$(cfgget run.out_root)
total_steps=$(cfgget run.total_steps)
save_freq=$(cfgget run.save_freq)
val_freq=$(cfgget run.val_freq)
val_before_train=$(cfgget run.val_before_train)

if ! [[ "$own_tokens_total" =~ ^[1-9][0-9]*$ ]] || (( own_tokens_total > max_response_length )); then
  echo "ERROR: own-token budget must be positive and <= max_response_tokens" >&2
  exit 1
fi
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
  exit 1
}

"$PYTHON_BIN" -m src.tools.search_r1 \
  --url "$retriever_url" --topk "$topk" --timeout-s "$timeout_s"

export SEARCH_R1_RETRIEVER_URL=$retriever_url
export SEARCH_R1_TOPK=$topk
export SEARCH_R1_TIMEOUT_S=$timeout_s
export SEARCH_R1_CONCURRENCY=$concurrency
export SEARCH_R1_MAX_QUERY_CHARS=$max_query_chars
export SEARCH_R1_MAX_RESULT_CHARS=$max_result_chars
export SEARCH_R1_MAX_TURNS=$max_turns

train_parquet=${train_jsonl%.jsonl}_search_r1.parquet
"$PYTHON_BIN" -m src.data.to_verl_parquet \
  --input "$train_jsonl" --out "$train_parquet" --search-r1
val_files_override="[$train_parquet]"

run_id="search_r1"
out_dir="${out_root%/}/${run_id}"
mkdir -p "$out_dir" outputs/logs
cp "$CFG" "$out_dir/search_r1.yaml.snapshot"
export VERL_RUN_ID=$run_id

action_stop_tokens_file=$(mktemp)
printf '%s' "$stop_tokens" > "$action_stop_tokens_file"
host=127.0.0.1
port=$(shuf -i 31001-32000 -n 1 2>/dev/null || jot -r 1 31001 32000)
tool_server_url="http://$host:$port/get_observation"
"$PYTHON_BIN" -m verl_tool.servers.serve --host "$host" --port "$port" \
  --tool_type "$tool_type" --workers_per_tool "$concurrency" \
  --max_concurrent_requests "$max_concurrent" \
  --uvi_workers 1 --router_workers "$router_workers" \
  > "outputs/logs/tool_server_${run_id}.log" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null || true; rm -f "$action_stop_tokens_file"' EXIT
sleep 5

case "${WANDB_ENABLED:-false}" in
  1|true|TRUE|yes|YES) trainer_loggers="['console','wandb']" ;;
  0|false|FALSE|no|NO|'') trainer_loggers="['console']" ;;
  *) echo "ERROR: WANDB_ENABLED must be true or false" >&2; exit 1 ;;
esac

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m verl_tool.trainer.main_ppo \
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
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url="$tool_server_url" \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_own_tokens_total=$own_tokens_total \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.force_finish_for_last_turn=True \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens="$action_stop_tokens_file" \
    actor_rollout_ref.agent.tool_call_timeout=$reward_timeout \
    actor_rollout_ref.agent.tool_call_max_retries=0 \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent \
    actor_rollout_ref.agent.enable_mtrl=False \
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
