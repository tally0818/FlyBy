#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/rl.yaml}
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
set +x
[ -f .env ] && set -a && source .env && set +a
set -x

cfgget() { python -c "import yaml,sys; c=yaml.safe_load(open('$CFG'));
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(v)" "$1"; }
cfgget_csv() { python -c "import yaml,sys; c=yaml.safe_load(open('$CFG'));
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(','.join(map(str, v)))" "$1"; }
cfgget_json() { python -c "import yaml,json,sys; c=yaml.safe_load(open('$CFG'))
v=c
for k in sys.argv[1].split('.'): v=v[k]
print(json.dumps({str(kk): vv for kk, vv in v.items()}, separators=(',', ':')))" "$1"; }
cfgget_default() { python -c "import yaml,sys; v=yaml.safe_load(open('$CFG'))
for k in sys.argv[1].split('.'):
    if not isinstance(v, dict) or k not in v:
        print(sys.argv[2]); break
    v=v[k]
else: print(v)" "$1" "$2"; }

model_name=$(cfgget model)
seed=$(cfgget seed)
export PYTHONHASHSEED=$seed
train_jsonl=$(cfgget data.train)
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
stop_tokens=$(cfgget_csv rollout.stop)
mask_observations=$(cfgget rollout.mask_observations)
tool_type=$(cfgget tool.type)
depth_tiers=$(cfgget_json tool.depths)

max_obs_length=$(python -c "import json,sys; print(max(t['max_tokens'] for t in json.loads(sys.argv[1]).values()) + 16)" "$depth_tiers")
timeout_s=$(cfgget tool.timeout_s)
max_retries=$(cfgget tool.max_retries)

tool_request_timeout=$(( timeout_s * (max_retries + 1) + 30 ))
reward_manager=$(cfgget reward.manager)
lam=$(cfgget reward.lam)
norm_cost=$(cfgget_default reward.norm_cost false)
p_gpu_hr=$(cfgget reward.p_gpu_hr)
throughput_model=$(cfgget reward.throughput_model)
throughput_json=$(cfgget reward.throughput_json)
adv=$(cfgget optim.adv)
norm_adv_by_std=$(cfgget_default optim.norm_adv_by_std_in_grpo true)
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
total_steps=$(cfgget run.total_steps)
save_freq=$(cfgget run.save_freq)
val_freq=$(cfgget run.val_freq)
val_before_train=$(cfgget run.val_before_train)
wandb_project=$(cfgget run.wandb_project)
max_concurrent=$(cfgget run.max_concurrent_trajectories)
out_root=$(cfgget run.out_root)

if ! [[ "$own_tokens_total" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: rollout.own_tokens_total must be a positive integer, got: $own_tokens_total" >&2
  exit 1
fi
if (( own_tokens_total > max_response_length )); then
  echo "ERROR: rollout.own_tokens_total ($own_tokens_total) exceeds rollout.max_response_tokens ($max_response_length)" >&2
  exit 1
fi

[ -f "$throughput_json" ] || { echo "ERROR: $throughput_json missing — run scripts/bench_throughput.sh first"; exit 1; }
python -c 'import flash_attn' >/dev/null 2>&1 || {
  echo "ERROR: flash-attn is required for GRPO with use_remove_padding=True."
  echo "Install it in the active environment with:"
  echo "  uv pip install --python .venv/bin/python packaging psutil ninja"
  echo "  MAX_JOBS=4 uv pip install --python .venv/bin/python flash-attn --no-build-isolation"
  exit 1
}

export EFFIE_DEPTH_TIERS="$depth_tiers"
export EFFIE_Q_MAX_CHARS=$(cfgget tool.q_max_chars)
export EFFIE_THINKING=$(cfgget_default tool.thinking false)
export EFFIE_TIMEOUT_S=$timeout_s
export EFFIE_MAX_RETRIES=$max_retries
export EFFIE_BUDGET_USD=$(cfgget tool.budget_usd)
export EFFIE_CONCURRENCY=$(cfgget tool.concurrency)
export EFFIE_GOLD_REDACTION=$(cfgget tool.gold_redaction)
export LRM_MAX_CONCURRENCY=$EFFIE_CONCURRENCY

train_parquet=${train_jsonl%.jsonl}.parquet
python -m src.data.to_verl_parquet --input "$train_jsonl" --out "$train_parquet"
for suite in gpqa_diamond; do
  [ -f "data/eval/${suite}.parquet" ] || \
    python -m src.data.to_verl_parquet --input "data/eval/${suite}.jsonl" \
      --out "data/eval/${suite}.parquet" --split eval
done
val_files="[data/eval/gpqa_diamond.parquet]"

default_run_id="rl"
run_id=${RUN_ID:-$default_run_id}
if ! [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ERROR: RUN_ID must contain only letters, numbers, '.', '_' or '-': $run_id" >&2
  exit 1
fi
out_dir="${out_root%/}/${run_id}"
mkdir -p "$out_dir" outputs/logs
cp "$CFG" "$out_dir/rl.yaml.snapshot"
export VERL_RUN_ID=$run_id

action_stop_tokens_file=$(mktemp)
printf '%s' "$stop_tokens" > "$action_stop_tokens_file"

host=127.0.0.1
port=$(shuf -i 30000-31000 -n 1 2>/dev/null || jot -r 1 30000 31000)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port \
  --tool_type "$tool_type" --workers_per_tool 1 \
  --max_concurrent_requests "$max_concurrent" \
  --uvi_workers 1 --router_workers "$(cfgget tool.router_workers)" \
  > outputs/logs/tool_server_${run_id}.log 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null || true' EXIT
sleep 5

case "${WANDB_ENABLED:-false}" in
  1|true|TRUE|yes|YES) trainer_loggers="['console','wandb']" ;;
  0|false|FALSE|no|NO|'') trainer_loggers="['console']" ;;
  *) echo "ERROR: WANDB_ENABLED must be true or false" >&2; exit 1 ;;
esac

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$adv \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std \
    data.train_files="$train_parquet" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.seed=$seed \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.lam=$lam \
    +reward_model.reward_kwargs.norm_cost=$norm_cost \
    +reward_model.reward_kwargs.p_gpu_hr=$p_gpu_hr \
    +reward_model.reward_kwargs.throughput_json=$throughput_json \
    +reward_model.reward_kwargs.throughput_model=$throughput_model \
    actor_rollout_ref.model.path=$model_name \
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
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_mem \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=$seed \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_own_tokens_total=$own_tokens_total \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.force_finish_for_last_turn=True \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.tool_call_timeout=$tool_request_timeout \
    actor_rollout_ref.agent.tool_call_max_retries=0 \
    actor_rollout_ref.agent.max_concurrent_trajectories=$max_concurrent \
    actor_rollout_ref.agent.enable_mtrl=False \
    trainer.critic_warmup=0 \
    trainer.logger="$trainer_loggers" \
    trainer.project_name=$wandb_project \
    trainer.experiment_name=$run_id \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$val_freq \
    trainer.val_before_train=$val_before_train \
    trainer.total_training_steps=$total_steps \
    trainer.default_local_dir="$out_dir" \
    "$@" 2>&1 | tee "outputs/logs/train_${run_id}.log"
