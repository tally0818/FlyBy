# Knowing When Thinking Is Not Enough:
## Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge

Official implementation of **FlyBy**, which trains small reasoning models to
reason locally and selectively query stronger external models when additional
information is needed.

This repository contains the complete training and evaluation pipeline for
FlyBy, together with ForkingRL and Search-R1 baselines. The default experiments
use Qwen3-4B.
## 1. Setup

Local training and evaluation use Linux, Python 3.10, CUDA 12.8, and an NVIDIA GPU.
DeepSeek API evaluation requires no GPU. Run the commands below
from the repository root with the virtual environment activated.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip sync requirements.lock

uv pip install packaging psutil ninja
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

cp .env.example .env
```

Set the following values in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | External model calls during data synthesis, FlyBy RL, tool-enabled evaluation, and DeepSeek evaluation. |
| `HF_TOKEN` | Hugging Face access for datasets that require authentication, including GPQA. |
| `WANDB_ENABLED` | Set to `true` to enable training logs in Weights & Biases; defaults to `false`. |
| `WANDB_API_KEY` | Weights & Biases credentials when logging is enabled. |

For Search-R1, also install the retrieval dependencies:

```bash
uv pip install -r requirements-retriever.txt
```

## 2. Measure inference throughput

Measure throughput before FlyBy RL and before local-model evaluation with cost accounting.
The cost model uses separate prefill and decoding rates, stored in
`outputs/bench/throughput.json`.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/bench_throughput.sh --models Qwen/Qwen3-4B
```

Use the same serving GPU for measurement and evaluation. The paper uses an H200;
measurements on other hardware will produce different cost estimates. The GPU
price defaults to USD 3.49/hour. Set `reward.p_gpu_hr` in `configs/rl.yaml` and
`pricing.p_gpu_hr` in `configs/eval.yaml` if using a different price.

FlyBy, ForkingRL, and Search-R1 checkpoints trained from Qwen3-4B use the
`Qwen/Qwen3-4B` throughput entry. For 8B or 14B models, explicitly pass the
matching `--throughput-model` during evaluation.

## 3. Prepare data

Build the training data and core evaluation datasets:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/build_data.sh
```

This downloads the configured sources, screens problem difficulty, synthesizes
SFT examples through OpenRouter, and builds the RL dataset. It requires a local
GPU and `OPENROUTER_API_KEY`.

The main outputs are:

| Path | Contents |
| --- | --- |
| `data/processed/sft.parquet` | SFT examples with supervision masks. |
| `data/processed/train_rl.jsonl` | Shared RL training problems for FlyBy, ForkingRL, and Search-R1. |
| `data/eval/` | Evaluation questions. |

Dataset sources and splits are configured in `configs/data.yaml`. Generation,
sampling, and API budgets are configured in `configs/data_generation.yaml`.

To evaluate existing models without generating training data, prepare only the
core evaluation datasets:

```bash
scripts/build_data.sh --split eval
```

`scripts/eval.sh` prepares the remaining benchmark inputs, including the June
ArXivMath release, MedXpertQA, MMLU-Pro, and ChemBench, when needed.

## 4. Train

The commands below use one GPU by default. All three RL recipes run for 80 steps
and save checkpoints at steps 40 and 80. RL scripts accept additional Hydra
arguments at the end of the command, and `CFG=path/to/config.yaml` selects a
custom recipe.

### FlyBy: SFT followed by RL

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh
CUDA_VISIBLE_DEVICES=0 scripts/train_rl.sh
```

SFT uses `configs/sft.yaml` and saves to `outputs/models/sft`. RL starts from that
checkpoint, uses `configs/rl.yaml`, and saves to `outputs/models/rl`.
FlyBy RL starts and stops its OpenRouter tool server automatically. It requires
the throughput JSON from step 2 and an API key.

### ForkingRL

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_forking_rl.sh
```

The recipe is `configs/forking_rl.yaml`. The final checkpoint is:

```text
outputs/models/baselines/forking_rl/global_step_80/actor/huggingface
```

### Search-R1

Download and prepare the retrieval assets:

```bash
python -m src.search_r1.setup_wiki18
```

Start the retriever in a separate terminal with the same environment activated,
and leave it running during training:

```bash
scripts/run_search_r1_retriever.sh --device cpu
```

The default endpoint is `http://127.0.0.1:8000/retrieve`. Train in another terminal:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_search_r1.sh
```

The recipe is `configs/search_r1.yaml`. The training script checks the retriever
and starts its tool bridge automatically. The final checkpoint is:

```text
outputs/models/baselines/search_r1/global_step_80/actor/huggingface
```

Search-R1 training and evaluation use the retrieval server and do not require
OpenRouter calls. Configure a custom endpoint in both `configs/search_r1.yaml`
and the `search_r1` block of `configs/eval.yaml`.

## 5. Evaluate with `scripts/eval.sh`

The evaluation entry point runs 16 rollouts per problem on the fixed hard subset
of ArXivMath, GPQA-Diamond, SuperGPQA, ChemBench, MedXpertQA, and MMLU-Pro.
The subset contains 1,158 problems that Qwen3-4B answered correctly at most
4 times in 16 selection rollouts. Its IDs are stored in
`configs/main_hard_pass8.json`.

The headline metric is estimated pass@8, averaged equally across the six
benchmarks. Evaluation uses temperature 0.6, top-p 0.95, and top-k 20.
Local models have a budget of 16,384 generated tokens; FlyBy and Search-R1 allow
up to 4 tool calls per trajectory.

### FlyBy, SFT-only, and Prompt Only

```bash
scripts/eval.sh --mode flyby \
  --model outputs/models/rl/global_step_80/actor/huggingface \
  --throughput-model Qwen/Qwen3-4B \
  --gpus 0,1 --num-gpus 2 --run-id flyby-4b
```

Use the same mode with the following model and a new run ID for each comparison:

| Experiment | `--model` | Example `--run-id` |
| --- | --- | --- |
| SFT-only | `outputs/models/sft` | `flyby-4b-sft` |
| Prompt Only | `Qwen/Qwen3-4B` | `prompt-only-4b` |

All three use the tool interface and require `OPENROUTER_API_KEY`.

### DeepSeek-V4-Pro

Set `OPENROUTER_API_KEY` in `.env`, prepare the evaluation data, and run:

```bash
scripts/build_data.sh --split eval
scripts/eval.sh --mode deepseek --run-id deepseek-v4-pro-thinking
```

`--mode deepseek` automatically selects `deepseek/deepseek-v4-pro` and sends the full problem directly to the
API model, with 16 rollouts per hard problem and a 16,384-token completion budget.
It requires no local model, GPU selection,
or throughput measurement. Costs include API input and completion tokens, with
zero local GPU cost.

The backend and prices come from `tool.depths.3` in `configs/eval.yaml`.

### Vanilla Qwen and ForkingRL

```bash
scripts/eval.sh --mode vanilla \
  --model Qwen/Qwen3-4B \
  --throughput-model Qwen/Qwen3-4B \
  --gpus 0 --num-gpus 1 --run-id qwen3-4b

scripts/eval.sh --mode vanilla \
  --model outputs/models/baselines/forking_rl/global_step_80/actor/huggingface \
  --throughput-model Qwen/Qwen3-4B \
  --gpus 0 --num-gpus 1 --run-id forking-rl-4b
```

For a larger model, specify its matching throughput entry:

```bash
scripts/eval.sh --mode vanilla \
  --model Qwen/Qwen3-14B \
  --throughput-model Qwen/Qwen3-14B \
  --gpus 0 --num-gpus 1 --run-id qwen3-14b
```

For Qwen3-8B or FlyBy-8B checkpoints, use `--throughput-model Qwen/Qwen3-8B`.
The default is always Qwen3-4B, so changing `--model` alone does not change the
cost reference.

### Search-R1

```bash
scripts/eval.sh --mode search_r1 \
  --model outputs/models/baselines/search_r1/global_step_80/actor/huggingface \
  --throughput-model Qwen/Qwen3-4B \
  --gpus 0,1 --num-gpus 2 --run-id search-r1-4b
```

Evaluation reuses a running retriever. With the default configuration, if none
is running, it prepares the Wiki-18 assets, starts a CPU retriever, and stops
that retriever after evaluation. Install the retrieval dependencies first.
Reported Search-R1 monetary cost includes local model inference; retrieval
server cost is not included.

### Options, restarting, and results

| Option | Meaning |
| --- | --- |
| `--model` | Hugging Face model ID or local Hugging Face checkpoint directory. Optional for `deepseek`. |
| `--mode` | `flyby` / `ours`, `vanilla`, `search_r1`, or `deepseek` (Thinking enabled). |
| `--gpus 0,1` | Candidate GPUs for local-model evaluation. The scheduler waits for available GPUs. Unused by `deepseek`. |
| `--num-gpus 2` | Maximum concurrent evaluation workers. Each local worker loads the model on one GPU; benchmarks are distributed across workers. `deepseek` uses one API worker. |
| `--throughput-model` | Base-model key in `outputs/bench/throughput.json` for local cost accounting. |
| `--run-id NAME` | Output directory name. Reuse it with the same settings to restart an interrupted evaluation. |
| `--tool-budget-usd-per-worker 1000` | FlyBy or DeepSeek API budget per evaluation worker; defaults to 1,000 USD. |
| `--prepare-only` | Prepare benchmark inputs and run configuration, then exit before inference. |
| `--dry-run` | Prepare inputs, check model/cost prerequisites, and print worker commands without inference. |
| `--no-cost` | Evaluate accuracy without throughput or API cost reporting. External tool calls and DeepSeek generations still use the API. |
| `--aggregate-only` | Recompute metrics from a completed run; provide `--run-id`. |

Rerun the same evaluation command to restart: complete benchmark outputs are
reused, while incomplete benchmarks are rerun. Keep the model, mode, worker
count, cost options, and run ID unchanged. Use a new run ID for a new experiment.

Results are written under `outputs/evals/RUN_ID/`:

- `results.json` and `results.csv`: benchmark metrics, macro/micro averages, and costs.

The headline value in `results.json` is `hard.macro_6.estimated_pass_at_8`.

## Configuration reference

| File | Settings |
| --- | --- |
| `configs/data.yaml` | Training sources, held-out splits, and core evaluation datasets. |
| `configs/data_generation.yaml` | Difficulty screening, SFT synthesis, and API budgets. |
| `configs/sft.yaml` | FlyBy SFT. |
| `configs/rl.yaml` | FlyBy RL, tool backends, and cost-aware reward. |
| `configs/forking_rl.yaml` | ForkingRL training. |
| `configs/search_r1.yaml` | Search-R1 training and retrieval endpoint. |
| `configs/eval.yaml` | Evaluation sampling, generation budgets, tool backends, retrieval, and prices. |
| `configs/arxivmath.yaml`, `configs/medxpertqa.yaml`, `configs/mmlu_pro.yaml`, `configs/chembench.yaml` | Additional benchmark preparation. |
# FlyBy
