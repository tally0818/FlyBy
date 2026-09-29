# FlyBy

Official code for [Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge](https://arxiv.org/abs/2609.34327).

FlyBy trains small reasoning models to selectively query stronger external models when local reasoning is insufficient.

RL training is built on [Verl-Tool](https://github.com/TIGER-AI-Lab/verl-tool).

## Setup

Requirements: Linux, Python 3.10, CUDA 12.8, NVIDIA GPU.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip sync requirements.lock

uv pip install packaging psutil ninja
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

cp .env.example .env
```

Configure `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Data synthesis, FlyBy training/evaluation, DeepSeek evaluation |
| `HF_TOKEN` | Hugging Face authentication |
| `WANDB_ENABLED` | Enable Weights & Biases |
| `WANDB_API_KEY` | Weights & Biases authentication |

For Search-R1:

```bash
uv pip install -r requirements-retriever.txt
```

## Throughput

```bash
CUDA_VISIBLE_DEVICES=0 scripts/bench_throughput.sh --models Qwen/Qwen3-4B
```

Results are stored in `outputs/bench/throughput.json`.

The paper uses an H200 at USD 3.49/hour. GPU pricing is configured in `configs/rl.yaml` and `configs/eval.yaml`.

## Data

```bash
CUDA_VISIBLE_DEVICES=0 scripts/build_data.sh
```

Outputs:

| Path | Contents |
| --- | --- |
| `data/processed/sft.parquet` | SFT data |
| `data/processed/train_rl.jsonl` | RL data |
| `data/eval/` | Evaluation data |

Evaluation data only:

```bash
scripts/build_data.sh --split eval
```

Data configuration is in `configs/data.yaml` and `configs/data_generation.yaml`.

## Training

### FlyBy

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh
CUDA_VISIBLE_DEVICES=0 scripts/train_rl.sh
```

Configs: `configs/sft.yaml`, `configs/rl.yaml`.

### [ForkingRL](https://arxiv.org/abs/2506.01939)

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_forking_rl.sh
```

Config: `configs/forking_rl.yaml`.

### [Search-R1](https://arxiv.org/abs/2503.09516)

```bash
python -m src.search_r1.setup_wiki18
scripts/run_search_r1_retriever.sh --device cpu
CUDA_VISIBLE_DEVICES=0 scripts/train_search_r1.sh
```

Config: `configs/search_r1.yaml`.

## Evaluation

Evaluation is handled by `scripts/eval.sh`. For example:

```bash
scripts/eval.sh --mode flyby \
  --model outputs/models/rl/global_step_80/actor/huggingface \
  --throughput-model Qwen/Qwen3-4B \
  --gpus 0,1 --num-gpus 2 \
  --run-id flyby-4b
```

Available modes:

| Mode | Description |
| --- | --- |
| `flyby` / `ours` | FlyBy, SFT-only, and prompt-only models |
| `vanilla` | Vanilla Qwen and ForkingRL |
| `search_r1` | Search-R1 |
| `deepseek` | DeepSeek-V4-Pro |

The main evaluation uses 16 rollouts on the fixed 1,158-problem hard subset. The primary metric is macro-average estimated pass@8.

Results are stored in `outputs/evals/RUN_ID/`.

See `scripts/eval.sh --help` for additional options.

## Configuration

| File | Purpose |
| --- | --- |
| `configs/data.yaml` | Data sources and splits |
| `configs/data_generation.yaml` | Data generation |
| `configs/sft.yaml` | SFT |
| `configs/rl.yaml` | FlyBy RL |
| `configs/forking_rl.yaml` | ForkingRL |
| `configs/search_r1.yaml` | Search-R1 |
| `configs/eval.yaml` | Evaluation |

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{lee2026knowing,
  title={Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge},
  author={Lee, Chanuk and Kang, Minki and Park, Sangwoo and Yeo, Woongyeong and Baek, Jinheon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2609.34327},
  year={2026}
}
```
