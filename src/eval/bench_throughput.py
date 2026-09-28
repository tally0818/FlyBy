'Measure per-model H200 vLLM prefill/decode throughput.'
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BENCH_PROMPT = (
    "Solve step by step: Find the number of ordered pairs of integers (a, b) with "
    "1 <= a, b <= 100 such that a*b is divisible by a + b. Think carefully and at length."
)


def bench_model(model: str, n_prompts: int, prefill_tokens: int,
                max_tokens: int, gpu_mem: float) -> dict[str, float]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    short_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": BENCH_PROMPT}], tokenize=False, add_generation_prompt=True
    )
    repeats = max(1, prefill_tokens // max(1, len(tokenizer.encode(BENCH_PROMPT))))
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem)
    llm.generate([short_prompt] * 2, SamplingParams(temperature=0.7, max_tokens=64))


    inputs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Benchmark sample {i}.\n" + (BENCH_PROMPT + "\n") * repeats}],
        tokenize=False, add_generation_prompt=True,
    ) for i in range(n_prompts)]
    input_tokens = sum(len(tokenizer.encode(p)) for p in inputs)
    start = time.time()
    llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=1))
    prefill_elapsed = time.time() - start

    start = time.time()
    decode_inputs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Benchmark sample {i}.\n{BENCH_PROMPT}"}],
        tokenize=False, add_generation_prompt=True,
    ) for i in range(n_prompts)]
    outputs = llm.generate(decode_inputs,
                           SamplingParams(temperature=0.7, max_tokens=max_tokens))
    decode_elapsed = time.time() - start
    out_tokens = sum(len(o.token_ids) for out in outputs for o in out.outputs)
    return {
        "prefill_tok_s": input_tokens / prefill_elapsed,
        "decode_tok_s": out_tokens / decode_elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
    ])
    parser.add_argument("--n-prompts", type=int, default=32,
                        help="parallel sequences; matches serving-style batching")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", default="outputs/bench/throughput.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}


    results = {k: v for k, v in existing.items() if isinstance(v, dict)}
    for model in args.models:
        rates = bench_model(model, args.n_prompts, args.prefill_tokens,
                            args.max_tokens, args.gpu_memory_utilization)
        results[model] = {k: round(v, 2) for k, v in rates.items()}
        print(f"{model}: prefill={rates['prefill_tok_s']:.1f} tok/s, "
              f"decode={rates['decode_tok_s']:.1f} tok/s")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
