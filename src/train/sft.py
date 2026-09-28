'Train the SFT model with segment-level loss masks.'
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from ..common.config import load_config
from ..common.run import set_seed, init_wandb, new_run_id
from ..tools import protocol


def _is_query_segment(text: str) -> bool:
    'Recognize the attribute-bearing opening tag used by protocol v2.'
    return str(text).lstrip().startswith(protocol.QUERY_OPEN_PREFIX)


def weighted_causal_lm_loss(logits, labels, loss_weights):
    'Causal CE normalized by supervised-token count, not by weight mass.'
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].to(shift_logits.dtype).contiguous()
    valid = shift_labels.ne(-100)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels)
    denominator = valid.sum().clamp_min(1).to(token_loss.dtype)
    return (token_loss * shift_weights * valid).sum() / denominator


def _save_checkpoint(model, tokenizer, out_dir: Path, torch, save_bf16: bool) -> None:
    'Save an eval-ready checkpoint without casting live fp32 master weights.'
    out_dir.mkdir(parents=True, exist_ok=True)
    if not save_bf16:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        return




    state_dict = {
        name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for name, tensor in model.state_dict().items()
    }
    original_dtype = getattr(model.config, "dtype", None)
    model.save_pretrained(out_dir, state_dict=state_dict, safe_serialization=True)
    model.config.dtype = torch.bfloat16
    model.config.save_pretrained(out_dir)
    model.config.dtype = original_dtype or torch.float32
    tokenizer.save_pretrained(out_dir)
    del state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])

    from contextlib import nullcontext

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["model_revision"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"


    use_bf16_autocast = bool(tcfg.get("bf16")) and device == "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["model_revision"], torch_dtype=torch.float32
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    df = pd.read_parquet(cfg["data"])
    max_len = tcfg["max_seq_len"]

    role_names = sorted({
        str(segment.get("role") or (
            "query" if _is_query_segment(segment["text"]) else "continuation"
        ))
        for raw in df.get("segments_json", [])
        if isinstance(raw, str)
        for segment in json.loads(raw)
        if segment.get("train")
    })
    if not role_names:
        role_names = ["continuation", "query"]
    role_to_id = {name: index for index, name in enumerate(role_names)}

    class SpanDataset(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            row = df.iloc[i]
            if "segments_json" in row and isinstance(row["segments_json"], str):
                segments = json.loads(row["segments_json"])
            else:
                segments = [
                    {"text": row["prefix"], "train": False},
                    {"text": row["completion"], "train": True},
                ]
            input_ids, labels, loss_weights, role_ids = [], [], [], []
            for segment in segments:
                ids = tokenizer.encode(segment["text"], add_special_tokens=False)
                input_ids.extend(ids)

                if not segment["train"]:
                    labels.extend([-100] * len(ids))
                    loss_weights.extend([0.0] * len(ids))
                    role_ids.extend([-1] * len(ids))
                    continue

                role = str(segment.get("role") or (
                    "query" if _is_query_segment(segment["text"]) else "continuation"
                ))
                weight = float(segment.get("loss_weight", 1.0))
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"row {i} has invalid loss_weight={weight}")
                if role not in role_to_id:
                    raise ValueError(f"row {i} has unknown supervision role={role}")
                role_id = role_to_id[role]
                if role == "query" or _is_query_segment(segment["text"]):

                    labels.extend(ids)
                    loss_weights.extend([weight] * len(ids))
                    role_ids.extend([role_id] * len(ids))
                else:

                    cap = int(cfg["mask"].get("continuation_max_tokens", len(ids)))
                    supervised = min(cap, len(ids))
                    labels.extend(ids[:supervised])
                    labels.extend([-100] * (len(ids) - supervised))
                    loss_weights.extend([weight] * supervised)
                    loss_weights.extend([0.0] * (len(ids) - supervised))
                    role_ids.extend([role_id] * supervised)
                    role_ids.extend([-1] * (len(ids) - supervised))

            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
            loss_weights = loss_weights[-max_len:]
            role_ids = role_ids[-max_len:]
            if not any(x != -100 for x in labels):
                raise ValueError(f"row {i} has no supervised tokens after truncation")
            return {
                "input_ids": input_ids,
                "labels": labels,
                "loss_weights": loss_weights,
                "role_ids": role_ids,
            }

    def collate(batch):
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, weights, roles, attn = [], [], [], [], []
        for x in batch:
            n = width - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad] * n)
            labels.append(x["labels"] + [-100] * n)
            weights.append(x["loss_weights"] + [0.0] * n)
            roles.append(x["role_ids"] + [-1] * n)
            attn.append([1] * (width - n) + [0] * n)
        return (
            torch.tensor(input_ids),
            torch.tensor(labels),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(roles),
            torch.tensor(attn),
        )

    dataset = SpanDataset()
    supervision_tokens: Counter[str] = Counter()
    supervision_mass: defaultdict[str, float] = defaultdict(float)
    for index in range(len(dataset)):
        item = dataset[index]
        for role_id, weight, label in zip(
            item["role_ids"], item["loss_weights"], item["labels"]
        ):
            if label == -100:
                continue
            role = role_names[role_id]
            supervision_tokens[role] += 1
            supervision_mass[role] += float(weight)
    supervision_report = {
        "tokens": dict(supervision_tokens),
        "weighted_mass": {
            role: round(mass, 2) for role, mass in supervision_mass.items()
        },
    }
    print(f"supervision {json.dumps(supervision_report, sort_keys=True)}")

    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    steps = math.ceil(len(loader) * tcfg["epochs"] / tcfg["grad_accum"])
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])

    warmup = int(tcfg.get("warmup_steps", round(steps * float(tcfg.get("warmup_ratio", 0.0)))))
    sched = get_cosine_schedule_with_warmup(optim, min(warmup, max(steps - 1, 0)), steps)
    run = init_wandb("effireasoner", new_run_id("sft"), cfg)
    if run:
        run.log({
            **{f"supervision/tokens/{role}": count for role, count in supervision_tokens.items()},
            **{f"supervision/weighted_mass/{role}": mass for role, mass in supervision_mass.items()},
            "step": 0,
        })

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_autocast
        else nullcontext()
    )
    model.train()
    step = 0
    for epoch in range(tcfg["epochs"]):
        for i, (input_ids, labels, loss_weights, _role_ids, attn) in enumerate(loader):
            with autocast:
                out = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attn.to(device),
                )
                loss = weighted_causal_lm_loss(
                    out.logits,
                    labels.to(device),
                    loss_weights.to(device),
                )
            (loss / tcfg["grad_accum"]).backward()
            is_last = i == len(loader) - 1
            if (i + 1) % tcfg["grad_accum"] == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")
                    if run:
                        run.log({"loss": loss.item(), "step": step})

        if tcfg.get("save_each_epoch") and epoch + 1 < int(tcfg["epochs"]):
            _save_checkpoint(
                model,
                tokenizer,
                Path(cfg["out"]) / f"epoch_{epoch + 1}",
                torch,
                use_bf16_autocast,
            )

    out_dir = Path(cfg["out"])
    _save_checkpoint(model, tokenizer, out_dir, torch, use_bf16_autocast)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
