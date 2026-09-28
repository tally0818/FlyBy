'Keep a small visible CUDA allocation alive while an eval pipeline owns a GPU.'
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    if args.memory_mib <= 0:
        raise ValueError("--memory-mib must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the GPU reservation process")
    torch.cuda.set_device(0)
    allocation = torch.empty(
        args.memory_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    allocation.zero_()
    torch.cuda.synchronize()
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "memory_mib": args.memory_mib,
                "torch_device": torch.cuda.get_device_name(0),
            }
        ),
        encoding="utf-8",
    )

    while allocation is not None:
        time.sleep(60)


if __name__ == "__main__":
    main()
