'Run bookkeeping: run ids, seeding, .env loading, wandb init (all optional deps guarded).'
from __future__ import annotations

import datetime
import os
import random
import re
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    'Minimal .env loader (KEY=VALUE lines); does not override existing env.'
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", prefix)
    return f"{slug}_{stamp}"


def set_seed(seed: int) -> None:
    'Seed process-local RNGs used by training and evaluation.'
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass



seed_all = set_seed


def init_wandb(project: str, run_id: str, config: dict | None = None):
    'Returns the wandb run or None if wandb is unavailable/disabled.'
    if os.environ.get("WANDB_ENABLED", "false").lower() not in ("1", "true", "yes"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(project=project, name=run_id, config=config or {}, resume="allow")
