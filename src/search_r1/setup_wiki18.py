'Download and materialize the canonical Search-R1 Wiki-18/E5 assets.'
from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

INDEX_REPO = "PeterJinGo/wiki-18-e5-index"
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
INDEX_REVISION = "a4d31160a035f30764604f4827cd8f1d0315eb86"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"


def copy_stream(inputs: list[Path], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for source in inputs:
            with source.open("rb") as part:
                shutil.copyfileobj(part, destination, length=16 * 1024 * 1024)
    temporary.replace(output)


def decompress_gzip(source: Path, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(source, "rb") as compressed, temporary.open("wb") as destination:
        shutil.copyfileobj(compressed, destination, length=16 * 1024 * 1024)
    temporary.replace(output)


def materialize(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        Path(
            hf_hub_download(
                repo_id=INDEX_REPO,
                filename=name,
                repo_type="dataset",
                revision=INDEX_REVISION,
                local_dir=output_dir,
            )
        )
        for name in ("part_aa", "part_ab")
    ]
    compressed = Path(
        hf_hub_download(
            repo_id=CORPUS_REPO,
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=output_dir,
        )
    )
    index = output_dir / "e5_Flat.index"
    corpus = output_dir / "wiki-18.jsonl"
    if not index.exists():
        copy_stream(parts, index)
    if not corpus.exists():
        decompress_gzip(compressed, corpus)
    return index, corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/search_r1/wiki18"))
    args = parser.parse_args()
    index, corpus = materialize(args.output_dir)
    print(f"index:  {index} ({index.stat().st_size / 2**30:.2f} GiB)")
    print(f"corpus: {corpus} ({corpus.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
