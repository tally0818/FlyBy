'FastAPI server for the canonical Search-R1 Wiki-18/E5 flat index.'
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Any


@dataclass
class ServerConfig:
    index_path: Path
    corpus_path: Path
    model: str = "intfloat/e5-base-v2"
    model_revision: str = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
    topk: int = 3
    device: str = "cpu"
    faiss_gpu: bool = False
    query_max_length: int = 256
    batch_size: int = 128
    fp16: bool = False


class Wiki18Retriever:
    def __init__(self, config: ServerConfig):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is required in the retriever environment; install faiss-cpu "
                "or the CUDA-matched faiss-gpu package"
            ) from exc
        import datasets

        self.faiss = faiss
        self.config = config
        self.index = faiss.read_index(str(config.index_path))
        if config.faiss_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("--faiss-gpu requested but CUDA is unavailable")
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=options)
        self.corpus = datasets.load_dataset(
            "json", data_files=str(config.corpus_path), split="train", num_proc=4
        )
        self.device = torch.device(config.device)
        dtype = torch.float16 if config.fp16 and self.device.type == "cuda" else None
        self.model = AutoModel.from_pretrained(
            config.model,
            revision=config.model_revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.model_revision,
            use_fast=True,
            trust_remote_code=True,
        )

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        prefixed = [f"query: {query}" for query in queries]
        inputs = self.tokenizer(
            prefixed,
            max_length=self.config.query_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        mask = inputs["attention_mask"].unsqueeze(-1).bool()
        hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
        embeddings = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, order="C")

    def batch_search(self, queries: list[str], topk: int) -> tuple[list[list[dict]], list[list[float]]]:
        all_documents: list[list[dict]] = []
        all_scores: list[list[float]] = []
        for start in range(0, len(queries), self.config.batch_size):
            batch = queries[start : start + self.config.batch_size]
            scores, indices = self.index.search(self.encode(batch), k=topk)
            for row_indices, row_scores in zip(indices.tolist(), scores.tolist()):
                documents = [self.corpus[int(index)] for index in row_indices if index >= 0]
                all_documents.append(documents)
                all_scores.append(row_scores[: len(documents)])
        return all_documents, all_scores


def create_app(retriever: Wiki18Retriever, default_topk: int) -> Any:
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="Search-R1 Wiki-18 Retriever")

    @app.get("/health")
    def health():
        return {"status": "ok", "corpus_size": len(retriever.corpus)}

    @app.post("/retrieve")
    def retrieve(request: dict[str, Any] = Body(...)):




        queries = request.get("queries")
        if not isinstance(queries, list) or not queries or not all(
            isinstance(query, str) and query.strip() for query in queries
        ):
            raise HTTPException(
                status_code=422,
                detail="queries must be a non-empty list of non-empty strings",
            )
        topk = request.get("topk") or default_topk
        if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
            raise HTTPException(status_code=422, detail="topk must be a positive integer")
        return_scores = bool(request.get("return_scores", False))
        documents, scores = retriever.batch_search(queries, topk)
        if return_scores:
            result = [
                [
                    {"document": document, "score": score}
                    for document, score in zip(row_documents, row_scores)
                ]
                for row_documents, row_scores in zip(documents, scores)
            ]
        else:
            result = documents
        return {"result": result}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, default=Path("data/search_r1/wiki18/e5_Flat.index"))
    parser.add_argument("--corpus-path", type=Path, default=Path("data/search_r1/wiki18/wiki-18.jsonl"))
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument(
        "--model-revision",
        default="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    for path in (args.index_path, args.corpus_path):
        if not path.is_file():
            raise SystemExit(f"missing retrieval asset: {path}; run python -m src.search_r1.setup_wiki18")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    config = ServerConfig(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model=args.model,
        model_revision=args.model_revision,
        topk=args.topk,
        device=device,
        faiss_gpu=args.faiss_gpu,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
    retriever = Wiki18Retriever(config)
    app = create_app(retriever, args.topk)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
