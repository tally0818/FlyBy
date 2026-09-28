'Search-R1 protocol and client shared by training and standalone evaluation.'
from __future__ import annotations

import argparse
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"
INFORMATION_OPEN = "<information>"
INFORMATION_CLOSE = "</information>"
ACTION_STOP_TOKENS = [SEARCH_CLOSE]


@dataclass(frozen=True)
class SearchCall:
    query: str | None = None
    valid: bool = True
    error: str | None = None
    span: tuple[int, int] = (0, 0)


def parse_search_call(text: str) -> SearchCall | None:
    'Parse the last closed Search-R1 action, preserving malformed attempts.'
    close = text.rfind(SEARCH_CLOSE)
    if close < 0:
        return None
    end = close + len(SEARCH_CLOSE)
    opening = text.rfind(SEARCH_OPEN, 0, close)
    if opening < 0:
        return SearchCall(valid=False, error="missing <search> open tag", span=(0, end))
    start = opening
    query = text[opening + len(SEARCH_OPEN) : close].strip()
    if not query:
        return SearchCall(valid=False, error="search query must be non-empty", span=(start, end))
    if SEARCH_OPEN in query or SEARCH_CLOSE in query:
        return SearchCall(
            query=query,
            valid=False,
            error="search query must not contain nested search tags",
            span=(start, end),
        )
    return SearchCall(query=query, span=(start, end))


def has_search_attempt(text: str) -> bool:
    return SEARCH_OPEN in text or SEARCH_CLOSE in text


def has_unterminated_search_attempt(text: str) -> bool:
    return text.count(SEARCH_OPEN) > text.count(SEARCH_CLOSE)


def format_information(text: str) -> str:
    return f"\n\n{INFORMATION_OPEN}{text.strip()}{INFORMATION_CLOSE}\n\n"


def format_search_error(message: str) -> str:
    return format_information(f"Search unavailable: {message}")


def system_prompt(max_turns: int = 4, topk: int = 3) -> str:
    'Search-R1 policy prompt adapted only at the final-answer boundary.'
    return (
        "You are a careful reasoning assistant with access to a Wikipedia search engine.\n"
        "Reason step by step. When external knowledge is needed, emit exactly one search "
        "query as <search>query</search>. The system will append the top "
        f"{topk} retrieved Wikipedia passages inside <information> and </information>. "
        f"You may make at most {max_turns} searches. Do not fabricate <information> tags.\n"
        "When no more search is needed, finish with the final answer inside \\boxed{}. "
        "For multiple-choice questions, put only the answer letter inside the box."
    )


def _document_contents(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    document = item.get("document", item)
    if not isinstance(document, dict):
        return str(document)
    contents = document.get("contents")
    if contents:
        return str(contents)
    title = str(document.get("title", "")).strip()
    body = str(document.get("text", "")).strip()
    return "\n".join(part for part in (title, body) if part)


def passages_to_text(passages: list[Any]) -> str:
    'Render Search-R1 retrieval results in the upstream ``Doc N`` format.'
    rendered = []
    for index, item in enumerate(passages, start=1):
        contents = _document_contents(item).strip()
        if not contents:
            continue
        title, _, body = contents.partition("\n")
        rendered.append(f"Doc {index}(Title: {title.strip(chr(34))}) {body.strip()}".rstrip())
    return "\n".join(rendered)


def truncate_result(text: str, max_chars: int) -> str:
    'Bound injected retrieval text identically in training and evaluation.'
    if max_chars <= 0:
        raise ValueError("max_result_chars must be positive")
    if len(text) <= max_chars:
        return text
    marker = "\n...[retrieval truncated]...\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return text[:max_chars]
    left = remaining // 2
    return text[:left] + marker + text[-(remaining - left) :]


class SearchR1Client:
    'Thread-safe client for the separately hosted Wiki-18 retriever.'

    def __init__(
        self,
        url: str = "http://127.0.0.1:8000/retrieve",
        *,
        topk: int = 3,
        timeout_s: float = 30.0,
        concurrency: int = 16,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if topk <= 0:
            raise ValueError("topk must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self.url = url
        self.topk = int(topk)
        self.timeout_s = float(timeout_s)
        self._slots = threading.BoundedSemaphore(int(concurrency))
        self._client = httpx.Client(timeout=self.timeout_s, transport=transport)

    def search(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        with self._slots:
            response = self._client.post(
                self.url,
                json={"queries": [query], "topk": self.topk, "return_scores": True},
            )
            response.raise_for_status()
            payload = response.json()
        result = payload.get("result")
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
            raise ValueError("retriever response must contain one result list per query")
        passages = result[0]
        text = passages_to_text(passages)
        return {
            "text": text,
            "passages": passages,
            "num_docs": len(passages),
            "result_chars": len(text),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    async def asearch(self, query: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.search, query)

    def close(self) -> None:
        self._client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a Search-R1 retriever endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--query", default="artificial intelligence")
    args = parser.parse_args()
    client = SearchR1Client(args.url, topk=args.topk, timeout_s=args.timeout_s, concurrency=1)
    try:
        result = client.search(args.query)
    finally:
        client.close()
    if result["num_docs"] <= 0:
        raise SystemExit("retriever returned no documents")
    print(
        f"retriever ok: docs={result['num_docs']} chars={result['result_chars']} "
        f"latency_ms={result['latency_ms']:.1f}"
    )


if __name__ == "__main__":
    main()
