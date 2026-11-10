"""``JobKnowledge`` adapter: Vertex AI text embeddings and cosine similarity in memory.

The job information is small (a few dozen short pieces), so a vector database adds a
service without a benefit. The index is built once per process, on first use or at
``warm_up``. A multilingual embedding model lets a Spanish question match English text.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from loguru import logger

from screening_agent.application.ports import KnowledgeHit


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    text: str


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeChunk(id=item["id"], text=item["text"]) for item in data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rank(
    query: list[float],
    index: list[tuple[KnowledgeChunk, list[float]]],
    limit: int,
    min_score: float,
) -> list[KnowledgeHit]:
    """The best chunks for a query vector, best first, above ``min_score``."""
    scored = [(cosine(query, vector), chunk) for chunk, vector in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        KnowledgeHit(id=chunk.id, text=chunk.text, score=round(score, 3))
        for score, chunk in scored[:limit]
        if score >= min_score
    ]


class GeminiJobKnowledge:
    def __init__(
        self,
        client: genai.Client,
        model: str,
        chunks: list[KnowledgeChunk],
        min_score: float,
    ) -> None:
        self._client = client
        self._model = model
        self._chunks = chunks
        self._min_score = min_score
        self._index: list[tuple[KnowledgeChunk, list[float]]] | None = None
        self._lock = asyncio.Lock()

    async def warm_up(self) -> None:
        """Build the index before the first question, so the candidate does not wait."""
        await self._ensure_index()

    async def search(self, question: str, limit: int) -> list[KnowledgeHit]:
        index = await self._ensure_index()
        query = await self._embed(question, "RETRIEVAL_QUERY")
        hits = rank(query, index, limit, self._min_score)
        logger.info(f"Job search '{question}': {[(h.id, h.score) for h in hits]}")
        return hits

    async def _ensure_index(self) -> list[tuple[KnowledgeChunk, list[float]]]:
        async with self._lock:
            if self._index is None:
                vectors = await asyncio.gather(
                    *(self._embed(chunk.text, "RETRIEVAL_DOCUMENT") for chunk in self._chunks)
                )
                self._index = list(zip(self._chunks, vectors, strict=True))
                logger.info(f"Job knowledge index built: {len(self._index)} pieces")
            return self._index

    async def _embed(self, text: str, task_type: str) -> list[float]:
        # One text per request: gemini-embedding-001 on Vertex AI takes a single input.
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("The embedding response has no values.")
        return list(response.embeddings[0].values)
