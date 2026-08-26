from __future__ import annotations

import hashlib

import httpx
import numpy as np

from rag_learning import config


def _is_local(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


class Embedder:
    """Text embedding via a local Ollama embedding model.

    Used by the State module (framework module 1) to compute semantic metrics
    (Ua, Ue, delta_I). This is separate from the chat LLM (llm.ChatLLM): the
    chat model answers questions, the embedder only turns text into vectors.

    Embeddings are cached in-memory keyed by (model, text) so repeated
    paragraphs / questions during training and retrieval are only embedded once.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.EMBED_MODEL
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        # Bypass any VPN/system proxy for local calls (same reason as ChatLLM).
        trust_env = not _is_local(self.base_url)
        self.client = httpx.Client(base_url=self.base_url, trust_env=trust_env, timeout=60.0)
        self._cache: dict[str, np.ndarray] = {}

    def _key(self, text: str) -> str:
        # `errors="replace"` so lone UTF-16 surrogates (occasionally present in
        # TriviaQA's `wiki_context`) don't crash the encoder. The hash is
        # only used as a cache key, so a lossy-but-stable encoding is fine.
        return hashlib.sha1(
            f"{self.model}\x00{text}".encode("utf-8", errors="replace")
        ).hexdigest()

    def _sanitize(self, text: str) -> str:
        """Replace unpaired UTF-16 surrogates with U+FFFD so the text is
        encodable as UTF-8.

        TriviaQA's `wiki_context` occasionally contains lone surrogates
        (e.g. U+DC80) that Python's default utf-8 encoder rejects with
        `UnicodeEncodeError: surrogates not allowed`. Encoding with
        errors='replace' turns them into U+FFFD ('?'), which is harmless
        for both embedding and JSON serialization over /api/embed.
        """
        return text.encode("utf-8", errors="replace").decode("utf-8")

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings.

        Accepts a single string or a list; always returns a 2D array so callers
        can treat the single-text case uniformly (shape (1, dim)).
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        # Sanitize before hashing / sending: some TriviaQA paragraphs contain
        # lone UTF-16 surrogates that would otherwise crash `_key`'s encoder
        # and httpx's JSON serializer. `_sanitize` is idempotent on clean text.
        items = [self._sanitize(t) for t in items]
        if not items:
            return np.empty((0, 0), dtype=np.float32)

        # Resolve from cache; only send the misses to Ollama.
        missing = [t for t in items if self._key(t) not in self._cache]
        if missing:
            response = self.client.post(
                "/api/embed", json={"model": self.model, "input": missing}
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if not vectors:
                raise RuntimeError(f"Ollama returned no embeddings for model {self.model!r}.")
            for text, vec in zip(missing, vectors):
                arr = np.asarray(vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                self._cache[self._key(text)] = arr

        return np.stack([self._cache[self._key(t)] for t in items])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors (0 if either is zero)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """(n, n) cosine-similarity matrix for row vectors in `matrix`.

    Assumes rows may not be normalized; normalizes defensively.
    """
    if matrix.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return unit @ unit.T
