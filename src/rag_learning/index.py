from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag_learning.data import Document


_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tokenize_corpus(documents: list, n_workers: int = 1) -> list[list[str]]:
    """Tokenize every document's ``"<title> <text>"`` once, optionally in parallel.

    For corpora up to ~10k paragraphs, single-process is fastest (the
    multiprocessing IPC overhead dwarfs the regex work). For larger corpora,
    parallel tokenization scales roughly linearly with cores because the
    regex holds the GIL — threads would not help.

    Uses ``ProcessPoolExecutor`` with a small ``chunksize`` (4 chunks per
    worker) to amortize IPC. The function is picklable, so spawn-based
    platforms (Windows / macOS 3.8+) work too.
    """
    if n_workers is None or n_workers <= 1 or len(documents) < 1000:
        return [tokenize(f"{doc.title} {doc.text}") for doc in documents]
    title_texts = [f"{doc.title} {doc.text}" for doc in documents]
    # ~4 chunks per worker keeps IPC overhead low while still spreading work.
    chunksize = max(1, len(documents) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(tokenize, title_texts, chunksize=chunksize))


# ---------------------------------------------------------------------------
# BM25 (unchanged from the original implementation)
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, documents: list[Document], *, n_workers: int = 1):
        if not documents:
            raise ValueError("Cannot build an index without documents.")
        self.documents = documents
        # Tokenization is GIL-bound (regex), so processes > threads. For
        # typical HotpotQA-scale corpora (<10k paragraphs) single-process
        # is fastest; opt into multi-process for larger corpora via
        # ``n_workers``. The bm25 stats computation in rank_bm25 is a
        # single C-level pass over the tokenized corpus — not worth
        # parallelizing.
        self.tokenized_corpus = tokenize_corpus(documents, n_workers=n_workers)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(self.documents[index], float(scores[index])) for index in ranked_indices]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(doc) for doc in self.documents]
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)
        documents = [Document(**item) for item in payload]
        return cls(documents)


# ---------------------------------------------------------------------------
# Dense (cosine over pre-computed L2-normalized embeddings)
# ---------------------------------------------------------------------------

class DenseIndex:
    """Brute-force cosine-similarity dense retriever.

    Stores pre-computed, L2-normalized embeddings in a (N, dim) float32 array.
    `search` returns top-k docs by cosine similarity against a query vector.
    Intended for small-to-medium corpora (~10k docs) where FAISS overhead is
    not justified. For 10k docs of dim 768, a single search takes < 5 ms.
    """

    def __init__(self, documents: list[Document], embeddings: np.ndarray):
        if not documents:
            raise ValueError("Cannot build a dense index without documents.")
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D (N, dim), got shape {embeddings.shape}"
            )
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"documents ({len(documents)}) and embeddings "
                f"({embeddings.shape[0]}) must agree on N"
            )
        self.documents = documents
        self.embeddings = embeddings.astype(np.float32, copy=False)
        # Defensive L2-normalize (Embedder already returns normalized vectors,
        # but a re-normalize here makes the class self-contained).
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def search(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        if query_vec.ndim != 1:
            query_vec = query_vec.reshape(-1)
        q = query_vec.astype(np.float32, copy=False)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [(doc, 0.0) for doc in self.documents[:top_k]]
        q = q / q_norm
        scores = self.embeddings @ q  # (N,)
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in ranked_indices]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [asdict(doc) for doc in self.documents],
            "embeddings": self.embeddings,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)
        documents = [Document(**item) for item in payload["documents"]]
        return cls(documents, payload["embeddings"])


# ---------------------------------------------------------------------------
# Hybrid (BM25 + Dense via Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf(rank_lists: list[list[int]], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion.

    For each doc index d, score(d) = sum_i 1 / (k + rank_i(d)),
    where rank_i(d) is d's 0-indexed position in the i-th ranked list
    (or `len(rank_list)` if d is absent from that list).

    k=60 follows the original Cormack et al. (2009) choice and is broadly
    robust across retrieval tasks. We don't need to tune it.
    """
    fused: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, doc_idx in enumerate(rank_list):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    if not fused:
        return []
    n_docs = max(fused.keys()) + 1
    scores = [0.0] * n_docs
    for doc_idx, s in fused.items():
        scores[doc_idx] = s
    return scores


class HybridIndex:
    """BM25 + Dense retriever with RRF fusion.

    Wraps a `BM25Index` and a `DenseIndex` over the same document set. The
    `search` method runs both retrievers, takes the top `top_k * OVERFETCH`
    docs from each, fuses their ranks with RRF, and returns the top-`top_k`
    by fused score. This avoids running dense search over the entire corpus
    (cheap for 10k docs but a habit worth keeping).

    For the BM25-only path, set `query_vec=None` — the dense branch is
    skipped. For dense-only, pass `query=None` — BM25 is skipped.
    """

    OVERFETCH = 4  # take 4x more candidates from each branch than top_k

    def __init__(self, bm25: BM25Index, dense: DenseIndex):
        if len(bm25.documents) != len(dense.documents):
            raise ValueError(
                "BM25Index and DenseIndex must cover the same documents "
                f"(got {len(bm25.documents)} vs {len(dense.documents)})"
            )
        self.bm25 = bm25
        self.dense = dense
        self.documents = bm25.documents  # alias for callers that expect .documents
        # O(1) doc-id -> list-index map. Replaces the previous O(n) list.index
        # scan that dominated `search` cost on 78万-doc TriviaQA corpora.
        # `setdefault` keeps the FIRST occurrence (matching list.index() semantics)
        # — TriviaQA's `entity_pages` puts the same paragraph under multiple
        # entities, so 11万 docs have duplicate ids; using `d[i] = i` would jump
        # to the LAST duplicate and silently drop / reorder the result.
        self._doc_id_to_idx: dict[str, int] = {}
        for i, doc in enumerate(self.documents):
            self._doc_id_to_idx.setdefault(doc.id, i)

    def search(
        self,
        query: str | None,
        query_vec: np.ndarray | None,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        # Determine which branches to run.
        run_bm25 = query is not None
        run_dense = query_vec is not None
        if not (run_bm25 or run_dense):
            raise ValueError("At least one of `query` or `query_vec` must be set.")

        candidate_k = min(top_k * self.OVERFETCH, len(self.documents))
        id2idx = self._doc_id_to_idx
        bm25_ranked: list[int] = []
        if run_bm25:
            bm25_hits = self.bm25.search(query, top_k=candidate_k)
            bm25_ranked = [id2idx[d.id] for d, _ in bm25_hits]

        dense_ranked: list[int] = []
        if run_dense:
            dense_hits = self.dense.search(query_vec, top_k=candidate_k)
            dense_ranked = [id2idx[d.id] for d, _ in dense_hits]

        # If only one branch ran, return its results directly (no fusion needed).
        if not run_bm25:
            return [(self.documents[i], 0.0) for i in dense_ranked[:top_k]]
        if not run_dense:
            return [(self.documents[i], 0.0) for i in bm25_ranked[:top_k]]

        # Both branches ran: RRF fusion.
        fused = rrf([bm25_ranked, dense_ranked])
        # Sort doc indices by fused score (desc).
        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        return [(self.documents[i], fused[i]) for i in order[:top_k]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse DenseIndex.save schema so the on-disk format is consistent.
        payload = {
            "documents": [asdict(doc) for doc in self.documents],
            "embeddings": self.dense.embeddings,
            # BM25 tokenized corpus + raw bm25 object pickled to bytes.
            "bm25_tokenized_corpus": self.bm25.tokenized_corpus,
            "bm25_object_bytes": pickle.dumps(self.bm25.bm25),
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)
        documents = [Document(**item) for item in payload["documents"]]
        dense = DenseIndex(documents, payload["embeddings"])
        # Reconstruct BM25Index from saved tokenized corpus + bm25 object.
        bm25 = BM25Index.__new__(BM25Index)
        bm25.documents = documents
        bm25.tokenized_corpus = payload["bm25_tokenized_corpus"]
        bm25.bm25 = pickle.loads(payload["bm25_object_bytes"])
        return cls(bm25, dense)


# ---------------------------------------------------------------------------
# Auto-detect loader: tries HybridIndex, then DenseIndex, then BM25Index
# ---------------------------------------------------------------------------

def load_any(path: str | Path):
    """Load an index file, returning a BM25Index, DenseIndex, or HybridIndex
    depending on what's inside the pickle. Use this when the caller doesn't
    care which one (e.g. legacy scripts that only need `.search(query, top_k)`).
    """
    path = Path(path)
    with path.open("rb") as file:
        payload = pickle.load(file)
    # HybridIndex.save schema has "embeddings" + "bm25_tokenized_corpus".
    if isinstance(payload, dict) and "bm25_tokenized_corpus" in payload:
        return HybridIndex.load(path)
    # DenseIndex.save schema: dict with "embeddings" and "documents".
    if isinstance(payload, dict) and "embeddings" in payload:
        return DenseIndex.load(path)
    # BM25Index.save schema: list of document dicts.
    documents = [Document(**item) for item in payload]
    return BM25Index(documents)
