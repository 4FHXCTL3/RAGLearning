from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi

from rag_learning.data import Document


_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("Cannot build an index without documents.")
        self.documents = documents
        self.tokenized_corpus = [tokenize(f"{doc.title} {doc.text}") for doc in documents]
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
