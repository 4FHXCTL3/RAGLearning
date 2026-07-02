from __future__ import annotations

from dataclasses import dataclass

from rag_learning.index import BM25Index
from rag_learning.llm import ChatLLM


@dataclass(frozen=True)
class RetrievedContext:
    id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]


SYSTEM_PROMPT = """You are a question answering assistant.
Answer the question based on the given context. Keep the answer short."""


def build_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    context_text = "\n\n".join(
        f"[{index}] Title: {ctx.title}\nContent: {ctx.text}"
        for index, ctx in enumerate(contexts, start=1)
    )
    return f"""Context:
{context_text}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, index: BM25Index, llm: ChatLLM):
        self.index = index
        self.llm = llm

    def answer(self, question: str, top_k: int = 5) -> RAGResult:
        contexts = [
            RetrievedContext(id=doc.id, title=doc.title, text=doc.text, score=score)
            for doc, score in self.index.search(question, top_k=top_k)
        ]
        prompt = build_prompt(question, contexts)
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT).strip()
        return RAGResult(question=question, answer=answer, contexts=contexts)
