from __future__ import annotations

import re
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


SYSTEM_PROMPT = """You are a precise question answering assistant for HotpotQA multi-hop reading.
Answer ONLY using the provided context paragraphs.
- If the answer is found in the context, extract the exact shortest answer span from the text.
- For yes/no questions, output ONLY "yes" or "no", nothing else.
- For names, numbers, dates and entities, copy them exactly as they appear in the context.
- Output only the final answer: no full sentences, no explanations, no quotes, no "Answer:" prefix, no trailing punctuation.
- Do NOT repeat yourself. Answer directly."""


def clean_answer(raw: str) -> str:
    """Normalize a (possibly reasoning-model) LLM output into a short answer span.

    GLM-Z1 style models emit <think> traces, restate the answer over several
    lines, and wrap it in "Answer:" / full sentences. HotpotQA is scored by
    exact-span EM/F1, so we strip that scaffolding down to the final span.
    """
    text = raw or ""
    # Drop chain-of-thought and stray tag markers.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"</?think>|</?answer>", " ", text, flags=re.I)

    # Reasoning models usually restate the final answer on the last line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]

    # Strip a leading "Answer:" / "The answer is" style prefix.
    text = re.sub(r"^(the\s+)?(final\s+)?answer\s*(is)?\s*[:\-]?\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()

    # Collapse an explicit yes/no verdict to the bare token.
    verdict = re.match(r"^(yes|no)\b", text, flags=re.I)
    if verdict:
        return verdict.group(1).lower()

    # Keep only the first sentence for long-winded answers.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        text = sentences[0]

    return text.rstrip(".").strip()


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
        answer = clean_answer(answer)
        return RAGResult(question=question, answer=answer, contexts=contexts)
