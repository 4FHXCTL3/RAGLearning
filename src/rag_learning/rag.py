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


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def align_to_context(answer: str, contexts: list["RetrievedContext"], threshold: float = 0.5) -> str:
    """Snap a paraphrased answer onto the exact span it matches in the context.

    HotpotQA is scored by exact-span EM/F1, so a correct-but-reworded answer
    (extra words, reordering, spelling variants) loses points. We slide a window
    over each retrieved paragraph and, if a window's token Jaccard overlap with
    the answer clears `threshold`, return that window's *original* substring so
    it matches the gold span verbatim. yes/no and empty answers are left as-is.
    """
    ans = (answer or "").strip()
    if not ans or ans.lower() in {"yes", "no", "unknown"}:
        return ans

    ans_tokens = [m.group(0).lower() for m in _WORD_RE.finditer(ans)]
    if not ans_tokens:
        return ans
    ans_set = set(ans_tokens)
    n = len(ans_tokens)

    best_score = 0.0
    best_span = None
    for ctx in contexts:
        text = ctx.text or ""
        matches = list(_WORD_RE.finditer(text))
        if not matches:
            continue
        # Try window sizes around the answer length to tolerate minor length diffs.
        for win in {max(1, n - 1), n, n + 1}:
            for start in range(0, max(1, len(matches) - win + 1)):
                window = matches[start:start + win]
                if not window:
                    continue
                win_set = {m.group(0).lower() for m in window}
                inter = len(ans_set & win_set)
                union = len(ans_set | win_set)
                score = inter / union if union else 0.0
                if score > best_score:
                    best_score = score
                    best_span = text[window[0].start():window[-1].end()]

    if best_span is not None and best_score >= threshold:
        return best_span.strip()
    return ans


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
        answer = align_to_context(answer, contexts)
        return RAGResult(question=question, answer=answer, contexts=contexts)
