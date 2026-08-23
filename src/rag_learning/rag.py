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


# Notes on the prompt:
# - We deliberately do NOT add a "yes/no questions -> only yes or no" rule. llama3:8b
#   over-triggers it on factual questions and ends up answering "no" when it shouldn't.
#   The cleaner choice is to let the model extract yes/no verbatim from the context if
#   the question is genuinely yes/no.
# - "unknown" is the explicit fallback when no context supports the answer. The
#   downstream pipeline (clean_answer, align_to_context, run_eval category) already
#   understands this token, so the policy can learn to reject honestly.
SYSTEM_PROMPT = """You are a precise question-answering assistant for multi-hop QA.

You will be given several context paragraphs from Wikipedia, each formatted as:
  [i] Title: <title>
      Content: <text>

Task:
- Read the paragraphs and answer the question using ONLY information from them.
- Output the SHORTEST correct answer span, copied verbatim from the text (a name, number, date, or short phrase).
- If the paragraphs do NOT contain enough information to answer, output exactly: unknown
- Do NOT use outside knowledge. Do NOT guess.

Output rules:
- Output only the answer — no preamble, no explanation, no "Answer:" prefix, no quotes, no trailing punctuation.
- Give one final answer."""


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

    # Strip a single trailing period only when the last word before it has
    # 4+ letters — that's the heuristic separating real sentence ends
    # ("today." -> "today", "Paris." -> "Paris") from abbreviations
    # ("U.S." stays "U.S.", "St." stays "St.", "Mt." stays "Mt.").
    # The 4-letter cutoff is conservative: the longest common abbrev
    # (e.g. "Mrs.", "etc.") is 3 letters.
    m = re.search(r"(\w+)\.$", text)
    if m and len(m.group(1)) >= 4:
        text = text[: m.start(1) + len(m.group(1))]
    return text.strip()


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
        # Try window sizes around the answer length to tolerate minor length
        # diffs. Wider range (n-2 to n+2) absorbs paraphrased answers that
        # pick up extra tokens, e.g. "David Seville (born 1915)" -> "David Seville".
        for win in {max(1, n - 2), max(1, n - 1), n, n + 1, n + 2}:
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
    # The parenthetical reinforces SYSTEM_PROMPT's two key directives (shortest
    # span; "unknown" fallback). "Answer:" stays at the end so the model
    # continues in the conventional completion style; clean_answer() strips any
    # "Answer:" prefix the model may echo back.
    return f"""Context:
{context_text}

Question: {question}

Answer (shortest span from the context, or 'unknown' if not present):"""


DIRECT_PROMPT = """Answer the following question concisely and accurately using only your own knowledge.

Question: {question}

Answer:"""


def direct_answer(llm: ChatLLM, question: str) -> str:
    """Answer a question directly without any retrieval context (Wo-RAG baseline)."""
    prompt = DIRECT_PROMPT.format(question=question)
    answer = llm.generate(prompt).strip()
    answer = clean_answer(answer)
    return answer


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
