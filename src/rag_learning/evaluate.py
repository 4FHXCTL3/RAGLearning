from __future__ import annotations

from collections import Counter
import re
import string
import unicodedata


def normalize_answer(text: str) -> str:
    """Lower text, strip diacritics, and remove punctuation, articles, and extra whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    # NFKD decomposes accented characters into base + combining mark, so
    # "Brontë" / "Café" / "naïve" collapse to "Bronte" / "Cafe" / "naive"
    # and match gold answers that may use the unaccented form (and vice versa).
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(prediction: str, golds: list[str]) -> tuple[float, float]:
    """Max-over-golds EM and F1 (standard for NQ/PopQA/TriviaQA).

    These datasets accept multiple gold answers / aliases; a prediction is
    scored against the gold it matches best.
    """
    if not golds:
        return 0.0, 0.0
    return (
        max(exact_match_score(prediction, g) for g in golds),
        max(f1_score(prediction, g) for g in golds),
    )


def context_recall(
    gold_titles: set[str],
    gold_answers: list[str],
    retrieved_titles: set[str],
    retrieved_text: str,
) -> tuple[float, bool, str]:
    """Step 0 retrieval quality: did the top-k context contain the evidence
    needed to answer the question?

    Two flavors, chosen by what the dataset provides:

    - **Title-based** (multi-hop datasets with ``supporting_facts``, e.g.
      HotpotQA / 2WikiMultihopQA / MuSiQue): fraction of gold supporting-
      paragraph titles present in ``retrieved_titles``. ``full_hit`` is True
      iff every gold title was retrieved.

    - **Answer-in-context** (open-domain datasets with no supporting facts,
      e.g. TriviaQA / PopQA / NQ-Swap): binary — at least one gold answer (or
      alias) appears as a case-insensitive substring in the joined
      ``retrieved_text``. ``full_hit`` is then equivalent to ``recall``
      (always 0 or 1). For NQ-Swap the corpus is deliberately unrelated, so
      this will be 0 by design — that is the swap test scenario.

    Args:
        gold_titles: normalized gold supporting-paragraph titles
            (empty for open-domain datasets).
        gold_answers: gold answer strings (with aliases for TriviaQA/PopQA).
        retrieved_titles: normalized titles of the retrieved top-k paragraphs.
        retrieved_text: concatenated text of the retrieved top-k paragraphs,
            used for the answer-in-context case.

    Returns:
        ``(recall, full_hit, evidence_kind)`` where ``evidence_kind`` is
        ``"title"`` or ``"answer"``.
    """
    if gold_titles:
        hit = gold_titles & retrieved_titles
        recall = len(hit) / len(gold_titles) if gold_titles else 0.0
        full_hit = gold_titles.issubset(retrieved_titles)
        return recall, full_hit, "title"

    norm_text = (retrieved_text or "").lower()
    found = any(
        (a or "").strip() and (a or "").strip().lower() in norm_text
        for a in gold_answers
    )
    recall = 1.0 if found else 0.0
    full_hit = found
    return recall, full_hit, "answer"
