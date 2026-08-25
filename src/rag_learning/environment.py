from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rag_learning.evaluate import compute_em_f1
from rag_learning.index import BM25Index
from rag_learning.llm import ChatLLM
from rag_learning.rag import SYSTEM_PROMPT, build_prompt, clean_answer
from rag_learning.state import State, StateEncoder

# Action space (framework module 2, simplified 3-action version).
ACTION_STOP = 0      # generate the final answer from current context
ACTION_RETRIEVE = 1  # widen retrieval, gather more docs, re-observe
ACTION_REJECT = 2    # actively abstain ("unknown")
NUM_ACTIONS = 3
ACTION_NAMES = {ACTION_STOP: "stop", ACTION_RETRIEVE: "retrieve", ACTION_REJECT: "reject"}


@dataclass
class RewardWeights:
    """Fixed weights for the composite reward (framework module 3).

    R = alpha*Racc - beta*Rcost + gamma*Rhonesty + lambda*delta_I + rho*Rretrieval

    Tuned for weak base models (llama3.2:3b) on multi-hop QA:
    - Use F1 (not EM threshold) for fine-grained accuracy signal
    - Add retrieval bonus when gold supporting docs are present
    - Lower beta/gamma to reduce action-collapse into rejection
    """

    alpha: float = 1.0      # accuracy (F1) weight
    beta: float = 0.05      # per-retrieval cost penalty (reduced from 0.1)
    gamma: float = 0.3      # honest-rejection reward (reduced from 0.5)
    lam: float = 0.3        # information-gain shaping (increased from 0.2)
    rho: float = 0.2        # retrieval relevance bonus (new)


@dataclass
class StepResult:
    action: int
    reward: float
    done: bool
    answer: str | None
    state: State
    info: dict = field(default_factory=dict)


class RAGEnv:
    """One episode = answering one multi-hop QA question with adaptive retrieval.

    The policy (module 2) observes a 4-dim State and chooses stop/retrieve/reject.
    The env executes the action against the retriever + generator (module 3) and
    returns a composite reward. Terminal actions: stop, reject, or hitting the
    retrieval budget.
    """

    def __init__(
        self,
        index: BM25Index,
        llm: ChatLLM | None = None,
        encoder: StateEncoder | None = None,
        weights: RewardWeights | None = None,
        initial_k: int = 5,
        k_increment: int = 5,
        max_retrievals: int = 2,
        group_size: int = 3,
        sample_temperature: float = 0.7,
        retrieval_mode: str = "bm25",
        embedder: "Embedder | None" = None,
        llms: list[ChatLLM] | None = None,
    ):
        if encoder is None:
            raise ValueError("RAGEnv requires a `encoder` (StateEncoder) — pass an instance explicitly.")
        if llm is None and llms is None:
            raise ValueError("RAGEnv requires at least one of `llm` or `llms`.")
        # Normalize to a list for uniform handling.
        if llms is not None:
            self.llms: list[ChatLLM] = list(llms)
        else:
            self.llms = [llm]  # type: ignore[list-item]
        self.llm = self.llms[0]  # backward-compat alias; may be stale between resets
        self._current_llm: ChatLLM = self.llms[0]

        self.index = index
        self.encoder = encoder
        self.weights = weights or RewardWeights()
        self.initial_k = initial_k
        self.k_increment = k_increment
        self.max_retrievals = max_retrievals
        self.group_size = group_size
        self.sample_temperature = sample_temperature
        # retrieval_mode ∈ {"bm25", "dense", "hybrid"}. "hybrid" requires the
        # index to be a HybridIndex; "dense" requires a DenseIndex. We don't
        # validate here — _retrieve() picks the right path at call time and
        # raises a clear error if the index type doesn't match the mode.
        self.retrieval_mode = retrieval_mode
        self.embedder = embedder
        # Lazy import: avoid pulling in Embedder at module load.
        from rag_learning.embed import Embedder as _Embedder
        if self.embedder is None and retrieval_mode in {"dense", "hybrid"}:
            self.embedder = _Embedder()

        # per-episode mutable state
        self.question: str = ""
        self.gold_answers: list[str] = []
        self.gold_titles: set[str] = set()
        self.current_k: int = initial_k
        self.retrieval_count: int = 0
        self.contexts: list = []
        self.prev_doc_texts: list[str] | None = None
        self._answer_samples: list[str] = []
        self._last_state: State | None = None
        # Cache of (answer_samples, state) keyed by (question, current_k).
        # The State observation depends only on (question, retrieved docs),
        # so GRPO group members that visit the same (q, k) reuse the same
        # LLM-sampled answers and the same encoded state. The savings are
        # large: each cached hit skips `group_size` LLM calls — the dominant
        # cost in training. Cache is per-env-instance and is bounded by
        # (num_questions * max_retrievals), so memory is not a concern for
        # realistic training sizes.
        self._observe_cache: dict[tuple[str, int], dict] = {}

    # --- retrieval / generation helpers ---------------------------------

    def _retrieve(self, k: int) -> list:
        """Top-k retrieval using the configured `retrieval_mode`.

        bm25  → BM25Index.search (lexical)
        dense → DenseIndex.search (cosine over pre-computed embeddings)
        hybrid → HybridIndex.search with RRF fusion (BM25 + dense)
        """
        mode = self.retrieval_mode
        if mode == "bm25":
            return [doc for doc, _score in self.index.search(self.question, top_k=k)]
        if mode == "dense":
            from rag_learning.index import DenseIndex
            if not isinstance(self.index, DenseIndex):
                raise TypeError(
                    f"retrieval_mode='dense' requires a DenseIndex (got {type(self.index).__name__}). "
                    "Build the index with --retrieval-mode hybrid (which also produces a DenseIndex)."
                )
            qv = self.embedder.embed(self.question)[0]
            return [doc for doc, _score in self.index.search(qv, top_k=k)]
        if mode == "hybrid":
            from rag_learning.index import HybridIndex
            if not isinstance(self.index, HybridIndex):
                raise TypeError(
                    f"retrieval_mode='hybrid' requires a HybridIndex (got {type(self.index).__name__}). "
                    "Build the index with --retrieval-mode hybrid."
                )
            qv = self.embedder.embed(self.question)[0]
            return [doc for doc, _score in self.index.search(self.question, qv, top_k=k)]
        raise ValueError(
            f"Unknown retrieval_mode={mode!r}. Expected 'bm25', 'dense', or 'hybrid'."
        )

    def _doc_texts(self, docs) -> list[str]:
        return [f"{d.title}. {d.text}" for d in docs]

    def _sample_answers(self, docs) -> list[str]:
        """Generate group_size answers; sample >0 temperature for diversity so
        semantic entropy (Hsem) is meaningful. First sample is greedy (temp 0).

        All ``group_size`` LLM calls go out in parallel — the local Ollama server
        can batch sequence slots for the same model, and waiting for the
        previous call to finish before issuing the next is pure dead time on
        the client side. The OpenAI client is thread-safe; one client is
        shared across worker threads inside ``generate_batch``.
        """
        prompt = build_prompt(self.question, docs)
        system_prompts = [SYSTEM_PROMPT] * self.group_size
        # First call is greedy (temp 0); the rest use the configured sampling
        # temperature so the Hsem distribution is non-degenerate.
        temperatures = [0.0] + [self.sample_temperature] * (self.group_size - 1)
        raws = self._current_llm.generate_batch(
            [prompt] * self.group_size,
            system_prompts=system_prompts,
            temperatures=temperatures,
        )
        return [clean_answer(r.strip()) for r in raws]

    def _observe(self, docs) -> State:
        # The state at (question, current_k) is fully determined by the
        # question and the retrieved docs at that k — not by the policy's
        # action or the group member's index. Multiple GRPO group members
        # visit the same (q, k) and would otherwise each pay the LLM cost
        # of resampling. Reuse the first observation.
        cache_key = (self.question, self.current_k)
        cached = self._observe_cache.get(cache_key)
        if cached is not None:
            self._answer_samples = list(cached["answer_samples"])
            return cached["state"]
        self._answer_samples = self._sample_answers(docs)
        state = self.encoder.encode(
            query=self.question,
            doc_texts=self._doc_texts(docs),
            answer_samples=self._answer_samples,
            prev_doc_texts=self.prev_doc_texts,
        )
        # Cache by (q, k). On cache hits the StateNormalizer sees fewer
        # updates (once per (q, k) instead of group_size times); for an
        # EMA normaliser with alpha=0.9 this is well within its smoothing
        # window and the policy input distribution stays in the same range.
        self._observe_cache[cache_key] = {
            "answer_samples": list(self._answer_samples),
            "state": state,
        }
        return state

    # --- gym-like API ----------------------------------------------------

    def reset(
        self,
        question: str,
        gold_answers: list[str],
        gold_titles: set[str] | None = None,
        llm: "ChatLLM | None" = None,
    ) -> State:
        self.question = question
        self.gold_answers = gold_answers
        self.gold_titles = gold_titles or set()
        # Per-episode LLM override. If passed, that LLM is used for this
        # episode's Hsem sampling and final answer; otherwise the env's
        # currently-set LLM is used (defaults to the first of `llms`).
        if llm is not None:
            self._current_llm = llm
        self.current_k = self.initial_k
        self.retrieval_count = 1
        self.contexts = self._retrieve(self.current_k)
        self.prev_doc_texts = None
        state = self._observe(self.contexts)
        self.prev_doc_texts = self._doc_texts(self.contexts)
        self._last_state = state
        return state

    # --- snapshot / restore ---------------------------------------------
    # Used by the GRPO trainer to share the initial observation across all
    # group members. Each member has its own stochastic policy rollout (so
    # the first action / trajectory differs), but the state observation
    # (LLM samples, embeddings, Hsem) is fully determined by the question
    # and the retrieved docs — so the env is rewound to the same starting
    # point between group members instead of re-paying the LLM cost.

    def snapshot(self) -> dict:
        """Capture the env's per-episode mutable state for later restore()."""
        return {
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "gold_titles": set(self.gold_titles),
            "_current_llm": self._current_llm,
            "current_k": self.current_k,
            "retrieval_count": self.retrieval_count,
            "contexts": list(self.contexts),
            "prev_doc_texts": list(self.prev_doc_texts) if self.prev_doc_texts is not None else None,
            "_answer_samples": list(self._answer_samples),
            "_last_state": self._last_state,
        }

    def restore(self, snap: dict) -> None:
        """Rewind the env to a previously captured snapshot.

        Cheap: just attribute assignments. Does NOT touch ``_observe_cache``
        — cached (q, k) observations remain valid because they depend only
        on (question, current_k), not on env state.
        """
        self.question = snap["question"]
        self.gold_answers = list(snap["gold_answers"])
        self.gold_titles = set(snap["gold_titles"])
        self._current_llm = snap["_current_llm"]
        self.current_k = snap["current_k"]
        self.retrieval_count = snap["retrieval_count"]
        self.contexts = list(snap["contexts"])
        self.prev_doc_texts = list(snap["prev_doc_texts"]) if snap["prev_doc_texts"] is not None else None
        self._answer_samples = list(snap["_answer_samples"])
        self._last_state = snap["_last_state"]

    def _greedy_answer(self) -> str:
        # First sample is the temp-0 greedy answer; reuse it as the final answer.
        if self._answer_samples:
            return self._answer_samples[0]
        return clean_answer(
            self._current_llm.generate(
                build_prompt(self.question, self.contexts), system_prompt=SYSTEM_PROMPT
            ).strip()
        )

    def _gold_is_retrievable(self) -> bool:
        """Whether the gold supporting docs are present in current contexts.

        Used to judge honest rejection: abstaining is honest only when the
        evidence needed to answer isn't actually available.

        ``self.gold_titles`` is already lowercased + unescaped by
        ``data.gold_titles()`` (which runs ``norm_title``), and the index
        builder also unescapes retrieved titles. So comparing already-normalized
        strings avoids the per-call ``html.unescape`` round-trip — the previous
        version unescaped every retrieved title on every step (a no-op for
        modern HotpotQA but real cost on the hot path).
        """
        if not self.gold_titles:
            return True  # unknown -> assume answerable, don't reward rejection
        retrieved = {d.title.strip().lower() for d in self.contexts}
        return self.gold_titles.issubset(retrieved)

    def step(self, action: int) -> StepResult:
        w = self.weights
        # Info-gain shaping uses the novelty of the currently observed docs.
        delta_i = self._last_state.delta_i if self._last_state is not None else 0.0

        if action == ACTION_STOP:
            answer = self._greedy_answer()
            em, f1 = compute_em_f1(answer, self.gold_answers)
            # Use F1 (more fine-grained than EM) for accuracy reward.
            racc = f1
            # Retrieval relevance bonus: reward having the gold docs in context.
            r_retrieval = w.rho if self._gold_is_retrievable() else 0.0
            reward = w.alpha * racc - w.beta * self.retrieval_count + w.lam * delta_i + r_retrieval
            state = self._last_state or self._observe(self.contexts)
            return StepResult(action, reward, True, answer, state,
                              {"em": em, "f1": f1, "racc": racc, "r_retrieval": r_retrieval})

        if action == ACTION_REJECT:
            honest = not self._gold_is_retrievable()
            # Honest abstention: gold truly unavailable -> reward; wrongly
            # abstaining when the answer was retrievable -> penalize.
            rhonesty = 1.0 if honest else -1.0
            reward = w.gamma * rhonesty - w.beta * self.retrieval_count
            state = self._last_state or self._observe(self.contexts)
            return StepResult(action, reward, True, "unknown", state,
                              {"honest": honest, "rhonesty": rhonesty})

        # ACTION_RETRIEVE
        if self.retrieval_count >= self.max_retrievals:
            # Budget exhausted: force a stop instead of retrieving again.
            forced = self.step(ACTION_STOP)
            forced.info["forced_stop"] = True
            return forced

        self.current_k += self.k_increment
        self.retrieval_count += 1
        new_docs = self._retrieve(self.current_k)
        state = self._observe(new_docs)
        # Retrieval relevance bonus: immediate feedback for finding gold docs.
        r_retrieval = w.rho if self._gold_is_retrievable() else 0.0
        reward = -w.beta * self.retrieval_count + w.lam * state.delta_i + r_retrieval
        self.contexts = new_docs
        self.prev_doc_texts = self._doc_texts(new_docs)
        self._last_state = state
        return StepResult(action, reward, False, None, state, {"k": self.current_k, "r_retrieval": r_retrieval})
