from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rag_learning.evaluate import exact_match_score, f1_score
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

    R = alpha*Racc - beta*Rcost + gamma*Rhonesty + lambda*delta_I
    Starting values; tune later (Phase E).
    """

    alpha: float = 1.0   # accuracy (EM/F1) weight
    beta: float = 0.1    # per-retrieval cost penalty
    gamma: float = 0.5   # honest-rejection reward
    lam: float = 0.2     # information-gain shaping


@dataclass
class StepResult:
    action: int
    reward: float
    done: bool
    answer: str | None
    state: State
    info: dict = field(default_factory=dict)


class RAGEnv:
    """One episode = answering one HotpotQA question with adaptive retrieval.

    The policy (module 2) observes a 4-dim State and chooses stop/retrieve/reject.
    The env executes the action against the retriever + generator (module 3) and
    returns a composite reward. Terminal actions: stop, reject, or hitting the
    retrieval budget.
    """

    def __init__(
        self,
        index: BM25Index,
        llm: ChatLLM,
        encoder: StateEncoder,
        weights: RewardWeights | None = None,
        initial_k: int = 5,
        k_increment: int = 5,
        max_retrievals: int = 2,
        group_size: int = 3,
        sample_temperature: float = 0.7,
    ):
        self.index = index
        self.llm = llm
        self.encoder = encoder
        self.weights = weights or RewardWeights()
        self.initial_k = initial_k
        self.k_increment = k_increment
        self.max_retrievals = max_retrievals
        self.group_size = group_size
        self.sample_temperature = sample_temperature

        # per-episode mutable state
        self.question: str = ""
        self.gold_answer: str = ""
        self.gold_titles: set[str] = set()
        self.current_k: int = initial_k
        self.retrieval_count: int = 0
        self.contexts: list = []
        self.prev_doc_texts: list[str] | None = None
        self._answer_samples: list[str] = []
        self._last_state: State | None = None

    # --- retrieval / generation helpers ---------------------------------

    def _retrieve(self, k: int) -> list:
        return [doc for doc, _score in self.index.search(self.question, top_k=k)]

    def _doc_texts(self, docs) -> list[str]:
        return [f"{d.title}. {d.text}" for d in docs]

    def _sample_answers(self, docs) -> list[str]:
        """Generate group_size answers; sample >0 temperature for diversity so
        semantic entropy (Hsem) is meaningful. First sample is greedy (temp 0)."""
        prompt = build_prompt(self.question, docs)
        samples: list[str] = []
        for i in range(self.group_size):
            temp = 0.0 if i == 0 else self.sample_temperature
            raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=temp)
            samples.append(clean_answer(raw.strip()))
        return samples

    def _observe(self, docs) -> State:
        self._answer_samples = self._sample_answers(docs)
        return self.encoder.encode(
            query=self.question,
            doc_texts=self._doc_texts(docs),
            answer_samples=self._answer_samples,
            prev_doc_texts=self.prev_doc_texts,
        )

    # --- gym-like API ----------------------------------------------------

    def reset(self, question: str, gold_answer: str, gold_titles: set[str] | None = None) -> State:
        self.question = question
        self.gold_answer = gold_answer
        self.gold_titles = gold_titles or set()
        self.current_k = self.initial_k
        self.retrieval_count = 1
        self.contexts = self._retrieve(self.current_k)
        self.prev_doc_texts = None
        state = self._observe(self.contexts)
        self.prev_doc_texts = self._doc_texts(self.contexts)
        self._last_state = state
        return state

    def _greedy_answer(self) -> str:
        # First sample is the temp-0 greedy answer; reuse it as the final answer.
        if self._answer_samples:
            return self._answer_samples[0]
        return clean_answer(
            self.llm.generate(
                build_prompt(self.question, self.contexts), system_prompt=SYSTEM_PROMPT
            ).strip()
        )

    def _gold_is_retrievable(self) -> bool:
        """Whether the gold supporting docs are present in current contexts.

        Used to judge honest rejection: abstaining is honest only when the
        evidence needed to answer isn't actually available.
        """
        if not self.gold_titles:
            return True  # unknown -> assume answerable, don't reward rejection
        import html
        retrieved = {html.unescape(d.title).strip().lower() for d in self.contexts}
        return self.gold_titles.issubset(retrieved)

    def step(self, action: int) -> StepResult:
        w = self.weights
        # Info-gain shaping uses the novelty of the currently observed docs.
        delta_i = self._last_state.delta_i if self._last_state is not None else 0.0

        if action == ACTION_STOP:
            answer = self._greedy_answer()
            em = exact_match_score(answer, self.gold_answer)
            f1 = f1_score(answer, self.gold_answer)
            racc = max(em, f1)  # reward the better of the two
            reward = w.alpha * racc - w.beta * self.retrieval_count + w.lam * delta_i
            state = self._last_state or self._observe(self.contexts)
            return StepResult(action, reward, True, answer, state,
                              {"em": em, "f1": f1, "racc": racc})

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
        # Info gain: novelty comes from the fresh state's delta_i.
        reward = -w.beta * self.retrieval_count + w.lam * state.delta_i
        self.contexts = new_docs
        self.prev_doc_texts = self._doc_texts(new_docs)
        self._last_state = state
        return StepResult(action, reward, False, None, state, {"k": self.current_k})
