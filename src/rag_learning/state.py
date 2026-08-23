from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_learning.embed import Embedder, cosine_similarity, pairwise_cosine


@dataclass(frozen=True)
class State:
    """The 4-dim state vector St = [Hsem, Ua, Ue, delta_I] (framework module 1).

    All components are kept in a roughly 0..1 range so the policy MLP
    (framework module 2) sees comparable input scales.
    """

    h_sem: float   # semantic entropy of answer samples (answer uncertainty)
    u_a: float     # data uncertainty: conflict among retrieved docs
    u_e: float     # epistemic uncertainty: query<->answer semantic distance
    delta_i: float # marginal info gain: novelty of new docs vs previous docs

    def to_vector(self) -> np.ndarray:
        return np.array([self.h_sem, self.u_a, self.u_e, self.delta_i], dtype=np.float32)


def _semantic_entropy(embedder: Embedder, answers: list[str], threshold: float = 0.85) -> float:
    """Approximate semantic entropy over a group of sampled answers.

    Cluster answers greedily by embedding cosine similarity (a cluster = one
    "meaning"), then take the normalized Shannon entropy of the cluster-size
    distribution. All answers agreeing -> 1 cluster -> entropy 0 (confident);
    every answer differing -> many clusters -> entropy -> 1 (uncertain).
    """
    cleaned = [a.strip() for a in answers if a and a.strip()]
    if len(cleaned) <= 1:
        return 0.0

    vectors = embedder.embed(cleaned)
    cluster_reps: list[np.ndarray] = []
    counts: list[int] = []
    for vec in vectors:
        for i, rep in enumerate(cluster_reps):
            if cosine_similarity(vec, rep) >= threshold:
                counts[i] += 1
                break
        else:
            cluster_reps.append(vec)
            counts.append(1)

    if len(counts) <= 1:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / sum(counts)
    entropy = -np.sum(probs * np.log(probs))
    # Normalize by log(n) so the value lands in [0, 1] regardless of group size.
    return float(entropy / np.log(len(cleaned)))


def _data_uncertainty(doc_vectors: np.ndarray) -> float:
    """Conflict/dispersion among retrieved docs, in [0, 1].

    framework describes this as the "variance of inter-document similarity", but
    with only 2 docs there is a single pairwise similarity and its variance is
    always 0 -- and multi-hop questions typically have exactly 2 gold docs, so
    variance would flatline. We instead use mean pairwise dissimilarity
    (1 - mean cosine): docs all pointing the same way -> ~0 (coherent evidence),
    docs unrelated/conflicting -> larger. This degrades gracefully to the 2-doc
    case while still capturing dispersion when there are more docs.
    """
    if doc_vectors.shape[0] < 2:
        return 0.0
    sims = pairwise_cosine(doc_vectors)
    iu = np.triu_indices(sims.shape[0], k=1)
    off_diag = sims[iu]
    if off_diag.size == 0:
        return 0.0
    return float(np.clip(1.0 - np.mean(off_diag), 0.0, 1.0))


def _epistemic_uncertainty(query_vec: np.ndarray, answer_vec: np.ndarray) -> float:
    """Query<->answer semantic distance mapped to [0, 1].

    Cosine distance = 1 - cosine similarity, clipped. Large distance means the
    answer is semantically far from the question -> model likely "doesn't know".
    """
    sim = cosine_similarity(query_vec, answer_vec)
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def _info_gain(new_vectors: np.ndarray, prev_vectors: np.ndarray | None) -> float:
    """Novelty of newly retrieved docs vs the previous round's docs.

    For each new doc, distance to its nearest previous doc; average over new
    docs. No previous docs (first retrieval) -> full novelty 1.0.
    """
    if prev_vectors is None or prev_vectors.shape[0] == 0:
        return 1.0
    if new_vectors.shape[0] == 0:
        return 0.0
    # cross cosine-sim matrix (new x prev); novelty = 1 - max sim to any prev doc
    norms_n = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    norms_p = np.linalg.norm(prev_vectors, axis=1, keepdims=True)
    norms_n[norms_n == 0] = 1.0
    norms_p[norms_p == 0] = 1.0
    sims = (new_vectors / norms_n) @ (prev_vectors / norms_p).T
    max_sim_per_new = sims.max(axis=1)
    novelty = 1.0 - max_sim_per_new
    return float(np.clip(np.mean(novelty), 0.0, 1.0))


class StateNormalizer:
    """Per-component running z-score for the 4-dim state.

    Each component (Hsem, Ua, Ue, delta_I) has its own mean/var tracked
    via EMA. The state is transformed to (state - mean) / sqrt(var + eps),
    so the policy sees "where in the training distribution is this state?"
    rather than absolute values. This makes the policy invariant to LLM
    calibration (a confident model has low raw Hsem, a hesitant one has
    high raw Hsem — both map to similar normalized positions).

    The normalizer is meant to be fitted during training and frozen for
    eval. Calling `update` continuously during eval would shift the
    normalization under the policy, which is unstable.
    """

    def __init__(self, alpha: float = 0.9, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps
        self.mean = np.zeros(4, dtype=np.float64)
        self.var = np.ones(4, dtype=np.float64)
        self.initialized = False

    def update(self, state_vec: np.ndarray) -> None:
        """Update mean/var with one new sample (call after each episode)."""
        x = state_vec.astype(np.float64)
        if not self.initialized:
            self.mean = x.copy()
            self.var = np.ones_like(x)
            self.initialized = True
            return
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        # EMA of squared deviation from the OLD mean. Strictly speaking this
        # is biased, but for alpha=0.9 it converges within ~50 samples which
        # is well within the warmup window.
        self.var = self.alpha * self.var + (1.0 - self.alpha) * (x - self.mean) ** 2

    def normalize(self, state_vec: np.ndarray) -> np.ndarray:
        """Return (state - mean) / sqrt(var + eps)."""
        return (state_vec.astype(np.float64) - self.mean) / np.sqrt(self.var + self.eps)

    def save(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "initialized": self.initialized,
        }

    def load(self, data: dict) -> None:
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.var = np.asarray(data["var"], dtype=np.float64)
        self.initialized = bool(data.get("initialized", True))


class StateEncoder:
    """Builds a State vector from a query, retrieved docs, and answer samples.

    Reuses one Embedder (with its cache) across calls so repeated docs/queries
    during a training episode are embedded once.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        normalizer: "StateNormalizer | None" = None,
    ):
        self.embedder = embedder or Embedder()
        # Optional running-z-score normalizer. When set, encode() updates its
        # stats with the raw state and returns a normalized State. Crucial
        # for cross-LLM transfer: Hsem is LLM-conditional (e.g. llama3:8b
        # produces Hsem ~0.2-0.4, llama3.2:3b ~0.4-0.6), so a policy trained
        # on raw Hsem from one LLM mis-classifies the other. Normalized
        # Hsem collapses these distributions into a common scale.
        self.normalizer = normalizer

    def encode(
        self,
        query: str,
        doc_texts: list[str],
        answer_samples: list[str],
        prev_doc_texts: list[str] | None = None,
    ) -> State:
        doc_vectors = self.embedder.embed(doc_texts) if doc_texts else np.empty((0, 0), dtype=np.float32)
        prev_vectors = (
            self.embedder.embed(prev_doc_texts)
            if prev_doc_texts
            else None
        )

        h_sem = _semantic_entropy(self.embedder, answer_samples)
        u_a = _data_uncertainty(doc_vectors)

        # Use the first non-empty answer sample as the representative answer.
        rep_answer = next((a for a in answer_samples if a and a.strip()), "")
        if rep_answer and doc_texts:
            query_vec = self.embedder.embed(query)[0]
            answer_vec = self.embedder.embed(rep_answer)[0]
            u_e = _epistemic_uncertainty(query_vec, answer_vec)
        else:
            u_e = 1.0  # no answer yet -> maximal epistemic uncertainty

        delta_i = _info_gain(doc_vectors, prev_vectors)

        state = State(h_sem=h_sem, u_a=u_a, u_e=u_e, delta_i=delta_i)

        # Optional normalization (frozen or fitted, depending on caller).
        if self.normalizer is not None:
            raw_vec = state.to_vector()
            self.normalizer.update(raw_vec)
            norm_vec = self.normalizer.normalize(raw_vec)
            # Reconstruct State with normalized components. `State` is a
            # frozen dataclass so we have to build a new instance.
            state = State(
                h_sem=float(norm_vec[0]),
                u_a=float(norm_vec[1]),
                u_e=float(norm_vec[2]),
                delta_i=float(norm_vec[3]),
            )

        return state
