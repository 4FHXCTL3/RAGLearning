from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

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


def _semantic_entropy(answer_vecs: np.ndarray, threshold: float = 0.85) -> float:
    """Approximate semantic entropy over a group of sampled answers.

    Cluster answers greedily by embedding cosine similarity (a cluster = one
    "meaning"), then take the normalized Shannon entropy of the cluster-size
    distribution. All answers agreeing -> 1 cluster -> entropy 0 (confident);
    every answer differing -> many clusters -> entropy -> 1 (uncertain).

    Vectors are passed in pre-embedded (caller already embedded the answer
    samples, so we just reuse them). The clustering uses a single precomputed
    pairwise-cosine matrix indexed by cluster — the previous implementation
    did one ``cosine_similarity`` call per pair, the intermediate version
    rebuilt a ``(k, d)`` reps stack every iteration. With both the matrix
    and the per-iteration index list in place, the per-call cost is dominated
    by the single ``(n, n)`` matmul, regardless of how many clusters form.
    """
    n = answer_vecs.shape[0]
    if n <= 1:
        return 0.0

    # Full pairwise cosine matrix, computed once (vectors are L2-normalized
    # by the Embedder). For n=group_size=3 this is a 3x3 matmul; for n=2 it's
    # 2x2. Cheap enough to never need a fallback.
    sims = answer_vecs @ answer_vecs.T
    counts: list[int] = []
    cluster_src: list[int] = []  # row index in `sims` of each cluster's rep
    for i in range(n):
        if cluster_src:
            # Index into the precomputed matrix instead of re-allocating a
            # (k, d) reps stack on every iteration. `sims[i, cluster_src]`
            # gives cosines to all current cluster reps in one slice op.
            sim_to_clusters = sims[i, cluster_src]
            # `argmax` on a bool array returns the first True (since True
            # sorts above False), which matches the original loop's
            # order-preserving short-circuit.
            joined = sim_to_clusters >= threshold
            if joined.any():
                idx = int(np.argmax(joined))
                counts[idx] += 1
                continue
        cluster_src.append(i)
        counts.append(1)

    if len(counts) <= 1:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / sum(counts)
    entropy = -np.sum(probs * np.log(probs))
    # Normalize by log(n) so the value lands in [0, 1] regardless of group size.
    return float(entropy / np.log(n))


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

    Both vectors are L2-normalized upstream, so cosine = dot product (avoids
    the previous `np.linalg.norm(a) * np.linalg.norm(b)` round-trip).
    """
    sim = float(np.dot(query_vec, answer_vec))
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def _info_gain(new_vectors: np.ndarray, prev_vectors: np.ndarray | None) -> float:
    """Novelty of newly retrieved docs vs the previous round's docs.

    For each new doc, distance to its nearest previous doc; average over new
    docs. No previous docs (first retrieval) -> full novelty 1.0.

    Both inputs are already L2-normalized by the Embedder, so the cross-similarity
    matrix is just a single matmul (skips the per-call re-normalization that the
    previous version did).
    """
    if prev_vectors is None or prev_vectors.shape[0] == 0:
        return 1.0
    if new_vectors.shape[0] == 0:
        return 0.0
    sims = new_vectors @ prev_vectors.T
    novelty = 1.0 - sims.max(axis=1)
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
        # Guard update/normalize/save/load so the normalizer can be shared
        # across threads (e.g. parallel question processing in run_eval.py).
        # Cheap: the locked region is just a few float ops.
        self._lock = Lock()

    def update(self, state_vec: np.ndarray) -> None:
        """Update mean/var with one new sample (call after each episode)."""
        x = state_vec.astype(np.float64)
        with self._lock:
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
        with self._lock:
            mean = self.mean
            var = self.var
        return (state_vec.astype(np.float64) - mean) / np.sqrt(var + self.eps)

    def save(self) -> dict:
        with self._lock:
            return {
                "mean": self.mean.tolist(),
                "var": self.var.tolist(),
                "initialized": self.initialized,
            }

    def load(self, data: dict) -> None:
        with self._lock:
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

        # Embed the answer samples ONCE and reuse the vectors for both the
        # semantic-entropy clustering and the epistemic-uncertainty term. The
        # previous code embedded the answer samples again inside _semantic_entropy
        # AND embedded a representative answer for U_e — two redundant embed
        # round-trips per state. With ~200 questions * group_size=3 + 1
        # retrieve step = ~800 extra embed calls per epoch eliminated.
        cleaned = [a.strip() for a in answer_samples if a and a.strip()]
        if cleaned:
            answer_vecs = self.embedder.embed(cleaned)
            h_sem = _semantic_entropy(answer_vecs)
            # Reuse the first embedded answer as the representative answer —
            # it is already L2-normalized and is exactly the same vector
            # _semantic_entropy would have computed.
            rep_answer_vec = answer_vecs[0]
        else:
            h_sem = 0.0
            rep_answer_vec = None

        u_a = _data_uncertainty(doc_vectors)

        if rep_answer_vec is not None and doc_vectors.shape[0] > 0:
            query_vec = self.embedder.embed(query)[0]
            u_e = _epistemic_uncertainty(query_vec, rep_answer_vec)
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
