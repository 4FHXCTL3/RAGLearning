"""Microbenchmark for training-loop hot path optimizations.

Compares the optimized state.py / policy.py / environment.py against a
faithful re-implementation of the previous (pre-optimization) versions,
inlined here so the bench runs the optimized and reference versions in
the same process with identical inputs. Network-bound pieces (real LLM /
embedder calls) are replaced with deterministic in-memory stand-ins so
the bench isolates the Python-level overhead the optimizations target.

Run: python scripts/_bench_train_speed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---- Constants ----
EMBED_DIM = 768
N_ITER = 5000
N_ITER_HEAVY = 2000


# ---- Reference (pre-optimization) implementations ------------------------
# Inline copies of the OLD code paths so the bench is apples-to-apples.

def ref_semantic_entropy(embedder, answer_samples, threshold: float = 0.85) -> float:
    """Old O(n*k) loop with single-pair cosine calls, re-embedding answers."""
    cleaned = [a.strip() for a in answer_samples if a and a.strip()]
    if len(cleaned) <= 1:
        return 0.0
    # Re-embed the answers INSIDE the function — this is the wasted work.
    vectors = embedder.embed(cleaned)
    cluster_reps: list = []
    counts: list[int] = []
    for vec in vectors:
        for i, rep in enumerate(cluster_reps):
            denom = np.linalg.norm(vec) * np.linalg.norm(rep)
            sim = float(np.dot(vec, rep) / denom) if denom else 0.0
            if sim >= threshold:
                counts[i] += 1
                break
        else:
            cluster_reps.append(vec)
            counts.append(1)
    if len(counts) <= 1:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / sum(counts)
    return float(-np.sum(probs * np.log(probs)) / np.log(len(cleaned)))


def ref_info_gain(new_vectors, prev_vectors) -> float:
    if prev_vectors is None or prev_vectors.shape[0] == 0:
        return 1.0
    if new_vectors.shape[0] == 0:
        return 0.0
    norms_n = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    norms_p = np.linalg.norm(prev_vectors, axis=1, keepdims=True)
    norms_n[norms_n == 0] = 1.0
    norms_p[norms_p == 0] = 1.0
    sims = (new_vectors / norms_n) @ (prev_vectors / norms_p).T
    novelty = 1.0 - sims.max(axis=1)
    return float(np.clip(np.mean(novelty), 0.0, 1.0))


def ref_epistemic_uncertainty(query_vec, answer_vec):
    denom = np.linalg.norm(query_vec) * np.linalg.norm(answer_vec)
    if denom == 0:
        return 1.0
    sim = float(np.dot(query_vec, answer_vec) / denom)
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def ref_gold_is_retrievable(gold_titles, contexts):
    import html
    retrieved = {html.unescape(d.title).strip().lower() for d in contexts}
    return gold_titles.issubset(retrieved)


def ref_sample_action_2tuple(policy, state, rng):
    """Old sample_action returning (action, prob) without cache."""
    probs = policy.action_probs(state)
    action = int(rng.choice(3, p=probs))
    return action, float(probs[action])


def ref_grpo_update_single_forward(policy, samples, advantages, lr, clip_eps):
    """Old grpo_update: re-runs forward per sample, single-return objective."""
    grads = {k: np.zeros_like(v) for k, v in policy._params().items()}
    total_obj = 0.0
    n = len(samples)
    if n == 0:
        return 0.0
    for sample, adv in zip(samples, advantages):
        cache = policy._forward_cache(sample["state"])  # no cache reuse
        new_prob = cache["probs"][sample["action"]]
        old_prob = max(sample["old_prob"], 1e-8)
        ratio = new_prob / old_prob
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        total_obj += min(unclipped, clipped)
        if unclipped <= clipped:
            glp = policy._grad_log_prob(cache, sample["action"])
            scale = ratio * adv
            for k in grads:
                grads[k] += scale * glp[k]
    for k, param in policy._params().items():
        param += lr * grads[k] / n
    return float(total_obj / n)


# ---- Bench runner --------------------------------------------------------

def bench(label: str, fn, n_iter: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    elapsed = time.perf_counter() - t0
    per_call = elapsed / n_iter * 1000  # ms
    print(f"  {label:42s}  {n_iter:>5d}  total={elapsed*1000:>7.2f}ms  per={per_call:>7.4f}ms")
    return per_call


def speedup(ref_t: float, opt_t: float, label: str) -> None:
    print(f"     -> {label}: {ref_t / opt_t:.2f}x  ({(1 - opt_t / ref_t) * 100:+.1f}%)")


def main() -> None:
    print("=" * 78)
    print("Training-loop hot path microbench (CPU only, no LLM / embed I/O)")
    print("=" * 78)

    rng = np.random.default_rng(42)
    n_answer = 3
    n_docs = 5
    n_prev = 5
    answer_vecs = rng.normal(0, 1, (n_answer, EMBED_DIM)).astype(np.float32)
    answer_vecs /= np.linalg.norm(answer_vecs, axis=1, keepdims=True) + 1e-12
    doc_vecs = rng.normal(0, 1, (n_docs, EMBED_DIM)).astype(np.float32)
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12
    prev_vecs = rng.normal(0, 1, (n_prev, EMBED_DIM)).astype(np.float32)
    prev_vecs /= np.linalg.norm(prev_vecs, axis=1, keepdims=True) + 1e-12
    query_vec = rng.normal(0, 1, EMBED_DIM).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-12

    # ---- [1] _semantic_entropy ----
    print("\n[1] _semantic_entropy (3 answer vectors)")
    from rag_learning.state import _semantic_entropy as opt_se

    fake_emb = _FakeEmbedder()
    fake_emb._cache = {a: answer_vecs[i] for i, a in enumerate(["a", "b", "c"])}
    ref_t = bench("ref (per-pair cosine loop)",
                  lambda: ref_semantic_entropy(fake_emb, ["a", "b", "c"]), N_ITER)
    opt_t = bench("opt (precomputed sims, index reuse)", lambda: opt_se(answer_vecs), N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [2] _info_gain ----
    print("\n[2] _info_gain (5 new vs 5 prev docs)")
    from rag_learning.state import _info_gain as opt_ig
    ref_t = bench("ref (re-normalizes inputs)", lambda: ref_info_gain(doc_vecs, prev_vecs), N_ITER)
    opt_t = bench("opt (uses pre-normalized)", lambda: opt_ig(doc_vecs, prev_vecs), N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [3] _epistemic_uncertainty ----
    print("\n[3] _epistemic_uncertainty (single pair)")
    from rag_learning.state import _epistemic_uncertainty as opt_eu
    ref_t = bench("ref (norm round-trip)", lambda: ref_epistemic_uncertainty(query_vec, answer_vecs[0]), N_ITER)
    opt_t = bench("opt (single dot product)", lambda: opt_eu(query_vec, answer_vecs[0]), N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [4] _gold_is_retrievable ----
    print("\n[4] _gold_is_retrievable (over 5 contexts)")
    gold_titles = {"title 0", "title 2"}
    contexts = [_FakeDoc(f"Title {i}", f"text {i}") for i in range(5)]
    ref_t = bench("ref (html.unescape per title)", lambda: ref_gold_is_retrievable(gold_titles, contexts), N_ITER)
    def opt_gold_is_retrievable():
        retrieved = {d.title.strip().lower() for d in contexts}
        return gold_titles.issubset(retrieved)
    opt_t = bench("opt (no html.unescape)", opt_gold_is_retrievable, N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [5] params_snapshot elimination ----
    print("\n[5] params_snapshot deep-copy vs in-update tracking")
    from rag_learning.policy import PolicyNetwork

    def reset_policy(p):
        p.w1 = rng.normal(0, 0.1, p.w1.shape)
        p.b1 = np.zeros_like(p.b1)
        p.w2 = rng.normal(0, 0.1, p.w2.shape)
        p.b2 = np.zeros_like(p.b2)

    pol_ref = PolicyNetwork(seed=0)
    pol_opt = PolicyNetwork(seed=0)
    state = rng.normal(0, 1, 4).astype(np.float64)
    samples = [{"state": state, "action": 1, "old_prob": 0.5, "cache": None}] * 3
    advantages = np.array([0.1, -0.2, 0.3])

    def ref_step_with_snapshot():
        reset_policy(pol_ref)
        snap_before = {k: v.copy() for k, v in pol_ref._params().items()}
        pol_ref.grpo_update(samples, advantages, lr=0.05, clip_eps=0.2)
        snap_after = {k: v.copy() for k, v in pol_ref._params().items()}
        max(float(np.abs(snap_after[k] - snap_before[k]).max()) for k in snap_after)

    def opt_step_no_snapshot():
        reset_policy(pol_opt)
        pol_opt.grpo_update(samples, advantages, lr=0.05, clip_eps=0.2)

    ref_t = bench("ref (2 deep-copies + update)", ref_step_with_snapshot, N_ITER_HEAVY)
    opt_t = bench("opt (update only, delta in-line)", opt_step_no_snapshot, N_ITER_HEAVY)
    speedup(ref_t, opt_t, "speedup")

    # ---- [6] sample_action + grpo_update double forward ----
    print("\n[6] sample_action + grpo_update (recompute forward vs cache reuse)")
    pol_ref2 = PolicyNetwork(seed=0)
    pol_opt2 = PolicyNetwork(seed=0)
    reset_policy(pol_ref2)
    reset_policy(pol_opt2)
    rng2 = np.random.default_rng(1)

    def ref_action_then_update():
        action, old_prob = ref_sample_action_2tuple(pol_ref2, state, rng2)
        s = {"state": state, "action": action, "old_prob": old_prob}
        ref_grpo_update_single_forward(pol_ref2, [s], np.array([0.1]), 0.05, 0.2)

    def opt_action_then_update():
        action, old_prob, cache = pol_opt2.sample_action(state, rng2)
        s = {"state": state, "action": action, "old_prob": old_prob, "cache": cache}
        pol_opt2.grpo_update([s], np.array([0.1]), lr=0.05, clip_eps=0.2)

    ref_t = bench("ref (2 forward passes per step)", ref_action_then_update, N_ITER)
    opt_t = bench("opt (1 forward, cache reuse)", opt_action_then_update, N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [6b] GROUP-LEVEL forward sharing (3 group members, 1 forward) ----
    print("\n[6b] Per-group forward sharing (3 samples -> 1 forward vs 3 forwards)")
    pol_ref3 = PolicyNetwork(seed=0)
    pol_opt3 = PolicyNetwork(seed=0)
    reset_policy(pol_ref3)
    reset_policy(pol_opt3)
    rng3 = np.random.default_rng(2)

    def ref_group_of_3():
        # OLD: each of 3 group members calls sample_action -> 3 forwards
        # then grpo_update re-runs forward per sample -> 3 more forwards = 6 total
        samples = []
        for _ in range(3):
            action, old_prob = ref_sample_action_2tuple(pol_ref3, state, rng3)
            samples.append({"state": state, "action": action, "old_prob": old_prob})
        ref_grpo_update_single_forward(
            pol_ref3, samples, np.array([0.1, -0.2, 0.3]), 0.05, 0.2,
        )

    def opt_group_of_3():
        # NEW: ONE forward for the shared state, 3 samples drawn from same probs,
        # shared cache passed to grpo_update.
        shared_cache = pol_opt3._forward_cache(state)
        shared_probs = shared_cache["probs"]
        samples = []
        for _ in range(3):
            action, old_prob = pol_opt3.sample_action_from_probs(shared_probs, rng3)
            samples.append({
                "state": state, "action": action, "old_prob": old_prob,
                "cache": shared_cache,
            })
        pol_opt3.grpo_update(samples, np.array([0.1, -0.2, 0.3]), lr=0.05, clip_eps=0.2)

    ref_t = bench("ref (6 forwards per group of 3)", ref_group_of_3, N_ITER)
    opt_t = bench("opt (1 forward per group of 3)", opt_group_of_3, N_ITER)
    speedup(ref_t, opt_t, "speedup")

    # ---- [7] Full state encode (with mock embedder) ----
    print("\n[7] Full StateEncoder.encode (5 docs, 3 answers)")

    from rag_learning.state import StateEncoder, _data_uncertainty

    def ref_encode(emb, doc_texts, answer_samples, query):
        """Pre-optimization encode: includes _data_uncertainty for fairness."""
        doc_vectors = emb.embed(doc_texts)
        prev_vectors = None
        # _semantic_entropy re-embeds answer_samples internally
        ref_semantic_entropy(emb, answer_samples)
        _data_uncertainty(doc_vectors)  # old code also calls this
        rep_answer = next((a for a in answer_samples if a and a.strip()), "")
        if rep_answer and doc_texts:
            emb.embed(query)        # cached but still a dict lookup
            emb.embed(rep_answer)   # redundant: already in answer_samples
        ref_info_gain(doc_vectors, prev_vectors)
        return doc_vectors

    def opt_encode(encoder, doc_texts, answer_samples, query):
        return encoder.encode(query=query, doc_texts=doc_texts, answer_samples=answer_samples)

    emb_ref = _FakeEmbedder(seed=7)
    enc_opt = StateEncoder(embedder=_FakeEmbedder(seed=7))
    doc_texts = [f"doc {i}" for i in range(n_docs)]
    answer_samples = ["Paris", "London", "Paris"]
    query = "What is the capital of France?"

    ref_t = bench("ref (3 extra embed calls per encode)",
                  lambda: ref_encode(emb_ref, doc_texts, answer_samples, query), N_ITER_HEAVY)
    opt_t = bench("opt (zero extra embed calls)",
                  lambda: opt_encode(enc_opt, doc_texts, answer_samples, query), N_ITER_HEAVY)
    speedup(ref_t, opt_t, "speedup")

    print("\n" + "=" * 78)
    print("Bench complete. All numbers are wall-time per call (lower = faster).")
    print("=" * 78)


# ---- Tiny stand-ins for the bench ----------------------------------------

class _FakeEmbedder:
    def __init__(self, dim: int = EMBED_DIM, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def embed(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = []
        for t in items:
            if t not in self._cache:
                v = self.rng.normal(0, 1, self.dim).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                self._cache[t] = v
            out.append(self._cache[t])
        return np.stack(out)


class _FakeDoc:
    def __init__(self, title, text):
        self.title = title
        self.text = text


if __name__ == "__main__":
    main()
